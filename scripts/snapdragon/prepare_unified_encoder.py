"""Prepare the offline Unified EN encoder for QNN HTP from its ONNX export."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


LAYER_COUNT = 24
MODEL_WIDTH = 1024
MEL_BINS = 128
SUBSAMPLING = 8
QNN_PROVIDER = "QNNExecutionProvider"
SUBSAMPLER_MASKS = {
    "/pre_encode/conv/Mul_3": 0,
    "/pre_encode/conv/Mul_7": 1,
    "/pre_encode/conv/Mul_9": 1,
    "/pre_encode/conv/Mul_13": 2,
    "/pre_encode/conv/Mul_15": 2,
    "/pre_encode/conv/Mul_17": 2,
    "/pre_encode/conv/Mul_21": 3,
    "/pre_encode/conv/Mul_23": 3,
    "/pre_encode/conv/Mul_25": 3,
}


def layer_output(layer):
    return f"/layers.{layer}/norm_out/LayerNormalization_output_0"


def subsampled_frames(frames, factor=SUBSAMPLING):
    return (frames + factor - 1) // factor


def initializer_array(tensor, weights):
    if not tensor.external_data:
        return numpy_helper.to_array(tensor)
    storage = {entry.key: entry.value for entry in tensor.external_data}
    if storage["location"] != "encoder.int8.onnx.data":
        raise ValueError(f"Unexpected external data location for {tensor.name}")
    weights.seek(int(storage["offset"]))
    data = weights.read(int(storage["length"]))
    dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
    return np.frombuffer(data, dtype=dtype).reshape(tensor.dims)


def restored_initializer(original, quantized, weights):
    name = original.name
    if name in quantized:
        value = initializer_array(quantized[name], weights)
    else:
        # This release uses scalar per-tensor scales. Rebuild the float
        # weights for HTP FP16 instead of carrying dynamic integer ops into QNN.
        value = initializer_array(quantized[f"{name}_quantized"], weights).astype(np.float32)
        scale = initializer_array(quantized[f"{name}_scale"], weights)
        zero_point = initializer_array(quantized[f"{name}_zero_point"], weights)
        if scale.ndim or zero_point.ndim:
            raise ValueError(f"Expected scalar quantization parameters for {name}")
        value -= zero_point
        value *= scale
    if list(value.shape) != list(original.dims):
        raise ValueError(f"Restored dimensions disagree for {name}")
    return numpy_helper.from_array(value, name)


def stage_constants(mel_frames, valid_frames, start):
    constants = {"length": helper.make_tensor("length", TensorProto.INT64, [1], [valid_frames])}
    if start:
        encoder_frames = subsampled_frames(mel_frames)
        # The later blocks only use the shared subsampler's length and shape.
        # Supplying those constants keeps the subsampler in the first stage.
        for name in ("/Shape_output_0", "/pos_enc/Shape_output_0"):
            constants[name] = helper.make_tensor(
                name, TensorProto.INT64, [3], [1, encoder_frames, MODEL_WIDTH]
            )
        name = "/Cast_output_0"
        constants[name] = helper.make_tensor(name, TensorProto.INT64, [1], [subsampled_frames(valid_frames)])
    return constants


def length_mask_inputs(mel_frames, start):
    encoder_frames = subsampled_frames(mel_frames)
    inputs = [
        helper.make_tensor_value_info("attention_padding", TensorProto.BOOL, [1, 1, encoder_frames, encoder_frames]),
        helper.make_tensor_value_info("convolution_padding", TensorProto.BOOL, [1, 1, encoder_frames]),
    ]
    if not start:
        inputs.extend(
            helper.make_tensor_value_info(
                f"subsampling_valid_{i}", TensorProto.FLOAT, [1, 1, subsampled_frames(mel_frames, 2**i), 1]
            )
            for i in range(4)
        )
    return inputs


def rewire_mask(node):
    if node.name in SUBSAMPLER_MASKS:
        index, mask = 1, f"subsampling_valid_{SUBSAMPLER_MASKS[node.name]}"
    elif node.name.startswith("/layers.") and node.op_type == "Where":
        operation = node.name.split("/", 2)[2]
        index = 0
        if operation in ("self_attn/Where", "self_attn/Where_1"):
            mask = "attention_padding"
        elif operation == "conv/Where":
            mask = "convolution_padding"
        else:
            return node
    else:
        return node
    input_names = list(node.input)
    input_names[index] = mask
    rewired = onnx.NodeProto()
    rewired.CopyFrom(node)
    rewired.input[:] = input_names
    return rewired


def build_stage(source, quantized, weights, mel_frames, start, stop, valid_frames=None):
    encoder_frames = subsampled_frames(mel_frames)
    input_name = layer_output(start - 1) if start else "audio_signal"
    input_shape = [1, encoder_frames, MODEL_WIDTH] if start else [1, MEL_BINS, mel_frames]
    last = stop == LAYER_COUNT
    output = "outputs" if last else layer_output(stop - 1)
    output_shape = [1, MODEL_WIDTH, encoder_frames] if last else [1, encoder_frames, MODEL_WIDTH]
    constants = stage_constants(mel_frames, valid_frames if valid_frames is not None else mel_frames, start)
    mask_inputs = length_mask_inputs(mel_frames, start) if valid_frames is None else []
    inputs = [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape), *mask_inputs]
    boundaries = {value.name for value in inputs}
    producer = {name: i for i, node in enumerate(source.graph.node) for name in node.output}
    initializer = {value.name: value for value in source.graph.initializer}
    needed_nodes = {}
    needed_constants = set()
    pending = [output]
    while pending:
        name = pending.pop()
        if not name or name in boundaries:
            continue
        if name in constants or name in initializer:
            needed_constants.add(name)
            continue
        index = producer[name]
        if index not in needed_nodes:
            node = source.graph.node[index]
            if mask_inputs:
                node = rewire_mask(node)
            needed_nodes[index] = node
            pending.extend(node.input)
    stage_nodes = [needed_nodes[i] for i in sorted(needed_nodes)]
    if start and any(node.name.startswith("/pre_encode/") for node in stage_nodes):
        raise ValueError("An encoder stage unexpectedly depends on the subsampler")
    stage_initializers = [
        constants[name] if name in constants else restored_initializer(initializer[name], quantized, weights)
        for name in sorted(needed_constants)
    ]
    graph = helper.make_graph(
        stage_nodes,
        f"parakeet-unified-encoder-{start}-{stop}",
        inputs,
        [helper.make_tensor_value_info(output, TensorProto.FLOAT, output_shape)],
        initializer=stage_initializers,
    )
    return helper.make_model(graph, opset_imports=source.opset_import, ir_version=source.ir_version)


def fold_stage(source_path, destination):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(destination)
    ort.InferenceSession(str(source_path), sess_options=options, providers=["CPUExecutionProvider"])


class QnnContextCompiler:
    def __init__(self):
        import onnxruntime_qnn as qnn

        ort.register_execution_provider_library(QNN_PROVIDER, qnn.get_library_path())
        self.device = next((device for device in ort.get_ep_devices()
                            if device.ep_name == QNN_PROVIDER
                            and device.device.type == ort.OrtHardwareDeviceType.NPU), None)
        if self.device is None:
            raise RuntimeError("The Qualcomm QNN execution provider found no NPU")
        self.backend = qnn.get_qnn_htp_path()

    def export(self, source, destination):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_session_config_entry("ep.context_enable", "1")
        options.add_session_config_entry("ep.context_embed_mode", "0")
        options.add_session_config_entry("ep.context_file_path", str(destination))
        options.add_provider_for_devices([self.device], {
            "backend_path": self.backend,
            "enable_htp_fp16_precision": "1",
            "offload_graph_io_quantization": "0",
        })
        ort.InferenceSession(str(source), sess_options=options, enable_fallback=False)
        cached = onnx.load(destination, load_external_data=False)
        if not cached.graph.node or any(node.op_type != "EPContext" for node in cached.graph.node):
            raise RuntimeError(f"Expected a QNN context-only graph for {destination.name}")
        for node in cached.graph.node:
            attributes = {attribute.name: attribute.s for attribute in node.attribute}
            sidecar = attributes.get("ep_cache_context", b"").decode("utf-8")
            if attributes.get("source") != QNN_PROVIDER.encode() or not sidecar or Path(sidecar).name != sidecar:
                raise RuntimeError(f"Unexpected QNN context metadata for {destination.name}")
            if not destination.with_name(sidecar).is_file():
                raise RuntimeError(f"Missing QNN context data for {destination.name}")


def prepare(source_dir, output_dir, mel_frames, layers_per_stage, compile_contexts):
    source = onnx.load(source_dir / "offline_fp32_encoder.onnx", load_external_data=False)
    quantized_model = onnx.load(source_dir / "offline_int8_encoder.onnx", load_external_data=False)
    quantized = {value.name: value for value in quantized_model.graph.initializer}
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "encoder-manifest.json").exists():
        raise FileExistsError(f"Encoder already prepared in {output_dir}")
    compiler = QnnContextCompiler() if compile_contexts else None
    stages = []
    with (source_dir / "encoder.int8.onnx.data").open("rb") as weights:
        with tempfile.TemporaryDirectory(prefix="unified-encoder-", dir=output_dir) as temporary:
            scratch_dir = Path(temporary)
            unfolded = scratch_dir / "unfolded.onnx"
            folded = scratch_dir / "folded.onnx"
            for stage, start in enumerate(range(0, LAYER_COUNT, layers_per_stage)):
                stop = min(start + layers_per_stage, LAYER_COUNT)
                stage_path = scratch_dir / f"encoder-{stage:02}.onnx"
                model = build_stage(source, quantized, weights, mel_frames, start, stop)
                onnx.save_model(model, unfolded)
                del model
                fold_stage(unfolded, folded if compiler else stage_path)
                unfolded.unlink()
                if compiler:
                    compiler.export(folded, stage_path)
                    folded.unlink()
                stages.append(stage_path)
                print(f"Prepared encoder layers {start + 1} through {stop}", flush=True)
            manifest = scratch_dir / "encoder-manifest.json"
            manifest.write_text(json.dumps({
                "mel_frames": mel_frames,
                "encoder_frames": subsampled_frames(mel_frames),
                "length_masks": True,
                "compiled_contexts": compile_contexts,
                "stages": [path.name for path in stages],
            }, indent=2) + "\n", encoding="utf-8")
            for path in scratch_dir.iterdir():
                if path != manifest:
                    path.replace(output_dir / path.name)
            manifest.replace(output_dir / manifest.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mel-frames", default=1280, type=int, choices=range(8, 2401, 8), metavar="8..2400")
    parser.add_argument("--layers-per-stage", default=2, type=int, choices=range(1, 5), metavar="1..4")
    parser.add_argument("--compile-contexts", action="store_true")
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, args.mel_frames, args.layers_per_stage, args.compile_contexts)


if __name__ == "__main__":
    main()
