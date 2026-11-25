"""Prepare the fixed single-step Parakeet Unified RNNT decoder for QNN HTP."""

from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import onnx
from onnx import TensorProto, helper


SOURCE = "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d"
ASSETS = {
    "decoder_joint.onnx": "64648c91935ea4819e9c31ea8a20f011d3b1c960be3861cb8bc74df0259cd998",
    "vocab.txt": "ae61c9b743cb47c04ce6fb130444210646ed0f40c953c77ee1855185960af94f",
}
INPUTS = {
    "encoder_outputs": (1, 1024, 1),
    "targets": (1, 1),
    "input_states_1": (2, 1, 640),
    "input_states_2": (2, 1, 640),
}
OUTPUTS = {
    "outputs": (1, 1, 1, 1025),
    "output_states_1": (2, 1, 640),
    "output_states_2": (2, 1, 640),
}


def sha256(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def download_assets(directory):
    for name, expected in ASSETS.items():
        target = directory / name
        if target.is_file() and sha256(target) == expected:
            continue
        partial = directory / f"{name}.partial"
        try:
            with urlopen(f"{SOURCE}/{name}", timeout=60) as source, partial.open("wb") as output:
                copyfileobj(source, output)
            if sha256(partial) != expected:
                raise ValueError(f"SHA-256 verification failed for {name}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def fix_dimensions(values, shapes):
    if {value.name for value in values} != shapes.keys():
        raise ValueError("Unexpected decoder graph input or output")
    for value in values:
        dimensions = value.type.tensor_type.shape.dim
        expected = shapes[value.name]
        if len(dimensions) != len(expected):
            raise ValueError(f"Unexpected rank for {value.name}")
        for dim, size in zip(dimensions, expected, strict=True):
            dim.dim_value = size


def add_token_selection(graph):
    logits = next(output for output in graph.output if output.name == "outputs")
    token_shape = [dimension.dim_value for dimension in logits.type.tensor_type.shape.dim[:-1]]
    graph.output.remove(logits)
    graph.output.insert(0, helper.make_tensor_value_info("token_id", TensorProto.INT32, token_shape))
    graph.node.extend([
        # NumPy argmax takes the first index when maximum scores tie.
        helper.make_node(
            "ArgMax", ["outputs"], ["token_id_int64"], name="rnnt_token_argmax",
            axis=-1, keepdims=0, select_last_index=0,
        ),
        helper.make_node(
            "Cast", ["token_id_int64"], ["token_id"], name="rnnt_token_int32", to=TensorProto.INT32,
        ),
    ])


def freeze_decoder(directory, destination, token_output=False, frame_count=1):
    model = onnx.load(directory / "decoder_joint.onnx")
    graph = model.graph
    # The prediction length is passed through unchanged. The fixed RNNT
    # step always consumes exactly one symbol and ignores that output.
    for group, obsolete in (
        (graph.input, lambda value: value.name == "target_length"),
        (graph.output, lambda value: value.name == "prednet_lengths"),
        (graph.node, lambda node: tuple(node.output) == ("prednet_lengths",)),
    ):
        (index,) = (index for index, value in enumerate(group) if obsolete(value))
        del group[index]
    fix_dimensions(graph.input, INPUTS | {"encoder_outputs": (1, 1024, frame_count)})
    fix_dimensions(graph.output, OUTPUTS | {"outputs": (1, frame_count, 1, 1025)})
    if token_output:
        add_token_selection(graph)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model)
    partial = destination.with_suffix(".onnx.partial")
    try:
        onnx.save(model, partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def compile_decoder(directory, destination, token_output=False, frame_count=1):
    from prepare_unified_encoder import QnnContextCompiler

    with TemporaryDirectory(prefix="unified-decoder-", dir=directory) as temporary:
        scratch = Path(temporary)
        raw = scratch / "decoder-uncompiled.onnx"
        cached = scratch / destination.name
        freeze_decoder(directory, raw, token_output, frame_count)
        QnnContextCompiler().export(raw, cached)
        raw.unlink()
        for path in scratch.iterdir():
            if path != cached:
                path.replace(destination.with_name(path.name))
        cached.replace(destination)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--compile-context", action="store_true")
    parser.add_argument("--token-output", action="store_true", help="Select the greedy token inside the graph.")
    parser.add_argument("--frame-count", type=int, choices=(1, 4), default=1,
                        help="Number of encoder frames to score with the same predictor state.")
    args = parser.parse_args()
    directory = args.model_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    download_assets(directory)
    destination = directory / "decoder-joint-static.onnx"
    if args.compile_context:
        if destination.exists():
            raise FileExistsError(f"A fixed decoder already exists at {destination}")
        compile_decoder(directory, destination, args.token_output, args.frame_count)
    else:
        freeze_decoder(directory, destination, args.token_output, args.frame_count)
    print(f"Prepared {destination}")


if __name__ == "__main__":
    main()
