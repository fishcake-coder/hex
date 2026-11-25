"""Build fixed-window Unified English spectral-energy contexts for Qualcomm HTP."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from prepare_unified_encoder import QnnContextCompiler
from unified_rnnt import FFT_SIZE, HOP_SIZE, MEL_BINS, WINDOW_SIZE, mel_filterbank


RESIDUAL_SCALE = 2048
WEAK_BIN_LIMIT = 1.25
WEAK_BIN_BOOST = 128


def dft_kernels():
    padding = (FFT_SIZE - WINDOW_SIZE) // 2
    window = np.pad(np.hanning(WINDOW_SIZE), padding)
    frequencies = np.arange(FFT_SIZE // 2 + 1)
    phase = (-2 * np.pi / FFT_SIZE) * frequencies[:, None] * np.arange(FFT_SIZE)
    # Keep the largest expected magnitudes finite after squaring in FP16.
    return (np.concatenate((np.cos(phase), np.sin(phase)))[:, None, None, :] * (window / 4)).astype(np.float32)


def frontend_model(mel_frames):
    if mel_frames < 2:
        raise ValueError("The spectral window must have at least two frames")
    samples = mel_frames * HOP_SIZE
    frequencies = FFT_SIZE // 2 + 1
    nodes = []
    weights = []

    def constant(name, value, dtype=np.float32):
        weights.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, output, *inputs, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output], name=output, **attributes))
        return output

    def transform(output, waveform, kernel):
        return op("Conv", output, waveform, kernel,
                  pads=[0, FFT_SIZE // 2, 0, FFT_SIZE // 2 - HOP_SIZE], strides=[1, HOP_SIZE])

    kernels = dft_kernels()
    coarse = kernels.astype(np.float16).astype(np.float32)
    residual = (kernels - coarse) * RESIDUAL_SCALE
    high_kernel = constant("dft_high", coarse)
    low_kernel = constant("dft_low", residual)

    # Recover the two cross terms lost by rounding the waveform and Fourier
    # coefficients to FP16. Their residuals enter scaled to stay normal.
    high = transform("hi_transform", "waveform_high", high_kernel)
    coefficient_residual = transform("lo_transform_a", "waveform_high", low_kernel)
    signal_residual = transform("lo_transform_b", "waveform_low", high_kernel)
    low = op("Add", "lo_transform", coefficient_residual, signal_residual)
    low = op("Mul", "lo_scaled", low, constant("compensation_scale", 1 / RESIDUAL_SCALE))
    spectrum = op("Add", "spectrum", high, low)
    spectrum = op("Squeeze", "spectrum_frames", spectrum, constant("squeeze_axis", [2], np.int64))
    split = constant("complex_split", [frequencies, frequencies], np.int64)
    nodes.append(helper.make_node("Split", [spectrum, split], ["real", "imaginary"], axis=1, name="split_complex"))

    # Double the coefficients to keep the smallest nonzero mel weight normal
    # in FP16. The caller removes this factor and the Fourier amplitude scale.
    filters = constant("slaney_mel", mel_filterbank()[None, :, :] * 2)
    power = op("Add", "high_power", op("Mul", "real_power", "real", "real"),
               op("Mul", "imaginary_power", "imaginary", "imaginary"))
    op("MatMul", "mel_high", filters, power)

    # A boosted path preserves weak bands during squaring. Its clipping is
    # above any contributing magnitude when unboosted mel energy is < 2**-14.
    maximum = constant("upper_limit", WEAK_BIN_LIMIT)
    minimum = constant("lower_limit", -WEAK_BIN_LIMIT)
    boost = constant("weak_spectrum_boost", WEAK_BIN_BOOST)
    weak_power = []
    for component in ("real", "imaginary"):
        clipped = op("Clip", f"{component}_clipped", component, minimum, maximum)
        scaled = op("Mul", f"{component}_scaled", clipped, boost)
        weak_power.append(op("Mul", f"{component}_low_power", scaled, scaled))
    weak = op("Add", "low_power", *weak_power)
    op("MatMul", "mel_low", filters, weak)

    input_shape = [1, 1, 1, samples]
    output_shape = [1, MEL_BINS, mel_frames]
    graph = helper.make_graph(
        nodes,
        f"unified-compensated-mel-{mel_frames}",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, input_shape)
         for name in ("waveform_high", "waveform_low")],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
         for name in ("mel_high", "mel_low")],
        initializer=weights,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.checker.check_model(model)
    return model


def prepare(directory, windows, raw):
    directory.mkdir(parents=True, exist_ok=True)
    destinations = [directory / f"frontend-{frames}.onnx" for frames in windows]
    if any(path.exists() for path in destinations):
        raise FileExistsError(f"A requested frontend already exists under {directory}")

    compiler = None if raw else QnnContextCompiler()
    with TemporaryDirectory(prefix="unified-frontend-", dir=directory) as temporary:
        scratch = Path(temporary)
        for frames, destination in zip(windows, destinations, strict=True):
            source = scratch / destination.name if raw else scratch / "frontend-uncompiled.onnx"
            onnx.save_model(frontend_model(frames), source)
            if compiler is not None:
                compiler.export(source, scratch / destination.name)
                source.unlink()
        for path in scratch.iterdir():
            if path.suffix != ".onnx":
                path.replace(directory / path.name)
        for destination in destinations:
            (scratch / destination.name).replace(destination)
            print(f"Prepared {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mel-frames", type=int, nargs="+")
    parser.add_argument("--raw", action="store_true", help="Write portable ONNX graphs for parity checks")
    args = parser.parse_args()
    windows = args.mel_frames
    if windows is None:
        manifest = args.model_dir / "encoder-manifest.json"
        windows = [json.loads(manifest.read_text(encoding="utf-8-sig"))["mel_frames"]]
    prepare(args.model_dir.resolve(), sorted(set(windows)), args.raw)


if __name__ == "__main__":
    main()
