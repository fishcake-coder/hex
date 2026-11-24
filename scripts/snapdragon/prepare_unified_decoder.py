"""Prepare the fixed single-step Parakeet Unified RNNT decoder for QNN HTP."""

from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import onnx


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


def freeze_decoder(directory, destination):
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
    fix_dimensions(graph.input, INPUTS)
    fix_dimensions(graph.output, OUTPUTS)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model)
    partial = destination.with_suffix(".onnx.partial")
    try:
        onnx.save(model, partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def compile_decoder(directory, destination):
    from prepare_unified_encoder import QnnContextCompiler

    with TemporaryDirectory(prefix="unified-decoder-", dir=directory) as temporary:
        scratch = Path(temporary)
        raw = scratch / "decoder-uncompiled.onnx"
        cached = scratch / destination.name
        freeze_decoder(directory, raw)
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
    args = parser.parse_args()
    directory = args.model_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    download_assets(directory)
    destination = directory / "decoder-joint-static.onnx"
    if args.compile_context:
        if destination.exists():
            raise FileExistsError(f"A fixed decoder already exists at {destination}")
        compile_decoder(directory, destination)
    else:
        freeze_decoder(directory, destination)
    print(f"Prepared {destination}")


if __name__ == "__main__":
    main()
