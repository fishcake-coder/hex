"""Compile the Unified English 70/7/7 buffered encoder for Qualcomm HTP."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_unified_encoder import SUBSAMPLING, prepare


LEFT_FRAMES = 70
CHUNK_FRAMES = 7
RIGHT_FRAMES = 7
MEL_FRAMES = (LEFT_FRAMES + CHUNK_FRAMES + RIGHT_FRAMES) * SUBSAMPLING


def prepare_streaming(source, destination):
    if destination.exists():
        raise FileExistsError(f"A streaming encoder already exists at {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="unified-streaming-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "encoder"
        # NeMo's buffered inference selects a chunked attention mask at
        # runtime and keeps the checkpoint's symmetric depthwise convolution.
        # These graphs take the same padding inputs as the offline encoder.
        # The caller combines that padding with the buffered attention mask.
        prepare(source, staged, MEL_FRAMES, layers_per_stage=2, compile_contexts=True)
        manifest_path = staged / "encoder-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(left_frames=LEFT_FRAMES, chunk_frames=CHUNK_FRAMES, right_frames=RIGHT_FRAMES)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(destination)
    print(f"Prepared the Unified buffered encoder in {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare_streaming(args.source_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
