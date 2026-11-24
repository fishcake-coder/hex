"""Persistent Hex worker for Parakeet TDT v3 with a Qualcomm QNN encoder."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import wave

PROVIDER = "QNNExecutionProvider"

SAMPLE_RATE = 16_000
WINDOW = 8 * SAMPLE_RATE
OVERLAP = SAMPLE_RATE
FRAME_MS = 80
BLANK = 8192
MAX_SYMBOLS_PER_FRAME = 10


def load_encoder(path, profile_dir):
    import onnxruntime_qnn as qnn

    ort.register_execution_provider_library(PROVIDER, qnn.get_library_path())
    ep = next(
        (device for device in ort.get_ep_devices()
         if device.ep_name == PROVIDER and device.device.type == ort.OrtHardwareDeviceType.NPU),
        None,
    )
    if ep is None:
        raise RuntimeError("The Qualcomm QNN execution provider found no NPU")

    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {"backend_path": qnn.get_qnn_htp_path(), "htp_performance_mode": "burst"}
    if profile_dir:
        provider_options["profiling_level"] = "basic"
        provider_options["profiling_file_path"] = str(profile_dir / "encoder-qnn.csv")
    options.add_provider_for_devices([ep], provider_options)
    encoder = ort.InferenceSession(str(path), sess_options=options, enable_fallback=False)
    if PROVIDER not in encoder.get_providers():
        raise RuntimeError(f"{path.name} did not load with {PROVIDER}")
    return encoder


def cpu_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"], enable_fallback=False
    )


def pcm16(path):
    with wave.open(path, "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError("Expected a 16 kHz mono signed 16-bit PCM WAV")
        data = audio.readframes(audio.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def quiet_split(audio):
    """Find at least 120 ms of quiet in the last two seconds of a full window."""
    frame_samples = SAMPLE_RATE // 50
    search_at = WINDOW - 2 * SAMPLE_RATE
    frames = audio[search_at:WINDOW].reshape(-1, frame_samples)
    quiet = np.mean(frames ** 2, axis=1) < 0.004 ** 2
    pauses = np.flatnonzero(np.convolve(quiet.astype(np.int32), np.ones(6, np.int32), "valid") == 6)
    if len(pauses):
        return search_at + (int(pauses[-1]) + 3) * frame_samples
    return None


def windows(audio):
    """Prefer a quiet cut. If speech is continuous, give both sides context."""
    start = owned_from = 0
    while start < len(audio):
        end = min(start + WINDOW, len(audio))
        if end == len(audio):
            yield start, end, owned_from, end
            break
        split = quiet_split(audio[start:end])
        if split is not None:
            cut = start + split
            yield start, cut, owned_from, cut
            start = owned_from = cut
        else:
            cut = end - OVERLAP // 2
            yield start, end, owned_from, cut
            start, owned_from = end - OVERLAP, cut


class Parakeet:
    def __init__(self, models, profile_dir):
        if sys.platform != "win32" or platform.machine().upper() not in {"ARM64", "AARCH64"}:
            raise RuntimeError("Snapdragon transcription requires native Windows ARM64 Python")
        self.encoder = load_encoder(models / "encoder-model.onnx", profile_dir)
        self.preprocessor = cpu_session(models / "nemo128.onnx")
        self.decoder = cpu_session(models / "decoder_joint-model.int8.onnx")
        self.vocab = {}
        for line in (models / "vocab.txt").read_text(encoding="utf-8").splitlines():
            token, index = line.rsplit(" ", 1)
            self.vocab[int(index)] = token
        self.info = {
            "model": "Parakeet TDT 0.6B v3 (NPU encoder, CPU decoder)",
            "backend": "qnn",
            "device": "htp",
            "languages": ["auto"],
            "provider": PROVIDER,
            "architecture": platform.machine(),
            "onnx_runtime": ort.__version__,
            "qnn_ep": version("onnxruntime-qnn"),
            "cpu_fallback": False,
            "graph_devices": {"encoder": "htp", "decoder_joint": "cpu", "preprocessor": "cpu"},
        }

    def encode(self, audio):
        waveforms = np.zeros((1, WINDOW), dtype=np.float32)
        waveforms[0, :len(audio)] = audio
        features, features_len = self.preprocessor.run(
            ["features", "features_lens"],
            {"waveforms": waveforms, "waveforms_lens": np.array([len(audio)], dtype=np.int64)},
        )
        encoded, encoded_len = self.encoder.run(
            ["output_0", "output_1"],
            {"audio_signal": features, "length": features_len.astype(np.int32)},
        )
        frames = min(int(encoded_len[0]), encoded.shape[2], (int(features_len[0]) + 7) // 8)
        return encoded[:, :, :frames]

    def decode(self, encoded):
        start = np.zeros((2, 1, 640), dtype=np.float32)
        state1, state2 = start, start
        last_token = BLANK
        symbols_at_frame = 0
        frame = 0
        while frame < encoded.shape[2]:
            logits, next_state1, next_state2 = self.decoder.run(
                ["outputs", "output_states_1", "output_states_2"],
                {
                    "encoder_outputs": np.ascontiguousarray(encoded[:, :, frame:frame + 1]),
                    "targets": np.array([[last_token]], dtype=np.int32),
                    "target_length": np.ones(1, dtype=np.int32),
                    "input_states_1": state1,
                    "input_states_2": state2,
                },
            )
            logits = logits.reshape(-1)
            token = int(logits[:BLANK + 1].argmax())
            duration = int(logits[BLANK + 1:].argmax())
            if token != BLANK:
                yield token, frame * FRAME_MS, (frame + max(1, duration)) * FRAME_MS
                state1, state2 = next_state1, next_state2
                last_token = token
                symbols_at_frame += 1
            if duration > 0 or token == BLANK or symbols_at_frame == MAX_SYMBOLS_PER_FRAME:
                frame += max(1, duration)
                symbols_at_frame = 0

    def words(self, tokens):
        word = ""
        start, end = 0, 0
        for token, token_start, token_end in tokens:
            piece = self.vocab[token]
            if piece.startswith("<|") or piece in {"<pad>", "<unk>"}:
                continue
            if piece.startswith("▁"):
                if word:
                    yield word, start, end
                word, start = piece[1:], token_start
            else:
                if not word:
                    start = token_start
                word += piece
            end = token_end
        if word:
            yield word, start, end

    def transcribe(self, path, language):
        if language not in (None, "auto"):
            raise ValueError("Parakeet TDT v3 requires automatic language recognition")
        audio = pcm16(path)
        words = []
        audio_ms = len(audio) * 1000 // SAMPLE_RATE
        for start, end, split_start, split_end in windows(audio):
            waveform = audio[start:end]
            if np.any(waveform):
                offset_ms = start * 1000 // SAMPLE_RATE
                for word, word_start, word_end in self.words(self.decode(self.encode(waveform))):
                    word_start += offset_ms
                    word_end = min(audio_ms, word_end + offset_ms)
                    center = (word_start + word_end) * SAMPLE_RATE // 2000
                    if split_start <= center < split_end and word_start < word_end:
                        words.append((word, word_start, word_end))
        text = " ".join(word for word, _, _ in words)
        segments = []
        part = []
        for word, start, end in words:
            if part and start - last_end >= 400:
                segments.append({"start_ms": first_start, "end_ms": last_end, "text": " ".join(part)})
                part = []
            if not part:
                first_start = start
            part.append(word)
            last_end = end
        if part:
            segments.append({"start_ms": first_start, "end_ms": last_end, "text": " ".join(part)})
        return {"text": text, "language": None, "segments": segments}


def main():
    default = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Hex" / "snapdragon" / "models-tdt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=os.environ.get("HEX_SNAPDRAGON_MODEL_DIR", default)
    )
    model_dir = parser.parse_args().model_dir.resolve()
    profile_dir = os.environ.get("HEX_SNAPDRAGON_PROFILE_DIR")
    if profile_dir:
        profile_dir = Path(profile_dir).resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    worker = None
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON request object")
            request_id = request["id"]
            if request["method"] == "initialize":
                if worker is None:
                    # Keep dependency errors within the request/response protocol.
                    global np, ort
                    import numpy as np
                    import onnxruntime as ort

                    worker = Parakeet(model_dir, profile_dir)
                result = worker.info
            elif request["method"] == "transcribe" and worker is not None:
                result = worker.transcribe(request["audio_path"], request.get("language"))
            else:
                raise ValueError("Invalid request for the current worker state")
            response = {"id": request_id, "result": result}
        except Exception as error:
            response = {"id": request_id, "error": str(error) or type(error).__name__}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
