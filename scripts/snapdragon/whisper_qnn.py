"""Persistent Hex worker for Qualcomm's Whisper large-v3-turbo QNN export."""

import argparse
import json
import os
import platform
from pathlib import Path
import sys
import wave


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 30 * SAMPLE_RATE
HOP_SAMPLES = 160
FFT_SAMPLES = 400
TIMESTAMP_MS = 20
PROVIDER = "QNNExecutionProvider"
MODEL = "whisper-large-v3-turbo"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def create_session(path, device, backend, profile_dir):
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {"backend_path": backend, "htp_performance_mode": "burst"}
    if profile_dir:
        provider_options["profiling_level"] = "basic"
        provider_options["profiling_file_path"] = str(profile_dir / f"{path.stem}-qnn.csv")
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_dir / f"{path.stem}-ort")
    options.add_provider_for_devices([device], provider_options)
    session = ort.InferenceSession(str(path), sess_options=options, enable_fallback=False)
    if PROVIDER not in session.get_providers():
        raise RuntimeError(f"{path.name} did not load with {PROVIDER}")
    return session


def quantize(value, spec):
    params = spec["quantization_parameters"]
    limits = np.iinfo(spec["dtype"])
    raw = np.rint(value / params["scale"]) + params["zero_point"]
    return np.clip(raw, limits.min, limits.max).astype(spec["dtype"])


def dequantize(value, spec):
    params = spec["quantization_parameters"]
    return (value.astype(np.float32) - params["zero_point"]) * params["scale"]


def log_sum_exp(values):
    maximum = values.max(initial=-np.inf)
    if not np.isfinite(maximum):
        return maximum
    return maximum + np.log(np.exp(values - maximum).sum())


def quantized_zero(spec):
    return np.full(spec["shape"], spec["quantization_parameters"]["zero_point"], spec["dtype"])


def match_cache_specs(outputs, inputs, pairs):
    for output, input_name in pairs:
        if outputs[output] != inputs[input_name]:
            raise RuntimeError(f"Mismatched quantization for Whisper attention cache {output}")


def validate_tensor_specs(session, graph):
    for key, nodes in (("inputs", session.get_inputs()), ("outputs", session.get_outputs())):
        actual = {node.name: (node.type, node.shape) for node in nodes}
        expected = {
            name: (f"tensor({spec['dtype']})", spec["shape"])
            for name, spec in graph[key].items()
        }
        if actual != expected:
            raise RuntimeError(f"Whisper graph {key} do not match the Qualcomm export metadata")


def log_mel(audio, hann, filters):
    padded = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    centered = np.pad(padded, FFT_SAMPLES // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(centered, FFT_SAMPLES)[::HOP_SAMPLES][:-1]
    spectrum = np.fft.rfft(frames * hann)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    mel = np.log10(np.maximum(filters @ power.T, 1e-10))
    return ((np.maximum(mel, mel.max() - 8) + 4) / 4)[None, ...]


def quiet_cut(audio, limit):
    """Find a 200 ms pause in the three seconds leading up to a cut."""
    if len(audio) <= limit:
        return len(audio)
    step = SAMPLE_RATE // 50
    right = limit // step * step
    left = max(right // 2, right - 3 * SAMPLE_RATE) // step * step
    frames = audio[left:right].reshape(-1, step)
    power = np.mean(frames**2, axis=1)
    pause_width = 10
    if len(power) >= pause_width:
        pause_power = np.convolve(power, np.full(pause_width, 1 / pause_width), mode="valid")
        pauses = np.flatnonzero(pause_power <= max(0.002**2, np.median(power) * 0.025))
        if len(pauses):
            return left + (int(pauses[-1]) + pause_width // 2) * step
    return limit


def has_speech_energy(audio):
    """Check for sustained energy before decoding a model's unconsumed tail."""
    frame = SAMPLE_RATE // 20
    complete_frames = len(audio) // frame
    if complete_frames < 20:
        return False
    power = np.mean(audio[:complete_frames * frame].reshape(-1, frame)**2, axis=1)
    return np.count_nonzero(power > 0.004**2) >= 4


class Whisper:
    def __init__(self, model_dir, profile_dir):
        if sys.platform != "win32" or platform.machine().upper() not in {"ARM64", "AARCH64"}:
            raise RuntimeError("Snapdragon transcription requires native Windows ARM64 Python")
        from importlib.metadata import version
        import onnxruntime_qnn as qnn
        from tokenizers import Tokenizer

        metadata = read_json(model_dir / "metadata.json")
        if (
            metadata["model_id"] != "whisper_large_v3_turbo_quantized"
            or metadata["runtime"] != "precompiled_qnn_onnx"
            or metadata["precision"] != "w8a16"
        ):
            raise RuntimeError("Expected the Qualcomm w8a16 Whisper large-v3-turbo QNN export")
        graphs = metadata["model_files"]
        self.encoder_input = graphs["encoder.onnx"]["inputs"]["input_features"]
        self.cross_names = list(graphs["encoder.onnx"]["outputs"])
        self.decoder_inputs = graphs["decoder.onnx"]["inputs"]
        self.decoder_outputs = graphs["decoder.onnx"]["outputs"]
        self.self_names = [name for name in self.decoder_inputs if name.endswith("_in")]
        self.self_outputs = [name.removesuffix("_in") + "_out" for name in self.self_names]
        self.cache_zeros = {name: quantized_zero(self.decoder_inputs[name]) for name in self.self_names}
        match_cache_specs(self.decoder_outputs, self.decoder_inputs, zip(self.self_outputs, self.self_names))
        match_cache_specs(graphs["encoder.onnx"]["outputs"], self.decoder_inputs, zip(self.cross_names, self.cross_names))

        generation = read_json(model_dir / "generation_config.json")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.languages = {tag[2:-2]: token for tag, token in generation["lang_to_id"].items()}
        self.language_ids = list(self.languages.values())
        self.language_codes = list(self.languages)
        self.start_token = generation["decoder_start_token_id"]
        self.end_token = generation["eos_token_id"]
        self.transcribe_token = generation["task_to_id"]["transcribe"]
        self.no_speech_token = self.tokenizer.token_to_id("<|nospeech|>")
        self.timestamp_begin = generation["no_timestamps_token_id"] + 1
        self.suppress = list(set(generation["suppress_tokens"]) | set(range(self.end_token + 1, self.timestamp_begin)))
        self.max_initial_timestamp = generation["max_initial_timestamp_index"]
        self.hann = np.hanning(FFT_SAMPLES + 1)[:-1].astype(np.float32)
        with np.load(model_dir / "mel_filters.npz") as filters:
            self.mel_filters = filters["mel_128"]

        ort.register_execution_provider_library(PROVIDER, qnn.get_library_path())
        try:
            device = next(
                device for device in ort.get_ep_devices()
                if device.ep_name == PROVIDER and device.device.type == ort.OrtHardwareDeviceType.NPU
            )
        except StopIteration:
            raise RuntimeError("The Qualcomm QNN execution provider found no NPU") from None
        backend = qnn.get_qnn_htp_path()
        self.encoder = create_session(model_dir / "encoder.onnx", device, backend, profile_dir)
        self.decoder = create_session(model_dir / "decoder.onnx", device, backend, profile_dir)
        validate_tensor_specs(self.encoder, graphs["encoder.onnx"])
        validate_tensor_specs(self.decoder, graphs["decoder.onnx"])
        self.profile_dir = profile_dir
        self.info = {
            "model": MODEL,
            "backend": "qnn",
            "device": "htp",
            "languages": sorted(self.languages),
            "provider": PROVIDER,
            "architecture": platform.machine(),
            "onnx_runtime": ort.__version__,
            "qnn_ep": version("onnxruntime-qnn"),
            "qairt_export": metadata["tool_versions"]["qairt"],
            "cpu_fallback": False,
        }

    def finish(self):
        if self.profile_dir:
            self.encoder.end_profiling()
            self.decoder.end_profiling()

    def choose_token(self, logits, tokens, limit):
        """Apply Whisper's paired, monotonically increasing timestamp rules."""
        time = self.timestamp_begin
        end = self.end_token
        logits[self.suppress] = -np.inf
        logits[time + limit + 1:] = -np.inf
        if not tokens:
            logits[:time] = -np.inf
            logits[time + self.max_initial_timestamp + 1:] = -np.inf
            return int(np.argmax(logits))

        last_is_time = tokens[-1] >= time
        preceding_is_time = len(tokens) == 1 or tokens[-2] >= time
        if last_is_time and preceding_is_time:
            logits[time:] = -np.inf
        elif last_is_time:
            logits[:end] = -np.inf

        last_time = next((token for token in reversed(tokens) if token >= time), None)
        if last_time is not None:
            minimum = last_time if last_is_time and not preceding_is_time else last_time + 1
            logits[time:minimum] = -np.inf
        if log_sum_exp(logits[time:]) > logits[:time].max():
            logits[:time] = -np.inf
        return int(np.argmax(logits))

    def decode(self, audio, language):
        features = quantize(log_mel(audio, self.hann, self.mel_filters), self.encoder_input)
        cross = self.encoder.run(self.cross_names, {"input_features": features})
        mask_spec = self.decoder_inputs["attention_mask"]
        mask = np.full(mask_spec["shape"], -100.0, dtype=np.float32)
        mask = quantize(mask, mask_spec)
        inputs = dict(zip(self.cross_names, cross, strict=True)) | self.cache_zeros
        inputs["input_ids"] = np.empty((1, 1), dtype=np.int32)
        inputs["position_ids"] = np.empty((1,), dtype=np.int32)
        inputs["attention_mask"] = mask
        logits_spec = self.decoder_outputs["logits"]
        output_names = ["logits", *self.self_outputs]

        def step(token, position):
            inputs["input_ids"].fill(token)
            inputs["position_ids"].fill(position)
            mask[..., -1 - position] = mask_spec["quantization_parameters"]["zero_point"]
            logits, *cache = self.decoder.run(output_names, inputs)
            inputs.update(zip(self.self_names, cache, strict=True))
            return dequantize(logits.reshape(-1), logits_spec)

        logits = step(self.start_token, 0)
        if language is None:
            language = self.language_codes[int(np.argmax(logits[self.language_ids]))]
        no_speech = np.exp(logits[self.no_speech_token] - log_sum_exp(logits))
        step(self.languages[language], 1)
        logits = step(self.transcribe_token, 2)
        max_timestamp = (len(audio) + 2 * HOP_SAMPLES - 1) // (2 * HOP_SAMPLES)
        tokens = []
        log_probability = 0.0
        for position in range(3, mask.shape[-1]):
            token = self.choose_token(logits, tokens, max_timestamp)
            log_probability += float(logits[token] - log_sum_exp(logits))
            if token == self.end_token:
                if no_speech > 0.6 and log_probability / (len(tokens) + 1) < -1.0:
                    tokens = []
                return tokens, language, False
            tokens.append(token)
            logits = step(token, position)
        return tokens, language, True

    def segments(self, tokens, offset, length):
        start = None
        words = []
        for token in tokens:
            if token < self.timestamp_begin:
                words.append(token)
                continue
            elapsed = min((token - self.timestamp_begin) * TIMESTAMP_MS, length)
            if words and start is not None and elapsed > start:
                text = self.tokenizer.decode(words).strip()
                if text:
                    yield {"start_ms": offset + start, "end_ms": offset + elapsed, "text": text}
            start, words = elapsed, []

    def transcribe_window(self, audio, offset, language):
        if len(audio) == 0 or not np.any(audio):
            return
        tokens, detected, saturated = self.decode(audio, language)
        if saturated:
            if len(audio) < 2 * SAMPLE_RATE:
                raise RuntimeError("Whisper reached its token limit on less than two seconds of audio")
            cut = quiet_cut(audio, len(audio) // 2)
            yield from self.transcribe_window(audio[:cut], offset, language)
            yield from self.transcribe_window(audio[cut:], offset + cut, language)
            return

        text = self.tokenizer.decode([token for token in tokens if token < self.end_token]).strip()
        marks = list(self.segments(tokens, offset * 1000 // SAMPLE_RATE, len(audio) * 1000 // SAMPLE_RATE))
        yield text, detected, marks
        if marks and tokens[-1] >= self.timestamp_begin:
            consumed = (tokens[-1] - self.timestamp_begin) * 2 * HOP_SAMPLES
            if consumed >= SAMPLE_RATE and has_speech_energy(audio[consumed + 2 * HOP_SAMPLES:]):
                # A window may close on a valid timestamp while later speech remains.
                yield from self.transcribe_window(audio[consumed:], offset + consumed, language)

    def transcribe(self, audio_path, language):
        language = None if language in (None, "auto") else language
        if language is not None and (not isinstance(language, str) or language not in self.languages):
            raise ValueError(f"Unsupported Whisper language: {language}")
        with wave.open(str(audio_path), "rb") as file:
            if (file.getnchannels(), file.getsampwidth(), file.getframerate()) != (1, 2, SAMPLE_RATE):
                raise ValueError("Expected a 16 kHz mono signed 16-bit PCM WAV")
            pcm = file.readframes(file.getnframes())
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        offset = 0
        text_parts, segments = [], []
        detected_language = language
        while offset < len(audio):
            size = quiet_cut(audio[offset:], WINDOW_SAMPLES)
            for text, detected, marks in self.transcribe_window(audio[offset:offset + size], offset, language):
                if text:
                    text_parts.append(text)
                    detected_language = detected_language or detected
                segments.extend(marks)
            offset += size
        return {
            "text": " ".join(text_parts),
            "language": detected_language,
            "segments": segments,
        }


def main():
    default_models = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Hex" / "snapdragon" / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("HEX_SNAPDRAGON_MODEL_DIR", default_models))
    args = parser.parse_args()
    profile_dir = os.environ.get("HEX_SNAPDRAGON_PROFILE_DIR")
    if profile_dir:
        profile_dir = Path(profile_dir).resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)

    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    worker = None
    try:
        for line in sys.stdin:
            request_id = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON request object")
                request_id = request["id"]
                method = request["method"]
                if method == "initialize":
                    if worker is None:
                        # Keep dependency errors within the request/response protocol.
                        global np, ort
                        import numpy as np
                        import onnxruntime as ort

                        worker = Whisper(args.model_dir.resolve(), profile_dir)
                    result = worker.info
                elif method == "transcribe" and worker is not None:
                    result = worker.transcribe(request["audio_path"], request.get("language"))
                else:
                    raise ValueError(f"Invalid request for the current worker state: {method}")
                response = {"id": request_id, "result": result}
            except Exception as error:
                response = {"id": request_id, "error": str(error) or type(error).__name__}
            print(json.dumps(response, ensure_ascii=False), flush=True)
    finally:
        if worker is not None:
            worker.finish()


if __name__ == "__main__":
    main()
