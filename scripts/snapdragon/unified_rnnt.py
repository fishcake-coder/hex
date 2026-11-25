"""Hex worker for Parakeet Unified English with QNN HTP encoder and decoder."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import wave

import numpy as np


SAMPLE_RATE = 16_000
FFT_SIZE = 512
WINDOW_SIZE = 400
HOP_SIZE = 160
MEL_BINS = 128
SUBSAMPLING = 8
ENCODER_WIDTH = 1024
BLANK = 1024
MAX_SYMBOLS_PER_FRAME = 10
TRAILING_SILENCE_SAMPLES = 400 * SAMPLE_RATE // 1000
QNN_PROVIDER = "QNNExecutionProvider"


def mel_filterbank():
    """Build the Slaney, area-normalized filters used by NVIDIA NeMo."""
    hz_per_mel = 200 / 3
    linear_limit = 1000 / hz_per_mel
    log_step = np.log(6.4) / 27
    mel_limit = linear_limit + np.log(SAMPLE_RATE / 2 / 1000) / log_step
    mel = np.linspace(0, mel_limit, MEL_BINS + 2)
    corners = np.where(mel < linear_limit, hz_per_mel * mel, 1000 * np.exp(log_step * (mel - linear_limit)))
    bins = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    widths = np.diff(corners)[:, None]
    rising = (bins - corners[:-2, None]) / widths[:-1]
    falling = (corners[2:, None] - bins) / widths[1:]
    triangles = np.maximum(0, np.minimum(rising, falling))
    area_normalization = 2 / (corners[2:] - corners[:-2])
    return (triangles * area_normalization[:, None]).astype(np.float32)


class Frontend:
    device = "cpu"

    def __init__(self, mel_frames):
        self.mel_frames = mel_frames
        self.filters = mel_filterbank()
        padding = (FFT_SIZE - WINDOW_SIZE) // 2
        self.window = np.pad(np.hanning(WINDOW_SIZE).astype(np.float32), padding)

    def encode(self, audio):
        length = min(len(audio) // HOP_SIZE, self.mel_frames)
        features = np.zeros((1, MEL_BINS, self.mel_frames), dtype=np.float32)
        if length < 2:
            return features, length

        preemphasis = audio.copy()
        preemphasis[1:] -= np.float32(0.97) * audio[:-1]
        centered = np.pad(preemphasis, FFT_SIZE // 2)
        windows = np.lib.stride_tricks.sliding_window_view(centered, FFT_SIZE)[::HOP_SIZE][:length]
        fft = np.fft.rfft(windows * self.window)
        spectrum = (fft.real ** 2 + fft.imag ** 2).astype(np.float32)
        features[0, :, :length] = normalize_mel(self.filters @ spectrum.T, length)
        return features, length


def normalize_mel(energy, length):
    log_mel = np.log(energy + np.float32(2 ** -24))
    centered_mel = log_mel - log_mel[:, :1]
    centered_mel -= centered_mel.mean(axis=1, keepdims=True)
    std = np.sqrt(np.sum(centered_mel ** 2, axis=1, keepdims=True) / (length - 1))
    return centered_mel / (std + np.float32(1e-5))


class QnnSpectralFrontend(Frontend):
    device = "cpu+htp"

    def __init__(self, models, runtime, mel_frames):
        super().__init__(mel_frames)
        self.sessions = {}
        for frames, path in models:
            session = runtime.session(path)
            inputs = {node.name: (node.type, node.shape) for node in session.get_inputs()}
            outputs = {node.name: (node.type, node.shape) for node in session.get_outputs()}
            waveform = ("tensor(float)", [1, 1, 1, frames * HOP_SIZE])
            spectrum = ("tensor(float)", [1, MEL_BINS, frames])
            if (
                inputs != {"waveform_high": waveform, "waveform_low": waveform}
                or outputs != {"mel_high": spectrum, "mel_low": spectrum}
            ):
                raise ValueError(f"Unexpected tensors in Unified frontend graph {path.name}")
            self.sessions[frames] = session

    def encode(self, audio):
        length = len(audio) // HOP_SIZE
        # For very short windows the NumPy frontend avoids a dispatch overhead.
        if length < 160:
            return super().encode(audio)
        frames = next((size for size in self.sessions if size * HOP_SIZE >= len(audio)), None)
        if frames is None:
            return super().encode(audio)

        preemphasis = np.zeros(frames * HOP_SIZE, dtype=np.float32)
        samples = len(audio)
        preemphasis[0] = audio[0]
        preemphasis[1:samples] = audio[1:] - np.float32(0.97) * audio[:-1]
        peak = np.abs(preemphasis[:samples]).max()
        gain = np.float32(np.clip(0.25 / peak, 1, 128)) if peak else np.float32(1)
        preemphasis[:samples] *= gain
        coarse = preemphasis.astype(np.float16).astype(np.float32)
        # Keep the discarded FP16 bits in a second input so quiet audio retains
        # the precision required by the model's log and variance operations.
        residual = (preemphasis - coarse) * np.float32(2048)
        high, low = self.sessions[frames].run(["mel_high", "mel_low"], {
            "waveform_high": coarse.reshape(1, 1, 1, -1),
            "waveform_low": residual.reshape(1, 1, 1, -1),
        })
        high, low = high[0, :, :length], low[0, :, :length]
        power = np.where(high >= 2 ** -14, high, low * np.float32(1 / 128 ** 2))
        energy = power * np.float32(8 / gain ** 2)
        features = np.zeros((1, MEL_BINS, self.mel_frames), dtype=np.float32)
        features[0, :, :length] = normalize_mel(energy, length)
        return features, length


class RnntDecoder:
    """Run the matching offline predictor and joint network."""

    def __init__(self, model_dir, runtime):
        # The export's vocab omits SentencePiece ID 0 and writes continuation
        # pieces with a ## prefix. The graph reserves one more ID for blank.
        vocab = (model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
        if len(vocab) != BLANK - 1 or not all(vocab):
            raise ValueError("Expected the Parakeet Unified English export vocabulary")
        self.pieces = [" ⁇ "] + [
            token[2:] if token.startswith("##") else " " + token.replace("▁", "")
            for token in vocab
        ]

        token_model = model_dir / "acceleration" / "decoder-joint-static.onnx"
        self.selects_token_on_npu = token_model.is_file()
        decoder_model = token_model if self.selects_token_on_npu else model_dir / "decoder-joint-static.onnx"
        if not decoder_model.is_file():
            raise FileNotFoundError(f"Missing the Unified QNN decoder in {model_dir}. Run setup-parakeet-unified.ps1.")
        self.session = runtime.session(decoder_model)
        self.prediction_output = "token_id" if self.selects_token_on_npu else "outputs"
        inputs = {node.name: (node.type, node.shape) for node in self.session.get_inputs()}
        self.frames_per_step = inputs["encoder_outputs"][1][-1]
        if self.frames_per_step not in (1, 4) or (
            not self.selects_token_on_npu and self.frames_per_step != 1
        ):
            raise ValueError("Unsupported Unified decoder frame count")
        outputs = {node.name: (node.type, node.shape) for node in self.session.get_outputs()}
        state = ("tensor(float)", [2, 1, 640])
        expected_inputs = {
            "encoder_outputs": ("tensor(float)", [1, ENCODER_WIDTH, self.frames_per_step]),
            "targets": ("tensor(int32)", [1, 1]),
            "input_states_1": state,
            "input_states_2": state,
        }
        expected_outputs = {
            self.prediction_output: (
                ("tensor(int32)", [1, self.frames_per_step, 1]) if self.selects_token_on_npu
                else ("tensor(float)", [1, 1, 1, BLANK + 1])
            ),
            "output_states_1": state,
            "output_states_2": state,
        }
        if inputs != expected_inputs or outputs != expected_outputs:
            raise ValueError("The decoder tensors differ from the Parakeet Unified English export")

    def decode(self, encoded, frames):
        if self.frames_per_step > 1:
            return self.decode_batches(encoded, frames)
        target = np.full((1, 1), BLANK, dtype=np.int32)
        state1 = np.zeros((2, 1, 640), dtype=np.float32)
        state2 = np.zeros_like(state1)
        feeds = {"targets": target}
        output_names = [self.prediction_output, "output_states_1", "output_states_2"]
        emitted = []
        for frame in range(frames):
            activation = np.ascontiguousarray(encoded[:, :, frame:frame + 1])
            for _ in range(MAX_SYMBOLS_PER_FRAME):
                feeds.update(encoder_outputs=activation, input_states_1=state1, input_states_2=state2)
                prediction, next_state1, next_state2 = self.session.run(output_names, feeds)
                token = prediction.item() if self.selects_token_on_npu else int(np.argmax(prediction))
                if token == BLANK:
                    break
                emitted.append(self.pieces[token])
                state1, state2 = next_state1, next_state2
                target.fill(token)
        return " ".join("".join(emitted).split())


    def decode_batches(self, encoded, frames):
        step = self.frames_per_step
        activation = np.zeros((1, ENCODER_WIDTH, step), dtype=np.float32)
        target = np.full((1, 1), BLANK, dtype=np.int32)
        state1 = np.zeros((2, 1, 640), dtype=np.float32)
        state2 = np.zeros_like(state1)
        feeds = {"encoder_outputs": activation, "targets": target}
        output_names = ["token_id", "output_states_1", "output_states_2"]
        emitted = []
        frame = 0
        symbols_at_frame = 0
        while frame < frames:
            count = min(step, frames - frame)
            activation[:, :, :count] = encoded[:, :, frame:frame + count]
            activation[:, :, count:] = 0
            feeds.update(input_states_1=state1, input_states_2=state2)
            labels, next_state1, next_state2 = self.session.run(output_names, feeds)

            # The predictor stays fixed over leading blanks. Scores for any frame
            # beyond the first emitted token use stale predictor state.
            for index, value in enumerate(labels[0, :count, 0]):
                token = int(value)
                if token != BLANK:
                    break
            else:
                frame += count
                symbols_at_frame = 0
                continue

            if index:
                frame += index
                symbols_at_frame = 0
            emitted.append(self.pieces[token])
            target.fill(token)
            state1, state2 = next_state1, next_state2
            symbols_at_frame += 1
            if symbols_at_frame == MAX_SYMBOLS_PER_FRAME:
                frame += 1
                symbols_at_frame = 0
        return " ".join("".join(emitted).split())


class QnnRuntime:
    def __init__(self, profile_dir):
        import onnxruntime as ort
        import onnxruntime_qnn as qnn

        if sys.platform != "win32" or platform.machine().upper() not in {"ARM64", "AARCH64"}:
            raise RuntimeError("The Unified QNN worker requires native Windows ARM64 Python")
        if not any(device.ep_name == QNN_PROVIDER for device in ort.get_ep_devices()):
            ort.register_execution_provider_library(QNN_PROVIDER, qnn.get_library_path())
        self.device = next(
            (
                device for device in ort.get_ep_devices()
                if device.ep_name == QNN_PROVIDER and device.device.type == ort.OrtHardwareDeviceType.NPU
            ),
            None,
        )
        if self.device is None:
            raise RuntimeError("The Qualcomm QNN execution provider found no NPU")
        self.backend_path = qnn.get_qnn_htp_path()
        self.ort = ort
        self.profile_dir = profile_dir
        self.sessions = []
        self.info = {
            "backend": "qnn", "device": "htp", "provider": QNN_PROVIDER, "cpu_fallback": False,
            "onnx_runtime": ort.__version__, "qnn_ep": version("onnxruntime-qnn"),
        }

    def session(self, path):
        opts = self.ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        opts.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        provider_options = {
            "backend_path": self.backend_path,
            "htp_performance_mode": "burst",
            "offload_graph_io_quantization": "0",
        }
        if self.profile_dir:
            provider_options["profiling_level"] = "basic"
            provider_options["profiling_file_path"] = str(self.profile_dir / f"{path.stem}-qnn.csv")
            opts.enable_profiling = True
            opts.profile_file_prefix = str(self.profile_dir / f"{path.stem}-ort")
        opts.add_provider_for_devices([self.device], provider_options)
        session = self.ort.InferenceSession(str(path), sess_options=opts, enable_fallback=False)
        if QNN_PROVIDER not in session.get_providers():
            raise RuntimeError(f"{path.name} did not load with the Qualcomm QNN execution provider")
        self.sessions.append(session)
        return session

    def close(self):
        if self.profile_dir:
            for session in self.sessions:
                session.end_profiling()


class QnnEncoder:
    def __init__(self, model_dir, runtime):
        manifest = json.loads((model_dir / "encoder-manifest.json").read_text(encoding="utf-8-sig"))
        self.mel_frames = manifest["mel_frames"]
        self.encoder_frames = manifest["encoder_frames"]
        self.precompiled = manifest.get("compiled_contexts", False)
        if (
            self.mel_frames < SUBSAMPLING
            or self.mel_frames % SUBSAMPLING != 0
            or self.mel_frames // SUBSAMPLING != self.encoder_frames
            or manifest.get("length_masks") is not True
            or not manifest["stages"]
        ):
            raise ValueError("Invalid Parakeet Unified encoder window in the manifest")
        self.sessions = []
        input_shape = [1, MEL_BINS, self.mel_frames]
        stages = manifest["stages"]
        for i, filename in enumerate(stages):
            path = (model_dir / filename).resolve()
            if path.parent != model_dir or path.suffix != ".onnx":
                raise ValueError(f"Invalid Unified encoder stage: {filename}")
            session = runtime.session(path)
            inputs = {node.name: (node.type, node.shape) for node in session.get_inputs()}
            mask_specs = self.mask_specs(i)
            for name, spec in mask_specs.items():
                if inputs.pop(name, None) != spec:
                    raise ValueError(f"Unexpected length mask {name} in Unified encoder stage {filename}")
            ((input_name, input_spec),) = inputs.items()
            (output_node,) = session.get_outputs()
            output_shape = (
                [1, ENCODER_WIDTH, self.encoder_frames]
                if i + 1 == len(stages) else [1, self.encoder_frames, ENCODER_WIDTH]
            )
            if (
                input_spec != ("tensor(float)", input_shape)
                or output_node.type != "tensor(float)" or output_node.shape != output_shape
            ):
                raise ValueError(f"Unexpected tensor types or shapes in Unified encoder stage {filename}")
            self.sessions.append((session, input_name, output_node.name, mask_specs))
            input_shape = output_shape
    def mask_specs(self, stage):
        frames = self.encoder_frames
        specs = {
            "attention_padding": ("tensor(bool)", [1, 1, frames, frames]),
            "convolution_padding": ("tensor(bool)", [1, 1, frames]),
        }
        if stage == 0:
            for i in range(4):
                specs[f"subsampling_valid_{i}"] = ("tensor(float)", [1, 1, self.mel_frames >> i, 1])
        return specs

    def masks(self, valid):
        invalid = np.arange(self.encoder_frames) >= (valid + SUBSAMPLING - 1) // SUBSAMPLING
        masks = {
            "attention_padding": (invalid[:, None] | invalid[None, :])[None, None, :, :],
            "convolution_padding": invalid[None, None, :],
        }
        for i in range(4):
            valid_at_scale = (valid + (1 << i) - 1) >> i
            active = np.arange(self.mel_frames >> i) < valid_at_scale
            masks[f"subsampling_valid_{i}"] = active.astype(np.float32)[None, None, :, None]
        return masks

    def encode(self, features, valid):
        masks = self.masks(valid)
        for session, input_name, output_name, mask_specs in self.sessions:
            feeds = {input_name: features}
            feeds.update((name, masks[name]) for name in mask_specs)
            (features,) = session.run([output_name], feeds)
        return features

def cut_at_quiet(audio, max_samples):
    if len(audio) <= max_samples:
        return len(audio)
    # Look for a short pause near the maximum window to keep words together.
    search_samples = min(max_samples // 2, 4 * SAMPLE_RATE // 5)
    search = audio[max_samples - search_samples:max_samples]
    energy = np.mean(search[:len(search) // HOP_SIZE * HOP_SIZE].reshape(-1, HOP_SIZE) ** 2, axis=1)
    width = 8
    if len(energy) < width:
        return max_samples
    rolling = np.convolve(energy, np.full(width, 1 / width), mode="valid")
    pauses = np.flatnonzero(rolling <= max(0.002 ** 2, np.median(energy) * 0.05))
    if len(pauses):
        return max_samples - len(search) + (int(pauses[-1]) + width // 2) * HOP_SIZE
    return max_samples


def has_sustained_voiced_energy(audio):
    frame_samples = 2 * HOP_SIZE
    frames = audio[:len(audio) // frame_samples * frame_samples].reshape(-1, frame_samples)
    if len(frames) < 20:
        return False
    power = np.mean(frames ** 2, axis=1)
    noise_floor = np.percentile(power, 20)
    return np.count_nonzero(power > max(0.01 ** 2, 4 * noise_floor)) >= 12


class Unified:
    def __init__(self, model_dir, profile_dir):
        self.runtime = QnnRuntime(profile_dir)
        try:
            self.decoder = RnntDecoder(model_dir, self.runtime)
            self.encoder = QnnEncoder(model_dir, self.runtime)
            frontend_models = []
            for frames in sorted({256, 512, self.encoder.mel_frames}):
                path = model_dir / "acceleration" / f"frontend-{frames}.onnx"
                if frames <= self.encoder.mel_frames and path.is_file():
                    frontend_models.append((frames, path))
            self.frontend = (
                QnnSpectralFrontend(frontend_models, self.runtime, self.encoder.mel_frames)
                if frontend_models else Frontend(self.encoder.mel_frames)
            )
        except Exception:
            self.runtime.close()
            raise
        self.info = self.runtime.info | {
            "model": "Parakeet Unified English 0.6B (NPU encoder and decoder)",
            "languages": ["en"],
            "architecture": platform.machine(),
            "graph_devices": {"encoder": "htp", "decoder_joint": "htp", "preprocessor": self.frontend.device},
            "token_selection": "htp" if self.decoder.selects_token_on_npu else "cpu",
            "decoder_frames_per_step": self.decoder.frames_per_step,
            "encoder_stages": len(self.encoder.sessions),
            "encoder_length_masks": True,
            "encoder_precompiled": self.encoder.precompiled,
            "encoder_window_ms": self.encoder.mel_frames * HOP_SIZE * 1000 // SAMPLE_RATE,
        }

    def recognize(self, audio):
        features, valid = self.frontend.encode(audio)
        if valid < 2:
            return ""
        encoded = self.encoder.encode(features, valid)
        return self.decoder.decode(encoded, (valid + SUBSAMPLING - 1) // SUBSAMPLING)

    def recognize_window(self, clip, window_samples):
        if not np.any(clip):
            return
        text = self.recognize(clip)
        if text:
            yield 0, len(clip), text
            return
        if not has_sustained_voiced_energy(clip):
            return

        # Unified English can emit only blanks on voiced audio. Real right
        # context stays intact when silence is appended to the valid input.
        if len(clip) + TRAILING_SILENCE_SAMPLES <= window_samples:
            text = self.recognize(np.pad(clip, (0, TRAILING_SILENCE_SAMPLES)))
            if text:
                yield 0, len(clip), text
            return

        cut = cut_at_quiet(clip, len(clip) // 2)
        for start, end in ((0, cut), (cut, len(clip))):
            part = clip[start:end]
            if has_sustained_voiced_energy(part):
                padding = min(TRAILING_SILENCE_SAMPLES, window_samples - len(part))
                text = self.recognize(np.pad(part, (0, padding)))
                if text:
                    yield start, end, text

    def transcribe(self, audio_path, language):
        if language not in (None, "auto", "en"):
            raise ValueError(f"Parakeet Unified English does not support language {language!r}")
        with wave.open(str(audio_path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
                raise ValueError("Expected a 16 kHz mono signed 16-bit PCM WAV")
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

        offset = 0
        segments = []
        window_samples = self.encoder.mel_frames * HOP_SIZE
        while offset < len(audio):
            size = cut_at_quiet(audio[offset:], window_samples)
            clip = audio[offset:offset + size]
            for start, end, text in self.recognize_window(clip, window_samples):
                segments.append({
                    "start_ms": (offset + start) * 1000 // SAMPLE_RATE,
                    "end_ms": (offset + end) * 1000 // SAMPLE_RATE,
                    "text": text,
                })
            offset += size
        return {"text": " ".join(segment["text"] for segment in segments), "language": "en", "segments": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_model = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Hex" / "snapdragon" / "models-unified-qnn"
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("HEX_SNAPDRAGON_MODEL_DIR", default_model))
    model_dir = parser.parse_args().model_dir.resolve()
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
                        worker = Unified(model_dir, profile_dir)
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
            worker.runtime.close()


if __name__ == "__main__":
    main()
