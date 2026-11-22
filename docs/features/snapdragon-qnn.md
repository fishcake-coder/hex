# Snapdragon QNN dictation

This local Windows ARM64 build runs Whisper Large v3 Turbo on the Hexagon NPU.
Setup pins Qualcomm AI Hub's `v0.66.0` Snapdragon X Elite export, a precompiled
QNN ONNX model with 8-bit weights and 16-bit activations (`w8a16`). See the
[Qualcomm AI Hub model page](https://aihub.qualcomm.com/compute/models/whisper_large_v3_turbo_quantized).
The launcher opts in for the process it starts.

```ts
Hex Snapdragon.cmd                           // dictation.snapdragon-qnn
├── Launch -> starts the Windows listener after the QNN HTP handshake
├── Hold and release -> transcribes a finished clip, then processes and pastes as usual
├── hex dictation stop --print -> uses the same listener and prints the result
├── hex status --json -> names the verified model, language setting, and QNN HTP device
└── transcribe <file> [--language <code>]   // cli.transcribe-snapdragon-qnn
    └── Sends the decoded pieces in order to this command's own QNN worker
```

Run [the Windows ARM64 build script](../../scripts/build-snapdragon.ps1) and
[the QNN setup script](../../scripts/snapdragon/setup.ps1) first. Then
double-click [Hex Snapdragon.cmd](<../../Hex Snapdragon.cmd>) to open Hex.
In PowerShell, you can run the launcher with CLI arguments, for example
`& '.\Hex Snapdragon.cmd' transcribe .\recording.wav --language he`.
Use the same launcher for `status` and dictation control commands so they
address its isolated profile.
Launch at login is unavailable in the Snapdragon session. The Start Menu
shortcut can point to the launcher to open it with the QNN environment.

The launcher provides the absolute worker path in `HEX_SNAPDRAGON_WORKER`.
`HEX_SNAPDRAGON_PYTHON` and `HEX_SNAPDRAGON_MODEL_DIR` override the Python
environment and model directory under `%LOCALAPPDATA%\Hex\snapdragon`.
`HEX_SNAPDRAGON_LANGUAGE` sets a language code for desktop dictation.
It defaults to `auto`. `hex transcribe --language` overrides it for that file.
General displays the active model and the launcher-selected language.
The configured app skips the GGUF download step and keeps the worker warm
while listening. The usual Windows model picker and automatic model switches
are unavailable in this session. Clear `HEX_SNAPDRAGON_WORKER` and restart
to select a catalog model.

The listener and file command send short-lived mono 16 kHz WAVs over a
private local handoff. Hex removes each WAV when the request finishes.
An abrupt process kill can leave a `hex-snapdragon-*.wav` in the user's
temporary directory. Long clips cut near a quiet point to fit the 30-second
Whisper window. The configured process skips Hex's optional Silero detector.
The ordinary low-level silence and short-tap checks still apply to dictation.
QNN works on finished audio, so streaming live text is unavailable.

The Whisper encoder and decoder graphs run on the NPU through the QNN
execution provider. The CPU handles WAV loading, audio windowing, log-Mel
features, input quantization, and selection and decoding of text and
timestamp tokens. The worker disables CPU fallback for both model graphs.

A missing model, worker startup failure, or failed transcription reports
an error. Hex does not switch to CPU speech inference or download a substitute
model. Separate local API requests that explicitly select a catalog model
still use that catalog model. The standalone file command opens its own worker
and can consume NPU memory alongside the listener.

## Local verification

On one Snapdragon X Plus X1P42100, QNN recorded accelerator execution time
for six encoder runs and 240 decoder steps during the fixture checks. The
ARM64 worker matched a 2.24-second English fixture with automatic language
detection and a 4.7-second Hebrew fixture in both automatic and explicit
Hebrew modes. All 14 sentences in a 35.20-second English fixture matched
after normalizing case and punctuation. The last segment ran from 31.98 to
35.00 seconds. A two-second silent WAV returned empty text. The English
clips were generated with text-to-speech. These checks cover a small set
of clips in two languages on one device.

With profiling off and the worker already warm, three repeats of the same
2.24-second English fixture took 0.40–0.42 seconds each from the WAV request
to the transcription reply. The median was 0.41 seconds. This measurement
excludes app startup, microphone capture, and pasting.

Implementation lives in [the QNN adapter](../../src/transcription/snapdragon.rs),
[the Windows listener](../../src/windows/dictation.rs), and
[the file CLI](../../src/control/cli/transcribe.rs). Native verification should
check the HTP execution evidence as well as the transcript and status output.
