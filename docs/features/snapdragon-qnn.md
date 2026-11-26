# Snapdragon QNN dictation

Hex Snapdragon runs speech recognition through the Qualcomm QNN execution
provider on the Hexagon NPU. The Windows installer defaults to streaming
Parakeet Unified English. The source-build launcher defaults to Whisper
Large v3 Turbo.

## Windows installer

Download **Hex-Snapdragon-Setup-arm64.exe** from
[GitHub Releases](https://github.com/fishcake-coder/hex/releases). Run it on a
Qualcomm Snapdragon PC with Windows 11 24H2 or later, then open **Hex Snapdragon**
from Start. First-time setup needs an internet connection and about 5 GB of
free disk space for the runtime and model. Subsequent dictation runs locally.
The installer is unsigned. Its release includes a SHA-256 checksum.

To build the installer from source on Windows ARM64, install Inno Setup 6.7.4
and run `scripts/build-snapdragon-installer.ps1 -Version 0.1.0`.

## Source build

```ts
Hex Snapdragon NPU                           // dictation.snapdragon-qnn
├── Open from Start -> shows Hex and starts the listener after the QNN HTP handshake
├── Open again -> restores the window for the running listener
├── Hold and release -> transcribes a finished clip, then processes and pastes as usual
├── hex dictation stop --print -> uses the same listener and prints the result
├── hex status --json -> names the verified model, language setting, and QNN HTP device
├── snapdragon-model use <profile> -> verifies the installed QNN model and saves it for the next launch
└── transcribe <file> [--language <code>]   // cli.transcribe-snapdragon-qnn
    └── Sends the decoded pieces in order to this command's own QNN worker
```

Run [the Windows ARM64 build script](../../scripts/build-snapdragon.ps1),
[the QNN setup script](../../scripts/snapdragon/setup.ps1), and
[the Start menu launcher installer](../../scripts/snapdragon/install-launcher.ps1).
Open **Hex Snapdragon NPU** from the Windows Start menu.
In PowerShell, use [Hex Snapdragon.cmd](<../../Hex Snapdragon.cmd>) for CLI commands.
With a Whisper profile selected, `& '.\Hex Snapdragon.cmd' transcribe .\recording.wav --language he`
transcribes a Hebrew recording.
Use the same launcher for `status` and dictation control commands so they
address its isolated profile.
Launch at login is unavailable in the Snapdragon session.

## Models and switching

- `whisper` uses Qualcomm's precompiled 8-bit weights and 16-bit activations.
  Both model graphs run on the NPU. It supports automatic detection and explicit
  languages, including Hebrew. Install with [the Whisper setup script](../../scripts/snapdragon/setup.ps1).
  [Qualcomm model card](https://aihub.qualcomm.com/compute/models/whisper_large_v3_turbo_quantized)
- `whisper-float` uses Qualcomm's FP16 export with the same language controls.
  Both graphs run on the NPU. Pass `-Precision float` to the Whisper setup script
  to install the roughly 2.2 GB model in `models-float`.
  [Qualcomm export](https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo)
- `parakeet-tdt` uses an NPU encoder and a CPU TDT decoder. It automatically
  recognizes 25 European languages, including English. Install with
  [the TDT setup script](../../scripts/snapdragon/setup-parakeet-tdt.ps1).
  [NVIDIA model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
  [OpenWritr NPU export](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s),
  [ONNX decoder export](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
- `parakeet-unified-qnn` uses Parakeet Unified English 0.6B with a
  length-masked NPU encoder and an NPU decoder. The setup builds 12 encoder
  stages with a 12.8-second audio window by default. Install and compile the
  QNN contexts with [the Unified setup script](../../scripts/snapdragon/setup-parakeet-unified.ps1).
  [NVIDIA model](https://huggingface.co/nvidia/parakeet-unified-en-0.6b),
  [ONNX export](https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx)

Quit Hex, choose an installed model, then reopen **Hex Snapdragon NPU** from
Start. The selection is shared with the file command.

```powershell
& '.\Hex Snapdragon.cmd' snapdragon-model list
& '.\Hex Snapdragon.cmd' snapdragon-model use parakeet-unified-qnn
& '.\Hex Snapdragon.cmd' snapdragon-model status
```

Use `snapdragon-model use whisper` to switch back. `snapdragon-model status`
reports the selection for the next process. `status --json` reports the live listener.
Set `HEX_SNAPDRAGON_PROFILE` in the calling environment for a temporary
selection. Advanced overrides use `HEX_SNAPDRAGON_WORKER`,
`HEX_SNAPDRAGON_PYTHON`, and `HEX_SNAPDRAGON_MODEL_DIR`. Models and the
shared Python environment live under `%LOCALAPPDATA%\Hex\snapdragon`.

`HEX_SNAPDRAGON_LANGUAGE` defaults to `auto` for desktop dictation.
`hex transcribe --language` sets the language for one file with a compatible
model. General displays the active model and language setting.

The configured app keeps the worker warm while listening. Start standard Hex
to use its catalog and automatic model switching.

Whole-clip dictation and the file command send short-lived mono 16 kHz WAVs
over a private local handoff. Hex removes each WAV when the request finishes.
An abrupt process kill can leave a `hex-snapdragon-*.wav` in the user's
temporary directory. Long clips cut near a quiet point to fit the model window.
The configured process skips Hex's optional Silero detector.
The ordinary low-level silence and short-tap checks still apply to dictation.

## Parakeet Unified Live Text

Quit Hex before updating an existing build. Build the native binaries from
this branch, run Unified setup with `-Streaming`, and reinstall the launcher.
The worker detects the compiled streaming encoder alongside the Unified model.
Turn on **Live text** in Hex's Labs to display provisional words beside the
dictation indicator.

```powershell
& '.\scripts\build-snapdragon.ps1'
& '.\scripts\snapdragon\setup-parakeet-unified.ps1' -Streaming
& '.\scripts\snapdragon\install-launcher.ps1'
```

Hex sends incremental 16 kHz PCM through the local worker pipe. The buffered
encoder runs on QNN with 5.6 seconds of left context,
560 ms chunks, and 560 ms of lookahead. The decoder carries its recurrent
state between chunks. Releasing the shortcut flushes the last words and
continues through the usual output and paste flow. Hex can transcribe the
recorded clip offline if it cannot use the streaming result. Set
`HEX_STREAMING=0` before launch to use the whole-clip path.

On the Snapdragon X Plus X1P42100, the 560 ms configuration showed
provisional text before release on 102 of 108 recorded English clips. The
median first preview followed 1.76 seconds of audio. The median worker
finish took 105 ms after the last input, with 9.24% word error rate across
the same recordings used for the whole-clip comparison below.

A 320 ms experiment showed text after a median 1.28 seconds of audio. The
560 ms configuration used 68 seconds of worker CPU time across the corpus,
compared with 119 seconds at 320 ms. It also had slightly lower word error
rate. These timings cover recorded audio replay and exclude microphone
capture, app startup, and pasting.

On an existing Unified installation, run
`scripts/snapdragon/setup-parakeet-unified.ps1 -Optimize`, then rerun
`scripts/snapdragon/install-launcher.ps1` and restart Hex. Setup compiles the
extra QNN graphs under the model's `acceleration` directory. The worker uses
them automatically when present. The encoder and its model weights are reused.

Optimized Unified scores four frames per decoder call and chooses the token
IDs on the NPU. It recomputes scores from the first emitted token when the
predictor state changes. The NPU also computes spectral and mel energy using
the smallest available window that fits the audio. Windows under 1.6 seconds
use the CPU frontend to avoid dispatch overhead. The CPU handles audio
loading, windowing, pre-emphasis, log and feature normalization, the decoding
loop, and text assembly. Other profiles use their original audio preparation
and token selection.
Every graph assigned to QNN has CPU fallback disabled.

A missing model, worker startup failure, or failed transcription reports
an error. Separate local API requests that explicitly select a catalog model
still use that catalog model. The standalone file command opens its own worker
and can consume NPU memory alongside the listener.

## Typing inside the focused field

**Live typing in textbox** is on by default. Register and enable its Windows
text service with the commands below. Hex keeps an explicitly saved Labs
preference when you upgrade.

```powershell
& '.\scripts\windows-live-typing\build.ps1'
# Run Register from an Administrator PowerShell in the same checkout.
& '.\scripts\windows-live-typing\install.ps1' -Action Register
# Run Enable from your ordinary PowerShell.
& '.\scripts\windows-live-typing\install.ps1' -Action Enable
& '.\Hex Snapdragon.cmd' labs on live-typing
```

A supported focused field displays the provisional transcript while the
same Snapdragon worker runs recognition. In Toggle mode, you can edit or
move the caret mid-dictation and continue speaking there. Stopping commits
the latest span. Escape clears the current preview and leaves your edits.
The Windows text service supports English (UK and US) and Hebrew keyboard
layouts. The selected speech model still determines the recognition language.
To upgrade an existing text service, rebuild it, then run
`install.ps1 -Action Update` in an elevated PowerShell. Reopen any text
editor or browser that was already using the service.
The Windows text service receives transcript text and no microphone audio. Use
`-Action Disable` from your regular user session to deactivate the speech
profile and `-Action Unregister` from an elevated PowerShell to remove its
registration.

## Local verification

Across two replay rounds of the same 108 English recordings, the installed
optimized worker averaged 11.80 seconds of CPU time, down from 16.30 seconds.
Its wall time ranged from 13.95 to 25.92 seconds. Both workers scored 9.60%
word error rate. One word differed in one clip. The four-frame decoder alone
matched all transcripts and segment timestamps. These times exclude worker
startup, app capture, and pasting.

On a Snapdragon X Plus X1P42100, Parakeet Unified English recorded QNN
accelerator execution for all 12 encoder stages and its RNNT predictor/joint
graph. QNN CPU fallback was disabled. The CPU computed the audio features,
selected tokens, and assembled the text. The precompiled Unified install was
1.26 GB. An unprofiled warm worker returned a 1.26-second English clip in 91 ms.

The following comparison used 108 disjoint English clips, 36 each from AMI
meetings, Earnings22 calls and LibriSpeech test-other. The clips contained
717.09 seconds of audio and 1,916 normalized reference words. WER counts
substitutions, deletions and insertions after English normalization with
whisper-normalizer 0.1.12. Times cover the warm worker requests and exclude
model loading, microphone capture and pasting.

| Model on this laptop | Word errors | WER | Transcription time |
| --- | ---: | ---: | ---: |
| Parakeet Unified English (NPU encoder and decoder) | 184 | 9.60% | 18.74 s |
| Parakeet TDT v3 (NPU encoder, CPU decoder) | 205 | 10.70% | 17.47 s |
| Whisper Large v3 Turbo w8a16 (NPU encoder and decoder) | 211 | 11.01% | 50.47 s |

The 72-clip holdout was separate from the 36 clips used during development.
Unified led on meetings, TDT on earnings calls and Whisper on audiobooks.


For quantized Whisper on one Snapdragon X Plus X1P42100, QNN recorded accelerator execution time
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
excludes app startup, microphone capture, and pasting. On the same system,
the Parakeet TDT v3 encoder recorded six HTP runs. Four warm repetitions of
the English fixture took about 0.10 seconds per WAV request.

Implementation lives in [the QNN adapter](../../src/transcription/snapdragon.rs),
[the Windows listener](../../src/windows/dictation.rs), and
[the file CLI](../../src/control/cli/transcribe.rs). Native verification should
check the HTP execution evidence as well as the transcript and status output.
