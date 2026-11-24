"""Select the Snapdragon model for the Hex worker and local launchers."""

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import NamedTemporaryFile


DEFAULT_PROFILE = "whisper"
HYBRID_GRAPH_DEVICES = {"encoder": "htp", "decoder_joint": "cpu", "preprocessor": "cpu"}
PROFILES = {
    "whisper": ("whisper_qnn.py", "models", {"precision": "w8a16"}),
    "whisper-float": ("whisper_qnn.py", "models-float", {"precision": "float"}),
    "parakeet-tdt": ("parakeet_tdt.py", "models-tdt", {"graph_devices": HYBRID_GRAPH_DEVICES}),
    "parakeet-unified-qnn": (
        "unified_rnnt.py",
        "models-unified-qnn",
        {
            "model": "Parakeet Unified English 0.6B (NPU encoder and decoder)",
            "graph_devices": {"encoder": "htp", "decoder_joint": "htp", "preprocessor": "cpu"},
            "encoder_length_masks": True,
            "encoder_precompiled": True,
        },
    ),
}
SCRIPTS = Path(__file__).resolve().parent
LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home()))
RUNTIME = LOCAL_APP_DATA / "Hex" / "snapdragon"
APP_PROFILE = Path(os.environ.get("HEX_APPLICATION_SUPPORT_DIR") or (
    LOCAL_APP_DATA / "HexSnapdragonBuild" / "profile"
))
SELECTION = APP_PROFILE / "snapdragon-model.txt"


def active_profile():
    override = os.environ.get("HEX_SNAPDRAGON_PROFILE")
    if override:
        profile, source = override, "environment"
    else:
        try:
            profile, source = SELECTION.read_text(encoding="utf-8").strip(), "saved"
        except FileNotFoundError:
            profile, source = DEFAULT_PROFILE, "default"
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown Snapdragon model profile {profile!r}. "
            "Use Hex Snapdragon.cmd snapdragon-model list."
        )
    return profile, source


def verify(profile):
    worker, models, identity = PROFILES[profile]
    script = SCRIPTS / worker
    directory = RUNTIME / models
    if not script.is_file() or not directory.is_dir():
        raise ValueError(f"Install the Snapdragon model profile {profile!r} before selecting it.")
    env = dict(os.environ, HEX_SNAPDRAGON_MODEL_DIR=str(directory))
    env.pop("HEX_SNAPDRAGON_PROFILE_DIR", None)
    response = subprocess.run(
        [sys.executable, "-u", str(script)],
        env=env,
        input='{"id":1,"method":"initialize"}' + "\n",
        stdout=subprocess.PIPE,
        encoding="utf-8",
        timeout=120,
    )
    if response.returncode != 0:
        raise RuntimeError(f"The Snapdragon worker exited with code {response.returncode}.")
    reply = json.loads(response.stdout)
    if not isinstance(reply, dict) or reply.get("id") != 1:
        raise RuntimeError("The Snapdragon worker returned an invalid initialize reply.")
    if "result" not in reply:
        raise RuntimeError(reply.get("error", "The Snapdragon worker returned no model identity."))
    result = reply["result"]
    expected = {
        "backend": "qnn",
        "device": "htp",
        "provider": "QNNExecutionProvider",
        "cpu_fallback": False,
        **identity,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"The worker did not initialize the expected QNN model profile {profile!r}.")
    return result["model"]


def save_profile(profile):
    APP_PROFILE.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=APP_PROFILE, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(profile + "\n")
            file.close()
            temporary.replace(SELECTION)
        finally:
            file.close()
            temporary.unlink(missing_ok=True)


def model_command(arguments):
    parser = argparse.ArgumentParser(prog="Hex Snapdragon.cmd snapdragon-model", description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("list", help="List supported Snapdragon model profiles")
    commands.add_parser("status", help="Show the model profile for new processes")
    use = commands.add_parser("use", help="Verify and save a model profile")
    use.add_argument("profile", choices=PROFILES)
    options = parser.parse_args(arguments)
    if options.action == "list":
        print(*PROFILES, sep="\n")
    elif options.action == "status":
        profile, source = active_profile()
        print(f"{profile} ({source})")
        for variable in ("HEX_SNAPDRAGON_WORKER", "HEX_SNAPDRAGON_MODEL_DIR"):
            if os.environ.get(variable):
                print(f"{variable} overrides the selected profile.")
    else:
        name = verify(options.profile)
        save_profile(options.profile)
        print(f"Selected {options.profile} ({name}). Restart Hex to use it for dictation.")
    return 0


def run_worker():
    try:
        profile, _ = active_profile()
        worker, models, _ = PROFILES[profile]
        script = SCRIPTS / worker
        if not script.is_file():
            raise FileNotFoundError(f"Snapdragon worker missing at {script}. Run the launcher installer again.")
        if not os.environ.get("HEX_SNAPDRAGON_MODEL_DIR"):
            os.environ["HEX_SNAPDRAGON_MODEL_DIR"] = str(RUNTIME / models)
    except (OSError, ValueError) as error:
        # Failed selection still replies to Hex's initialize handshake.
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        try:
            request_id = json.loads(sys.stdin.readline()).get("id")
        except (AttributeError, ValueError):
            request_id = None
        print(json.dumps({"id": request_id, "error": str(error)}), flush=True)
        return 1
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "snapdragon-model":
        try:
            sys.exit(model_command(sys.argv[2:]))
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            print(f"Snapdragon model selection failed: {error}", file=sys.stderr)
            sys.exit(1)
    sys.exit(run_worker())
