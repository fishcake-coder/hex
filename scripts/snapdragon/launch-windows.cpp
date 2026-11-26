#include <windows.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {
namespace fs = std::filesystem;

int show_error(const std::wstring& message) {
    MessageBoxW(nullptr, message.c_str(), L"Hex Snapdragon NPU", MB_OK | MB_ICONERROR);
    return 1;
}

bool set_default(const wchar_t* name, const fs::path& value) {
    return GetEnvironmentVariableW(name, nullptr, 0) != 0 ||
           SetEnvironmentVariableW(name, value.c_str()) != FALSE;
}

#ifdef HEX_PACKAGED_SNAPDRAGON
bool file_exists(const fs::path& path) {
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY);
}

bool running_as_standard_user() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
    TOKEN_ELEVATION elevation{};
    DWORD bytes = 0;
    const BOOL checked = GetTokenInformation(token, TokenElevation, &elevation, sizeof(elevation), &bytes);
    CloseHandle(token);
    return checked && !elevation.TokenIsElevated;
}

fs::path ready_python(const fs::path& package, const fs::path& shared) {
    const fs::path models = shared / L"models-unified-qnn";
    if (!file_exists(models / L"encoder-manifest.json") ||
        !file_exists(models / L"streaming" / L"encoder-manifest.json")) return {};

    std::string receipt;
    std::ifstream readiness(package / L"ready");
    std::getline(readiness, receipt);
    const fs::path package_python = package / L"venv" / L"Scripts" / L"python.exe";
    const fs::path shared_python = shared / L"venv" / L"Scripts" / L"python.exe";
    if (receipt == "ready-package" && file_exists(package_python)) return package_python;
    if (receipt == "ready-shared" && file_exists(shared_python)) return shared_python;
    return {};
}

bool run_setup_script(const fs::path& bin, bool configure_runtime) {
    wchar_t system[32768];
    const UINT length = GetSystemDirectoryW(system, ARRAYSIZE(system));
    if (!length || length >= ARRAYSIZE(system)) return false;
    const fs::path powershell = fs::path(system) / L"WindowsPowerShell" / L"v1.0" / L"powershell.exe";
    const fs::path script = bin / L"install-runtime.ps1";
    if (!file_exists(script)) return false;
    std::wstring command = L"\"" + powershell.wstring() + L"\" -NoLogo -NoProfile -ExecutionPolicy Bypass";
    if (!configure_runtime) command += L" -NonInteractive";
    command += L" -File \"" + script.wstring() + L"\"";
    if (!configure_runtime) command += L" -EnableLiveTyping";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    wchar_t title[] = L"Hex Snapdragon speech setup";
    if (configure_runtime) startup.lpTitle = title;
    PROCESS_INFORMATION process{};
    const DWORD flags = configure_runtime ? CREATE_NEW_CONSOLE : CREATE_NO_WINDOW;
    if (!CreateProcessW(powershell.c_str(), command.data(), nullptr, nullptr, FALSE, flags,
                        nullptr, bin.c_str(), &startup, &process)) return false;
    CloseHandle(process.hThread);
    if (!configure_runtime) {
        CloseHandle(process.hProcess);
        return true;
    }
    const DWORD waited = WaitForSingleObject(process.hProcess, INFINITE);
    DWORD result = 1;
    GetExitCodeProcess(process.hProcess, &result);
    CloseHandle(process.hProcess);
    return waited == WAIT_OBJECT_0 && result == 0;
}
#endif
}  // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR arguments, int) {
    wchar_t local[32768];
    const DWORD local_size = GetEnvironmentVariableW(L"LOCALAPPDATA", local, ARRAYSIZE(local));
    if (local_size == 0 || local_size >= ARRAYSIZE(local)) {
        return show_error(L"Windows did not provide the local app data directory.");
    }

    wchar_t executable[32768];
    const DWORD length = GetModuleFileNameW(nullptr, executable, ARRAYSIZE(executable));
    if (length == 0 || length == ARRAYSIZE(executable)) {
        return show_error(L"Windows could not locate the Hex Snapdragon launcher.");
    }

    const fs::path bin = fs::path(executable).parent_path();
    const fs::path app = bin / L"hex.exe";
    const fs::path runtime = fs::path(local) / L"Hex" / L"snapdragon";
    fs::path python = runtime / L"venv" / L"Scripts" / L"python.exe";
    fs::path profile = bin.parent_path() / L"profile";
#ifdef HEX_PACKAGED_SNAPDRAGON
    if (!running_as_standard_user()) {
        return show_error(L"Open Hex Snapdragon from Start using your regular Windows account.");
    }
    const fs::path package = fs::path(local) / L"HexSnapdragon";
    const fs::path receipt = package / L"runtime-v1";
    profile = package / L"profile";
    python = ready_python(receipt, runtime);
    const bool repair = std::wstring_view(arguments) == L"--configure";
    if (python.empty() || repair) {
        if (MessageBoxW(nullptr,
                        L"Hex will prepare the local English speech model for your Snapdragon NPU. "
                        L"The first setup uses the internet, needs about 5 GB of free space, and can take several minutes.",
                        L"Set up Hex Snapdragon", MB_OKCANCEL | MB_ICONINFORMATION) != IDOK) return 0;
        if (!run_setup_script(bin, true)) {
            return show_error(L"Speech setup did not finish. Open Hex Snapdragon from Start to try again.");
        }
        python = ready_python(receipt, runtime);
        if (python.empty()) return show_error(L"Speech setup could not locate the verified Python runtime.");
    }
    if (!SetEnvironmentVariableW(L"HEX_APPLICATION_SUPPORT_DIR", profile.c_str()) ||
        !SetEnvironmentVariableW(L"HEX_SNAPDRAGON_DEFAULT_PROFILE", L"parakeet-unified-qnn") ||
        !SetEnvironmentVariableW(L"PYTHONDONTWRITEBYTECODE", L"1") ||
        !SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1")) {
        return show_error(L"Windows could not configure the installed Hex Snapdragon environment.");
    }
#else
    (void)arguments;
#endif
    const bool custom_worker = GetEnvironmentVariableW(L"HEX_SNAPDRAGON_WORKER", nullptr, 0) != 0;
    if (!set_default(L"HEX_APPLICATION_SUPPORT_DIR", profile) ||
        !set_default(L"HEX_SNAPDRAGON_WORKER", bin / L"select_model.py") ||
        !set_default(L"HEX_SNAPDRAGON_PYTHON", python) ||
        (custom_worker && !set_default(L"HEX_SNAPDRAGON_MODEL_DIR", runtime / L"models"))) {
        return show_error(L"Windows could not configure the Hex Snapdragon environment.");
    }

    for (const wchar_t* name : {L"HEX_SNAPDRAGON_WORKER", L"HEX_SNAPDRAGON_PYTHON"}) {
        wchar_t path[32768];
        const DWORD size = GetEnvironmentVariableW(name, path, ARRAYSIZE(path));
        if (size == 0 || size >= ARRAYSIZE(path) || GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) {
            return show_error(std::wstring(L"The Snapdragon runtime is missing. Run the setup script again.\n\n") + name);
        }
    }

#ifdef HEX_PACKAGED_SNAPDRAGON
    // The Windows speech profile is per session. The helper checks whether it
    // is active before enabling it, and failures leave ordinary dictation usable.
    run_setup_script(bin, false);
#endif
    std::wstring command = L"\"" + app.wstring() + L"\" app";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(app.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED,
                        nullptr, bin.c_str(), &startup, &process)) {
        return show_error(L"Windows could not start Hex. Error " + std::to_wstring(GetLastError()) +
                          L".\n\n" + app.wstring());
    }

    // Hand an Explorer launch's foreground permission to Hex before it opens
    // its window or asks a running copy to restore its window.
    AllowSetForegroundWindow(process.dwProcessId);
    const bool started = ResumeThread(process.hThread) != static_cast<DWORD>(-1);
    if (!started) {
        TerminateProcess(process.hProcess, 1);
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return started ? 0 : show_error(L"Windows could not resume the Hex process.");
}
