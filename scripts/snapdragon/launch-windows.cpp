#include <windows.h>

#include <filesystem>
#include <string>

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
}  // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
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
    if (!set_default(L"HEX_APPLICATION_SUPPORT_DIR", bin.parent_path() / L"profile") ||
        !set_default(L"HEX_SNAPDRAGON_WORKER", bin / L"whisper_qnn.py") ||
        !set_default(L"HEX_SNAPDRAGON_PYTHON", runtime / L"venv" / L"Scripts" / L"python.exe") ||
        !set_default(L"HEX_SNAPDRAGON_MODEL_DIR", runtime / L"models")) {
        return show_error(L"Windows could not configure the Hex Snapdragon environment.");
    }

    for (const wchar_t* name : {L"HEX_SNAPDRAGON_WORKER", L"HEX_SNAPDRAGON_PYTHON"}) {
        wchar_t path[32768];
        const DWORD size = GetEnvironmentVariableW(name, path, ARRAYSIZE(path));
        if (size == 0 || size >= ARRAYSIZE(path) || GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) {
            return show_error(std::wstring(L"The Snapdragon runtime is missing. Run the setup script again.\n\n") + name);
        }
    }

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
