#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <msctf.h>
#include <wrl/client.h>

#include <cstdio>
#include <string_view>

namespace {
using Microsoft::WRL::ComPtr;

constexpr CLSID bridge{
    0xbe2d82d0, 0xe30d, 0x4baf, {0xa7, 0xe8, 0x44, 0x62, 0x62, 0x60, 0x4b, 0x1f}};
constexpr GUID profile{
    0xaf663182, 0x0c21, 0x46c8, {0x8f, 0x6d, 0xe3, 0xf2, 0x23, 0x4b, 0x7d, 0x48}};
constexpr wchar_t name[] = L"Hex Live Typing";
constexpr LANGID languages[]{0x0809, 0x0409};

bool supported(LANGID language) {
    for (const auto entry : languages) {
        if (language == entry) return true;
    }
    return false;
}

HRESULT register_service(ITfInputProcessorProfiles* profiles) {
    ComPtr<ITfCategoryMgr> categories;
    HRESULT status = CoCreateInstance(CLSID_TF_CategoryMgr, nullptr, CLSCTX_INPROC_SERVER,
                                      IID_PPV_ARGS(&categories));
    if (FAILED(status)) return status;
    status = profiles->Register(bridge);
    if (FAILED(status)) return status;
    // Speech has its own category and does not replace the keyboard profile.
    status = categories->RegisterCategory(bridge, GUID_TFCAT_TIP_SPEECH, bridge);
    if (SUCCEEDED(status)) {
        for (const auto language : languages) {
            status = profiles->AddLanguageProfile(
                bridge, language, profile, name, ARRAYSIZE(name) - 1, L"", 0, 0);
            if (FAILED(status)) break;
        }
    }
    if (FAILED(status)) profiles->Unregister(bridge);
    return status;
}

struct SpeechProfiles {
    bool hex_active = false;
    bool other_active = false;
};

HRESULT speech_profiles(ITfInputProcessorProfiles* profiles, ITfInputProcessorProfileMgr* manager,
                        LANGID language, SpeechProfiles& result) {
    // GetActiveProfile only accepts the keyboard category. Enumerating the
    // profiles exposes the active flag for the speech category as well.
    ComPtr<IEnumTfInputProcessorProfiles> entries;
    HRESULT status = manager->EnumProfiles(language, &entries);
    if (FAILED(status)) return status;
    if (!entries) return E_UNEXPECTED;
    TF_INPUTPROCESSORPROFILE entry{};
    ULONG count = 0;
    while ((status = entries->Next(1, &entry, &count)) == S_OK && count == 1) {
        if (entry.catid != GUID_TFCAT_TIP_SPEECH || !(entry.dwFlags & TF_IPP_FLAG_ACTIVE)) continue;
        if (entry.clsid == bridge && entry.guidProfile == profile) result.hex_active = true;
        else result.other_active = true;
    }
    if (FAILED(status)) return status;

    if (!result.hex_active) {
        LANGID active_language = 0;
        GUID active_profile{};
        result.hex_active = profiles->GetActiveLanguageProfile(bridge, &active_language, &active_profile) == S_OK &&
                            active_language == language && active_profile == profile;
    }
    return S_OK;
}

HRESULT user_profile(ITfInputProcessorProfiles* profiles, std::wstring_view command) {
    ComPtr<ITfInputProcessorProfileMgr> manager;
    HRESULT status = profiles->QueryInterface(IID_PPV_ARGS(&manager));
    if (FAILED(status)) return status;
    if (command == L"disable") {
        HRESULT result = S_OK;
        for (const auto language : languages) {
            const HRESULT disabled = manager->DeactivateProfile(
                TF_PROFILETYPE_INPUTPROCESSOR, language, bridge, profile, nullptr,
                TF_IPPMF_FORSESSION | TF_IPPMF_DISABLEPROFILE);
            if (FAILED(disabled)) result = disabled;
        }
        return result;
    }

    LANGID language = 0;
    status = profiles->GetCurrentLanguage(&language);
    if (FAILED(status)) return status;
    SpeechProfiles speech{};
    status = speech_profiles(profiles, manager.Get(), language, speech);
    if (FAILED(status)) return status;
    if (command == L"status") {
        if (speech.hex_active) {
            std::puts("Hex Live Typing speech profile is active.");
            return S_OK;
        }
        std::puts("Hex Live Typing speech profile is inactive.");
        return S_FALSE;
    }

    if (!supported(language)) {
        std::fputs("Select an English (UK or US) Windows input language first.\n", stderr);
        return E_INVALIDARG;
    }
    if (speech.hex_active) return S_OK;
    if (speech.other_active) {
        std::fputs("Another Windows speech profile is active.\n", stderr);
        return E_ACCESSDENIED;
    }
    TF_INPUTPROCESSORPROFILE keyboard{};
    const HRESULT keyboard_status = manager->GetActiveProfile(GUID_TFCAT_TIP_KEYBOARD, &keyboard);
    BOOL was_enabled = FALSE;
    status = profiles->IsEnabledLanguageProfile(bridge, language, profile, &was_enabled);
    if (FAILED(status)) return status;
    const DWORD deactivate_flags = TF_IPPMF_FORSESSION | (was_enabled ? 0 : TF_IPPMF_DISABLEPROFILE);
    status = manager->ActivateProfile(
        TF_PROFILETYPE_INPUTPROCESSOR, language, bridge, profile, nullptr,
        TF_IPPMF_FORSESSION | TF_IPPMF_ENABLEPROFILE);
    if (status != S_OK) return status;

    TF_INPUTPROCESSORPROFILE current_keyboard{};
    const HRESULT current_status = manager->GetActiveProfile(
        GUID_TFCAT_TIP_KEYBOARD, &current_keyboard);
    if (current_status != keyboard_status ||
        (current_status == S_OK && (current_keyboard.clsid != keyboard.clsid ||
                                    current_keyboard.guidProfile != keyboard.guidProfile ||
                                    current_keyboard.hkl != keyboard.hkl))) {
        manager->DeactivateProfile(TF_PROFILETYPE_INPUTPROCESSOR, language, bridge, profile, nullptr,
                                   deactivate_flags);
        if (keyboard_status == S_OK) {
            manager->ActivateProfile(keyboard.dwProfileType, keyboard.langid, keyboard.clsid,
                                     keyboard.guidProfile, keyboard.hkl, TF_IPPMF_FORSESSION);
        }
        return E_UNEXPECTED;
    }
    SpeechProfiles activated{};
    status = speech_profiles(profiles, manager.Get(), language, activated);
    if (status == S_OK && activated.hex_active) return S_OK;
    manager->DeactivateProfile(TF_PROFILETYPE_INPUTPROCESSOR, language, bridge, profile, nullptr,
                               deactivate_flags);
    std::fputs("Windows accepted the activation request but did not report the Hex speech profile as active.\n", stderr);
    return FAILED(status) ? status : E_FAIL;
}
}

int wmain(int argc, wchar_t** argv) {
    const std::wstring_view command = argc == 2 ? argv[1] : L"";
    if (command != L"register" && command != L"unregister" && command != L"enable" &&
        command != L"disable" && command != L"status") {
        std::puts("Usage: HexLiveTypingProfile register|unregister|enable|disable|status");
        return command == L"--help" ? 0 : 2;
    }
    HRESULT status = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(status)) return 1;
    {
        ComPtr<ITfInputProcessorProfiles> profiles;
        status = CoCreateInstance(CLSID_TF_InputProcessorProfiles, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(&profiles));
        if (SUCCEEDED(status)) {
            if (command == L"register") status = register_service(profiles.Get());
            else if (command == L"unregister") status = profiles->Unregister(bridge);
            else status = user_profile(profiles.Get(), command);
        }
    }
    CoUninitialize();
    if (status != S_OK) {
        std::fprintf(stderr, "Windows text service returned HRESULT 0x%08lx.\n",
                     static_cast<unsigned long>(status));
        return 1;
    }
    return 0;
}
