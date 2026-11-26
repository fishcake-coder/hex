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

HRESULT contains_guid(IEnumGUID* entries, REFGUID expected, bool& found) {
    if (!entries) return E_UNEXPECTED;
    GUID entry{};
    HRESULT status = S_OK;
    while ((status = entries->Next(1, &entry, nullptr)) == S_OK) {
        if (entry == expected) {
            found = true;
            return S_OK;
        }
    }
    return FAILED(status) ? status : S_OK;
}

struct Registration {
    bool service = false;
    bool speech = false;
    bool profiles[ARRAYSIZE(languages)]{};
};

HRESULT registered_profiles(ITfInputProcessorProfileMgr* manager, Registration& registration) {
    ComPtr<IEnumTfInputProcessorProfiles> entries;
    HRESULT status = manager->EnumProfiles(0, &entries);
    if (FAILED(status)) return status;
    if (!entries) return E_UNEXPECTED;
    TF_INPUTPROCESSORPROFILE entry{};
    while ((status = entries->Next(1, &entry, nullptr)) == S_OK) {
        if (entry.clsid != bridge || entry.guidProfile != profile) continue;
        for (size_t index = 0; index < ARRAYSIZE(languages); ++index) {
            if (entry.langid == languages[index]) registration.profiles[index] = true;
        }
    }
    return FAILED(status) ? status : S_OK;
}

HRESULT read_registration(ITfInputProcessorProfiles* profiles, ITfCategoryMgr* categories,
                          Registration& registration) {
    ComPtr<IEnumGUID> services;
    HRESULT status = profiles->EnumInputProcessorInfo(&services);
    if (FAILED(status)) return status;
    status = contains_guid(services.Get(), bridge, registration.service);
    if (FAILED(status)) return status;
    ComPtr<IEnumGUID> speech;
    status = categories->EnumItemsInCategory(GUID_TFCAT_TIP_SPEECH, &speech);
    if (FAILED(status)) return status;
    status = contains_guid(speech.Get(), bridge, registration.speech);
    if (FAILED(status)) return status;
    ComPtr<ITfInputProcessorProfileMgr> manager;
    status = profiles->QueryInterface(IID_PPV_ARGS(&manager));
    return FAILED(status) ? status : registered_profiles(manager.Get(), registration);
}

HRESULT unregister_service(ITfInputProcessorProfiles* profiles, ITfCategoryMgr* categories,
                           const Registration& registration) {
    HRESULT result = S_OK;
    for (size_t index = 0; index < ARRAYSIZE(languages); ++index) {
        if (!registration.profiles[index]) continue;
        const HRESULT status = profiles->RemoveLanguageProfile(bridge, languages[index], profile);
        if (FAILED(status)) result = status;
    }
    if (registration.speech) {
        const HRESULT status = categories->UnregisterCategory(bridge, GUID_TFCAT_TIP_SPEECH, bridge);
        if (FAILED(status)) result = status;
    }
    if (registration.service) {
        const HRESULT status = profiles->Unregister(bridge);
        // Unregister also removes every profile and category for this CLSID.
        if (SUCCEEDED(status) || SUCCEEDED(result)) result = status;
    }
    return result;
}

HRESULT change_registration(ITfInputProcessorProfiles* profiles, bool install) {
    ComPtr<ITfCategoryMgr> categories;
    HRESULT status = CoCreateInstance(CLSID_TF_CategoryMgr, nullptr, CLSCTX_INPROC_SERVER,
                                      IID_PPV_ARGS(&categories));
    if (FAILED(status)) return status;
    Registration existing{};
    status = read_registration(profiles, categories.Get(), existing);
    if (FAILED(status)) return status;
    if (!install) return unregister_service(profiles, categories.Get(), existing);

    Registration added{};
    if (!existing.service) {
        status = profiles->Register(bridge);
        if (FAILED(status)) return status;
        added.service = true;
    }
    // Speech has its own category and does not replace the keyboard profile.
    if (!existing.speech) {
        status = categories->RegisterCategory(bridge, GUID_TFCAT_TIP_SPEECH, bridge);
        added.speech = SUCCEEDED(status);
    }
    if (SUCCEEDED(status)) {
        for (size_t index = 0; index < ARRAYSIZE(languages); ++index) {
            if (existing.profiles[index]) continue;
            status = profiles->AddLanguageProfile(
                bridge, languages[index], profile, name, ARRAYSIZE(name) - 1, nullptr, 0, 0);
            if (FAILED(status)) break;
            added.profiles[index] = true;
        }
    }
    if (FAILED(status)) {
        unregister_service(profiles, categories.Get(), added);
        return status;
    }
    return S_OK;
}

struct SpeechProfiles {
    bool hex_active = false;
    bool other_active = false;
};

HRESULT disable_profiles(ITfInputProcessorProfiles* profiles, ITfInputProcessorProfileMgr* manager) {
    Registration registration{};
    HRESULT status = registered_profiles(manager, registration);
    if (FAILED(status)) return status;
    LANGID active_language = 0;
    GUID active_profile{};
    const bool active = profiles->GetActiveLanguageProfile(bridge, &active_language, &active_profile) == S_OK &&
                        active_profile == profile;
    HRESULT result = S_OK;
    for (size_t index = 0; index < ARRAYSIZE(languages); ++index) {
        if (!registration.profiles[index]) continue;
        const auto language = languages[index];
        if (active && language == active_language) {
            status = manager->DeactivateProfile(TF_PROFILETYPE_INPUTPROCESSOR, language, bridge, profile,
                                                nullptr, TF_IPPMF_FORSESSION | TF_IPPMF_DISABLEPROFILE);
            if (FAILED(status)) result = status;
        } else {
            BOOL enabled = FALSE;
            status = profiles->IsEnabledLanguageProfile(bridge, language, profile, &enabled);
            if (status == S_OK && enabled) status = profiles->EnableLanguageProfile(bridge, language, profile, FALSE);
            if (FAILED(status)) result = status;
        }
    }
    return result;
}

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
    if (command == L"disable") return disable_profiles(profiles, manager.Get());

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
            if (command == L"register" || command == L"unregister") {
                status = change_registration(profiles.Get(), command == L"register");
            }
            else status = user_profile(profiles.Get(), command);
        }
    }
    CoUninitialize();
    if (command == L"status" && status == S_FALSE) return 1;
    if (status != S_OK) {
        std::fprintf(stderr, "Windows text service returned HRESULT 0x%08lx.\n",
                     static_cast<unsigned long>(status));
        return 1;
    }
    return 0;
}
