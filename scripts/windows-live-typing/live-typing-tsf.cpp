#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <msctf.h>
#include <initguid.h>
#include <inputscope.h>
#include <oleauto.h>
#include <wrl/client.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <utility>

namespace {
using Microsoft::WRL::ComPtr;

constexpr CLSID kBridgeClsid{
    0xbe2d82d0, 0xe30d, 0x4baf, {0xa7, 0xe8, 0x44, 0x62, 0x62, 0x60, 0x4b, 0x1f}};
constexpr wchar_t kWindowClass[] = L"HexLiveTyping.Tsf2";
constexpr UINT kCancelInactive = WM_APP + 1;
constexpr DWORD kMaxTextBytes = 16 * 1024;
constexpr auto kInputSettle = std::chrono::milliseconds(120);

enum class Operation : ULONG_PTR {
    begin = 1, update, commit, cancel, probe, accept, begin_owned, abandon
};
enum class Outcome : LRESULT { declined = 0, applied, uncertain, reanchor, selection_pending };

HMODULE module = nullptr;
std::atomic<long> live_objects{0};
std::atomic<long> server_locks{0};
SRWLOCK window_classes_lock = SRWLOCK_INIT;
unsigned window_class_users = 0;

bool same_object(IUnknown* left, IUnknown* right) {
    if (!left || !right) return left == right;
    ComPtr<IUnknown> left_identity;
    ComPtr<IUnknown> right_identity;
    return SUCCEEDED(left->QueryInterface(IID_PPV_ARGS(&left_identity))) &&
           SUCCEEDED(right->QueryInterface(IID_PPV_ARGS(&right_identity))) &&
           left_identity.Get() == right_identity.Get();
}

HWND focused_window() {
    const HWND foreground = GetForegroundWindow();
    if (!foreground || GetWindowThreadProcessId(foreground, nullptr) != GetCurrentThreadId()) {
        return nullptr;
    }
    const HWND focused = GetFocus();
    return focused && GetAncestor(focused, GA_ROOT) == foreground ? focused : nullptr;
}

bool same_text(ITfRange* range, TfEditCookie cookie, const std::wstring& expected) {
    std::wstring observed(expected.size() + 1, L'\0');
    ULONG count = 0;
    return SUCCEEDED(range->GetText(cookie, 0, observed.data(), static_cast<ULONG>(observed.size()), &count)) &&
           count == expected.size() && observed.compare(0, count, expected) == 0;
}

bool same_position(ITfRange* first, ITfRange* second, TfEditCookie cookie, TfAnchor anchor) {
    LONG comparison = 0;
    return SUCCEEDED(first->CompareStart(cookie, second, anchor, &comparison)) && comparison == 0;
}

ComPtr<ITfRange> current_selection(ITfContext* context, TfEditCookie cookie) {
    ComPtr<ITfRange> range;
    TF_SELECTION selection{};
    ULONG count = 0;
    if (FAILED(context->GetSelection(cookie, TF_DEFAULT_SELECTION, 1, &selection, &count)) || count != 1) {
        return range;
    }
    range.Attach(selection.range);
    return range;
}

wchar_t beside_caret(ITfRange* caret, TfEditCookie cookie, bool before) {
    ComPtr<ITfRange> range;
    if (FAILED(caret->Clone(&range)) || !range) return L'\0';
    LONG shifted = 0;
    const HRESULT result = before ? range->ShiftStart(cookie, -1, &shifted, nullptr)
                                  : range->ShiftEnd(cookie, 1, &shifted, nullptr);
    wchar_t character = L'\0';
    ULONG count = 0;
    return SUCCEEDED(result) && shifted &&
        SUCCEEDED(range->GetText(cookie, 0, &character, 1, &count)) && count == 1 ? character : L'\0';
}

bool spaced_letter(wchar_t character) {
    // Avoid introducing spaces in scripts that commonly run words together.
    if ((character >= 0x0e00 && character <= 0x109f) || // Thai, Lao, Myanmar and Tibetan
        (character >= 0x1780 && character <= 0x17ff)) return false; // Khmer
    WORD type = 0;
    WORD script = 0;
    return GetStringTypeW(CT_CTYPE1, &character, 1, &type) && (type & (C1_ALPHA | C1_DIGIT)) &&
        GetStringTypeW(CT_CTYPE3, &character, 1, &script) &&
        !(script & (C3_IDEOGRAPH | C3_HIRAGANA | C3_KATAKANA));
}

bool permits_space_after(wchar_t character) {
    switch (character) {
        case L'.':
        case L',':
        case L'!':
        case L'?':
        case L';':
        case L':':
        case L')':
        case L']':
        case L'}':
            return true;
        default:
            return spaced_letter(character);
    }
}

bool input_scope_allows_preview(ITfContext* context, ITfRange* range, TfEditCookie cookie, HWND window) {
    wchar_t window_class[32]{};
    const bool standard_edit = GetClassNameW(window, window_class, ARRAYSIZE(window_class)) &&
        (_wcsicmp(window_class, L"Edit") == 0 || _wcsicmp(window_class, L"RichEdit20A") == 0 ||
         _wcsicmp(window_class, L"RichEdit20W") == 0 || _wcsicmp(window_class, L"RichEdit50W") == 0);
    if (standard_edit && ((GetWindowLongPtrW(window, GWL_STYLE) & ES_PASSWORD) != 0 ||
                          SendMessageW(window, EM_GETPASSWORDCHAR, 0, 0) != 0)) {
        return false;
    }
    TF_STATUS status{};
    if (FAILED(context->GetStatus(&status)) || (status.dwDynamicFlags & TF_SD_READONLY) != 0) {
        return false;
    }
    ComPtr<ITfReadOnlyProperty> property;
    if (FAILED(context->GetAppProperty(GUID_PROP_INPUTSCOPE, &property)) || !property) {
        return standard_edit;
    }
    VARIANT value{};
    const HRESULT value_status = property->GetValue(cookie, range, &value);
    if (value.vt == VT_EMPTY && (value_status == S_FALSE || value_status == E_FAIL)) {
        // RichEdit returns E_FAIL for its default, unspecified input scope.
        // Only the standard controls with the password check above can opt in
        // without an explicit TSF scope.
        return standard_edit;
    }
    ComPtr<ITfInputScope> scopes;
    if (SUCCEEDED(value_status) && value.vt == VT_UNKNOWN && value.punkVal) {
        value.punkVal->QueryInterface(IID_PPV_ARGS(&scopes));
    }
    VariantClear(&value);
    if (!scopes) return false;

    InputScope* items = nullptr;
    UINT count = 0;
    if (FAILED(scopes->GetInputScopes(&items, &count))) return false;
    bool allowed = true;
    for (UINT index = 0; index < count; ++index) {
        switch (items[index]) {
            case IS_PASSWORD:
            case IS_PRIVATE:
            case IS_NUMERIC_PASSWORD:
            case IS_NUMERIC_PIN:
            case IS_ALPHANUMERIC_PIN:
            case IS_ALPHANUMERIC_PIN_SET:
                allowed = false;
                break;
            default:
                break;
        }
    }
    CoTaskMemFree(items);
    return allowed;
}

class Bridge;

class EditSession final : public ITfEditSession {
public:
    EditSession(Bridge* bridge, Operation operation, std::uint64_t id, std::wstring text);
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** result) override;
    ULONG STDMETHODCALLTYPE AddRef() override { return ++references_; }
    ULONG STDMETHODCALLTYPE Release() override;
    HRESULT STDMETHODCALLTYPE DoEditSession(TfEditCookie cookie) override;
    Outcome outcome() const { return outcome_; }

private:
    ~EditSession();
    Bridge* bridge_;
    Operation operation_;
    std::uint64_t id_;
    std::wstring text_;
    Outcome outcome_ = Outcome::declined;
    std::atomic<ULONG> references_{1};
};

class Bridge final : public ITfTextInputProcessor, public ITfCompositionSink,
                     public ITfThreadMgrEventSink, public ITfThreadFocusSink,
                     public ITfTextEditSink {
public:
    Bridge() { ++live_objects; }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** result) override {
        if (!result) return E_POINTER;
        *result = nullptr;
        if (id == IID_IUnknown || id == __uuidof(ITfTextInputProcessor)) {
            *result = static_cast<ITfTextInputProcessor*>(this);
        } else if (id == __uuidof(ITfCompositionSink)) {
            *result = static_cast<ITfCompositionSink*>(this);
        } else if (id == __uuidof(ITfThreadMgrEventSink)) {
            *result = static_cast<ITfThreadMgrEventSink*>(this);
        } else if (id == __uuidof(ITfThreadFocusSink)) {
            *result = static_cast<ITfThreadFocusSink*>(this);
        } else if (id == __uuidof(ITfTextEditSink)) {
            *result = static_cast<ITfTextEditSink*>(this);
        } else {
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    ULONG STDMETHODCALLTYPE AddRef() override { return ++references_; }

    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG remaining = --references_;
        if (!remaining) delete this;
        return remaining;
    }

    HRESULT STDMETHODCALLTYPE Activate(ITfThreadMgr* manager, TfClientId client) override {
        if (!manager || manager_ || client == TF_CLIENTID_NULL) return E_INVALIDARG;
        manager_ = manager;
        client_ = client;
        HRESULT status = manager_.As(&source_);
        if (SUCCEEDED(status)) {
            status = source_->AdviseSink(__uuidof(ITfThreadMgrEventSink),
                static_cast<ITfThreadMgrEventSink*>(this), &thread_manager_cookie_);
        }
        if (SUCCEEDED(status)) {
            status = source_->AdviseSink(__uuidof(ITfThreadFocusSink),
                static_cast<ITfThreadFocusSink*>(this), &thread_focus_cookie_);
        }
        if (SUCCEEDED(status) && !create_window()) status = E_FAIL;
        if (FAILED(status)) Deactivate();
        return status;
    }

    HRESULT STDMETHODCALLTYPE Deactivate() override {
        if (capture_id_) {
            try { request(Operation::cancel, capture_id_, {}); }
            catch (...) { uncertain_id_ = capture_id_; }
        }
        clear_capture();
        if (message_window_) destroy_window();
        if (source_) {
            if (thread_focus_cookie_ != TF_INVALID_COOKIE) {
                source_->UnadviseSink(thread_focus_cookie_);
                thread_focus_cookie_ = TF_INVALID_COOKIE;
            }
            if (thread_manager_cookie_ != TF_INVALID_COOKIE) {
                source_->UnadviseSink(thread_manager_cookie_);
                thread_manager_cookie_ = TF_INVALID_COOKIE;
            }
        }
        source_.Reset();
        manager_.Reset();
        client_ = TF_CLIENTID_NULL;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnInitDocumentMgr(ITfDocumentMgr*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE OnUninitDocumentMgr(ITfDocumentMgr* doc) override {
        if (document_ && same_object(document_.Get(), doc)) queue_cancel();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnSetFocus(ITfDocumentMgr* doc, ITfDocumentMgr*) override {
        if (document_ && !same_object(document_.Get(), doc)) queue_cancel();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnPushContext(ITfContext* context) override {
        if (context_ && !same_object(context_.Get(), context) && !still_focused()) queue_cancel();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnPopContext(ITfContext* context) override {
        if (context_ && same_object(context_.Get(), context)) queue_cancel();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnSetThreadFocus() override { return S_OK; }

    HRESULT STDMETHODCALLTYPE OnKillThreadFocus() override {
        if (capture_id_) queue_cancel();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnEndEdit(ITfContext* context, TfEditCookie, ITfEditRecord* edit) override {
        if ((!anchor_ && !composition_) || in_request_ || external_edit_ || !edit ||
            !same_object(context_.Get(), context)) return S_OK;
        BOOL own_session = FALSE;
        if (SUCCEEDED(context->InWriteSession(client_, &own_session)) && own_session) return S_OK;

        BOOL selection_changed = FALSE;
        if (SUCCEEDED(edit->GetSelectionStatus(&selection_changed)) && selection_changed) {
            external_edit_ = true;
        } else {
            ComPtr<IEnumTfRanges> changes;
            ComPtr<ITfRange> changed;
            ULONG count = 0;
            external_edit_ = SUCCEEDED(edit->GetTextAndPropertyUpdates(TF_GTP_INCL_TEXT, nullptr, 0, &changes)) &&
                changes && SUCCEEDED(changes->Next(1, &changed, &count)) && count != 0;
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnCompositionTerminated(TfEditCookie cookie, ITfComposition* composition) override {
        if (ending_composition_ || !composition_ || !same_object(composition_.Get(), composition)) return S_OK;
        if (!external_edit_ && (focus_lost_ || !still_focused())) {
            const auto id = capture_id_;
            ComPtr<ITfRange> range;
            try {
                if (FAILED(composition->GetRange(&range)) || !range || !same_text(range.Get(), cookie, preview_) ||
                    FAILED(range->SetText(cookie, 0, L"", 0))) uncertain_id_ = id;
            } catch (...) {
                uncertain_id_ = id;
            }
            clear_capture();
            return S_OK;
        }
        // An editor can terminate the composition as part of the user's edit.
        // The text now belongs to the editor, including any replacements.
        external_edit_ = true;
        composition_.Reset();
        return S_OK;
    }

    Outcome apply(Operation operation, std::uint64_t id, const std::wstring& text, TfEditCookie cookie) {
        if (id != capture_id_) return Outcome::declined;
        if (operation == Operation::cancel) return cancel(cookie);
        if (operation == Operation::abandon) {
            // A terminated client cannot confirm which edits reached the
            // document. Release its composition without replacing any text.
            if (!end_composition(cookie)) return Outcome::uncertain;
            clear_capture();
            return Outcome::applied;
        }
        if (focus_lost_ || !still_focused()) return failure();
        if (operation == Operation::accept) {
            const bool had_preview = composition_ || external_edit_;
            const Outcome accepted = accept_external_edit(cookie);
            if (accepted != Outcome::reanchor) return accepted;
            reanchor_pending_ |= had_preview;
            // A preview sent concurrently with the input hook must wait for
            // the key or click to reach the editor after the hook returns.
            input_settles_at_ = std::chrono::steady_clock::now() + kInputSettle;
            return Outcome::applied;
        }
        if (reanchor_pending_) {
            reanchor_pending_ = false;
            return Outcome::reanchor;
        }
        if (operation == Operation::commit && text.empty() && !composition_ && !external_edit_) {
            clear_capture();
            return Outcome::applied;
        }
        if ((operation == Operation::update || operation == Operation::commit) &&
            std::chrono::steady_clock::now() < input_settles_at_) return Outcome::selection_pending;
        ComPtr<ITfRange> selection = current_selection(context_.Get(), cookie);
        BOOL collapsed = FALSE;
        if (!selection || FAILED(selection->IsEmpty(cookie, &collapsed)) ||
            !input_scope_allows_preview(context_.Get(), selection.Get(), cookie, target_window_)) {
            return failure();
        }
        if (operation == Operation::begin) {
            if (!collapsed) return Outcome::declined;
            anchor_ = selection;
            return Outcome::applied;
        }
        if (external_edit_) return accept_external_edit(cookie);

        ComPtr<ITfRange> range;
        // Some text stores perform ordinary edits outside a TSF edit session.
        // Verify the text and caret again even when the sink saw no changes.
        if (composition_) {
            if (FAILED(composition_->GetRange(&range)) || !range) return failure();
            if (!collapsed || !same_position(selection.Get(), range.Get(), cookie, TF_ANCHOR_END) ||
                !same_text(range.Get(), cookie, preview_)) return accept_external_edit(cookie);
        } else if (anchor_ && (!collapsed || !same_position(selection.Get(), anchor_.Get(), cookie, TF_ANCHOR_START))) {
            return accept_external_edit(cookie);
        }

        if (!collapsed) return Outcome::selection_pending;

        if (!composition_) {
            ComPtr<ITfContextComposition> composition_api;
            ComPtr<IEnumITfCompositionView> compositions;
            ComPtr<ITfCompositionView> other;
            ULONG count = 0;
            if (FAILED(context_.As(&composition_api)) ||
                FAILED(composition_api->EnumCompositions(&compositions)) || !compositions ||
                FAILED(compositions->Next(1, &other, &count))) return failure();
            if (count != 0) return Outcome::selection_pending;
            if (operation == Operation::probe || text.empty()) return Outcome::applied;
            if (!anchor_) {
                left_neighbor_ = beside_caret(selection.Get(), cookie, true);
                right_neighbor_ = beside_caret(selection.Get(), cookie, false);
            }
            anchor_ = selection;
            if (FAILED(composition_api->StartComposition(cookie, anchor_.Get(), this, &composition_)) ||
                !composition_ || FAILED(composition_->GetRange(&range)) || !range ||
                !same_text(range.Get(), cookie, preview_) ||
                !same_position(selection.Get(), range.Get(), cookie, TF_ANCHOR_END)) return failure();
        }
        if (operation == Operation::probe) return Outcome::applied;
        if (focus_lost_ || !still_focused()) return failure();

        std::wstring next_preview(text);
        if (!text.empty()) {
            if (permits_space_after(left_neighbor_) && spaced_letter(text.front())) {
                next_preview.insert(next_preview.begin(), L' ');
            }
            if (spaced_letter(right_neighbor_) && permits_space_after(text.back())) {
                next_preview.push_back(L' ');
            }
        }

        // Once SetText is called, a failed document callback may have applied
        // part of the transaction. Preserve that distinction for the caller.
        ComPtr<ITfComposition> owned = composition_;
        if (FAILED(range->SetText(cookie, 0, next_preview.data(), static_cast<LONG>(next_preview.size()))) ||
            capture_id_ != id || !same_object(owned.Get(), composition_.Get())) {
            uncertain_id_ = id;
            return Outcome::uncertain;
        }
        preview_.swap(next_preview);
        ComPtr<ITfRange> caret;
        if (focus_lost_ || !still_focused() || FAILED(owned->GetRange(&caret)) || !caret ||
            FAILED(caret->Collapse(cookie, TF_ANCHOR_END)) || !set_selection(cookie, caret.Get()) ||
            capture_id_ != id || !same_object(owned.Get(), composition_.Get())) {
            uncertain_id_ = id;
            return Outcome::uncertain;
        }
        if (operation == Operation::update) return Outcome::applied;
        if (!end_composition(cookie)) {
            uncertain_id_ = capture_id_;
            return Outcome::uncertain;
        }
        clear_capture();
        return Outcome::applied;
    }

private:
    ~Bridge() {
        // TSF must call Deactivate before releasing its final reference.
        if (message_window_) destroy_window();
        --live_objects;
    }

    static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
        if (message == WM_NCCREATE) {
            const auto* setup = reinterpret_cast<const CREATESTRUCTW*>(lparam);
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(setup->lpCreateParams));
        }
        Bridge* bridge = reinterpret_cast<Bridge*>(GetWindowLongPtrW(window, GWLP_USERDATA));
        if (message == WM_NCDESTROY) SetWindowLongPtrW(window, GWLP_USERDATA, 0);
        if (bridge && (message == WM_COPYDATA || message == kCancelInactive)) {
            bridge->AddRef();
            Outcome result = Outcome::declined;
            try {
                if (message == WM_COPYDATA) {
                    result = bridge->on_copydata(reinterpret_cast<const COPYDATASTRUCT*>(lparam));
                } else if (bridge->capture_id_ && bridge->focus_lost_) {
                    result = bridge->request(Operation::cancel, bridge->capture_id_, {});
                }
            } catch (...) {
                result = bridge->failure();
            }
            bridge->Release();
            return static_cast<LRESULT>(result);
        }
        return DefWindowProcW(window, message, wparam, lparam);
    }

    bool create_window() {
        AcquireSRWLockExclusive(&window_classes_lock);
        bool class_ready = window_class_users > 0;
        if (!class_ready) {
            WNDCLASSEXW window_class{};
            window_class.cbSize = sizeof(window_class);
            window_class.hInstance = module;
            window_class.lpfnWndProc = window_proc;
            window_class.lpszClassName = kWindowClass;
            class_ready = RegisterClassExW(&window_class) != 0;
        }
        if (class_ready) {
            message_window_ = CreateWindowExW(0, kWindowClass, L"", 0, 0, 0, 0, 0,
                HWND_MESSAGE, nullptr, module, this);
            if (message_window_) {
                ++window_class_users;
            } else if (!window_class_users) {
                UnregisterClassW(kWindowClass, module);
            }
        }
        ReleaseSRWLockExclusive(&window_classes_lock);
        return message_window_ != nullptr;
    }

    void destroy_window() {
        const HWND old_window = message_window_;
        message_window_ = nullptr;
        DestroyWindow(old_window);
        AcquireSRWLockExclusive(&window_classes_lock);
        if (--window_class_users == 0) UnregisterClassW(kWindowClass, module);
        ReleaseSRWLockExclusive(&window_classes_lock);
    }

    Outcome on_copydata(const COPYDATASTRUCT* copy) {
        if (!copy || !copy->lpData || copy->cbData < sizeof(std::uint64_t) ||
            copy->cbData > sizeof(std::uint64_t) + kMaxTextBytes || (copy->cbData % 2) != 0 ||
            copy->dwData < static_cast<ULONG_PTR>(Operation::begin) ||
            copy->dwData > static_cast<ULONG_PTR>(Operation::begin_owned)) {
            return Outcome::declined;
        }
        const auto op = static_cast<Operation>(copy->dwData);
        std::uint64_t id = 0;
        std::memcpy(&id, copy->lpData, sizeof(id));
        if (!id) return Outcome::declined;
        const auto* bytes = static_cast<const char*>(copy->lpData);
        if (op == Operation::begin_owned) {
            DWORD owner = 0;
            if (copy->cbData != sizeof(id) + sizeof(owner)) return Outcome::declined;
            std::memcpy(&owner, bytes + sizeof(id), sizeof(owner));
            return owner ? request(Operation::begin, id, {}, owner) : Outcome::declined;
        }
        const size_t characters = (copy->cbData - sizeof(id)) / sizeof(wchar_t);
        if (op != Operation::update && op != Operation::commit && characters) {
            return Outcome::declined;
        }
        std::wstring text(characters, L'\0');
        if (characters) {
            std::memcpy(text.data(), bytes + sizeof(id), characters * sizeof(wchar_t));
            if (text.find(L'\0') != std::wstring::npos) return failure();
        }
        return request(op, id, text);
    }

    Outcome failure() const {
        return ((capture_id_ && capture_id_ == uncertain_id_) || (composition_ && !preview_.empty()))
            ? Outcome::uncertain : Outcome::declined;
    }

    bool find_focus(ComPtr<ITfDocumentMgr>& doc, ComPtr<ITfContext>& context, HWND& window) {
        window = focused_window();
        if (!window || !manager_ || FAILED(manager_->GetFocus(&doc)) || !doc ||
            FAILED(doc->GetTop(&context)) || !context) return false;
        ComPtr<ITfContextView> view;
        HWND view_window = nullptr;
        return SUCCEEDED(context->GetActiveView(&view)) && view &&
            SUCCEEDED(view->GetWnd(&view_window)) && view_window &&
            (view_window == window || IsChild(view_window, window));
    }

    bool still_focused() {
        ComPtr<ITfDocumentMgr> document;
        ComPtr<ITfContext> context;
        HWND window = nullptr;
        return find_focus(document, context, window) && window == target_window_ &&
            same_object(document.Get(), document_.Get()) && same_object(context.Get(), context_.Get());
    }

    Outcome request(Operation operation, std::uint64_t id, const std::wstring& text, DWORD owner = 0) {
        if (in_request_) return failure();
        if (operation == Operation::begin) {
            if (capture_id_) {
                if (!capture_owner_ || WaitForSingleObject(capture_owner_, 0) != WAIT_OBJECT_0) {
                    return Outcome::declined;
                }
                const auto abandoned = request(Operation::abandon, capture_id_, {});
                if (abandoned != Outcome::applied) return abandoned;
            }
            if (!find_focus(document_, context_, target_window_)) {
                clear_capture();
                return Outcome::declined;
            }
            if (owner) {
                capture_owner_ = OpenProcess(SYNCHRONIZE, FALSE, owner);
                if (!capture_owner_ || WaitForSingleObject(capture_owner_, 0) != WAIT_TIMEOUT) {
                    clear_capture();
                    return Outcome::declined;
                }
            }
            capture_id_ = id;
            if (SUCCEEDED(context_.As(&edit_source_))) {
                edit_source_->AdviseSink(__uuidof(ITfTextEditSink), static_cast<ITfTextEditSink*>(this),
                    &edit_cookie_);
            }
        } else if (id != capture_id_) {
            return id == uncertain_id_ ? Outcome::uncertain : Outcome::declined;
        } else if (operation != Operation::cancel && operation != Operation::abandon &&
                   (focus_lost_ || !still_focused())) {
            queue_cancel();
            return failure();
        }

        ComPtr<ITfContext> context = context_;
        ComPtr<EditSession> edit;
        edit.Attach(new (std::nothrow) EditSession(this, operation, id, text));
        if (!edit) {
            if (operation == Operation::begin) clear_capture();
            return failure();
        }
        HRESULT outcome = E_FAIL;
        in_request_ = true;
        const HRESULT request = context->RequestEditSession(
            client_, edit.Get(), TF_ES_SYNC | TF_ES_READWRITE, &outcome);
        in_request_ = false;
        Outcome result = SUCCEEDED(request) && outcome == S_OK ? edit->outcome() : failure();
        if (operation == Operation::begin && result != Outcome::applied) clear_capture();
        return result;
    }

    void queue_cancel() {
        focus_lost_ = true;
        if (message_window_) PostMessageW(message_window_, kCancelInactive, 0, 0);
    }

    bool set_selection(TfEditCookie cookie, ITfRange* range) {
        TF_SELECTION selection{range, {TF_AE_NONE, FALSE}};
        ComPtr<ITfContext> context = context_;
        return context && SUCCEEDED(context->SetSelection(cookie, 1, &selection));
    }

    bool end_composition(TfEditCookie cookie) {
        ComPtr<ITfComposition> composition = composition_;
        if (!composition) return true;
        ending_composition_ = true;
        const HRESULT ended = composition->EndComposition(cookie);
        ending_composition_ = false;
        return SUCCEEDED(ended);
    }

    Outcome accept_external_edit(TfEditCookie cookie) {
        external_edit_ = true;
        if (!end_composition(cookie)) {
            uncertain_id_ = capture_id_;
            return Outcome::uncertain;
        }
        composition_.Reset();
        anchor_.Reset();
        preview_.clear();
        left_neighbor_ = right_neighbor_ = L'\0';
        external_edit_ = false;
        return Outcome::reanchor;
    }

    Outcome cancel(TfEditCookie cookie) {
        if (!composition_) {
            clear_capture();
            return Outcome::applied;
        }
        ComPtr<ITfRange> range;
        const bool owned = !external_edit_ && SUCCEEDED(composition_->GetRange(&range)) && range &&
            same_text(range.Get(), cookie, preview_);
        ComPtr<ITfRange> selection;
        const bool focused = !focus_lost_ && still_focused();
        BOOL collapsed = FALSE;
        if (owned && focused) {
            selection = current_selection(context_.Get(), cookie);
        }
        const bool restore = selection && SUCCEEDED(selection->IsEmpty(cookie, &collapsed)) && collapsed &&
            same_position(selection.Get(), range.Get(), cookie, TF_ANCHOR_END);
        if (!owned || (focused && !restore)) {
            const Outcome accepted = accept_external_edit(cookie);
            if (accepted != Outcome::reanchor) return accepted;
            clear_capture();
            return Outcome::applied;
        }
        const std::uint64_t id = capture_id_;
        ComPtr<ITfComposition> composition = composition_;
        if (FAILED(range->SetText(cookie, 0, L"", 0)) || capture_id_ != id ||
            !same_object(composition.Get(), composition_.Get())) {
            uncertain_id_ = id;
            if (end_composition(cookie)) clear_capture();
            return Outcome::uncertain;
        }
        preview_.clear();
        if (!end_composition(cookie)) {
            uncertain_id_ = capture_id_;
            return Outcome::uncertain;
        }
        if (restore && !focus_lost_ && still_focused() && anchor_) set_selection(cookie, anchor_.Get());
        clear_capture();
        return Outcome::applied;
    }

    void clear_capture() {
        if (edit_source_ && edit_cookie_ != TF_INVALID_COOKIE) edit_source_->UnadviseSink(edit_cookie_);
        edit_cookie_ = TF_INVALID_COOKIE;
        edit_source_.Reset();
        composition_.Reset();
        anchor_.Reset();
        context_.Reset();
        document_.Reset();
        target_window_ = nullptr;
        if (capture_owner_) CloseHandle(capture_owner_);
        capture_owner_ = nullptr;
        capture_id_ = 0;
        focus_lost_ = false;
        external_edit_ = false;
        reanchor_pending_ = false;
        input_settles_at_ = {};
        preview_.clear();
        left_neighbor_ = right_neighbor_ = L'\0';
    }

    std::atomic<ULONG> references_{1};
    ComPtr<ITfThreadMgr> manager_;
    ComPtr<ITfSource> source_;
    TfClientId client_ = TF_CLIENTID_NULL;
    DWORD thread_manager_cookie_ = TF_INVALID_COOKIE;
    DWORD thread_focus_cookie_ = TF_INVALID_COOKIE;
    HWND message_window_ = nullptr;
    HWND target_window_ = nullptr;
    ComPtr<ITfDocumentMgr> document_;
    ComPtr<ITfContext> context_;
    ComPtr<ITfSource> edit_source_;
    DWORD edit_cookie_ = TF_INVALID_COOKIE;
    ComPtr<ITfRange> anchor_;
    ComPtr<ITfComposition> composition_;
    std::wstring preview_;
    wchar_t left_neighbor_ = L'\0';
    wchar_t right_neighbor_ = L'\0';
    HANDLE capture_owner_ = nullptr;
    std::uint64_t capture_id_ = 0;
    std::uint64_t uncertain_id_ = 0;
    std::chrono::steady_clock::time_point input_settles_at_{};
    bool focus_lost_ = false;
    bool external_edit_ = false;
    bool reanchor_pending_ = false;
    bool in_request_ = false;
    bool ending_composition_ = false;
};

EditSession::EditSession(Bridge* bridge, Operation operation, std::uint64_t id, std::wstring text)
    : bridge_(bridge), operation_(operation), id_(id), text_(std::move(text)) {
    bridge_->AddRef();
}

EditSession::~EditSession() { bridge_->Release(); }

HRESULT EditSession::QueryInterface(REFIID id, void** result) {
    if (!result) return E_POINTER;
    *result = nullptr;
    if (id != IID_IUnknown && id != __uuidof(ITfEditSession)) return E_NOINTERFACE;
    *result = static_cast<ITfEditSession*>(this);
    AddRef();
    return S_OK;
}

ULONG EditSession::Release() {
    const ULONG remaining = --references_;
    if (!remaining) delete this;
    return remaining;
}

HRESULT EditSession::DoEditSession(TfEditCookie cookie) {
    try {
        outcome_ = bridge_->apply(operation_, id_, text_, cookie);
    } catch (...) {
        outcome_ = Outcome::uncertain;
    }
    return S_OK;
}

class Factory final : public IClassFactory {
public:
    Factory() { ++live_objects; }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** result) override {
        if (!result) return E_POINTER;
        *result = nullptr;
        if (id != IID_IUnknown && id != IID_IClassFactory) return E_NOINTERFACE;
        *result = static_cast<IClassFactory*>(this);
        AddRef();
        return S_OK;
    }

    ULONG STDMETHODCALLTYPE AddRef() override { return ++references_; }

    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG remaining = --references_;
        if (!remaining) delete this;
        return remaining;
    }

    HRESULT STDMETHODCALLTYPE CreateInstance(IUnknown* outer, REFIID id, void** result) override {
        if (!result) return E_POINTER;
        *result = nullptr;
        if (outer) return CLASS_E_NOAGGREGATION;
        auto* bridge = new (std::nothrow) Bridge();
        if (!bridge) return E_OUTOFMEMORY;
        const HRESULT created = bridge->QueryInterface(id, result);
        bridge->Release();
        return created;
    }

    HRESULT STDMETHODCALLTYPE LockServer(BOOL lock) override {
        if (lock) ++server_locks;
        else --server_locks;
        return S_OK;
    }

private:
    ~Factory() { --live_objects; }
    std::atomic<ULONG> references_{1};
};
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) module = instance;
    return TRUE;
}

STDAPI DllGetClassObject(REFCLSID clsid, REFIID iid, void** result) {
    if (!result) return E_POINTER;
    *result = nullptr;
    if (clsid != kBridgeClsid) return CLASS_E_CLASSNOTAVAILABLE;
    auto* factory = new (std::nothrow) Factory();
    if (!factory) return E_OUTOFMEMORY;
    const HRESULT status = factory->QueryInterface(iid, result);
    factory->Release();
    return status;
}

STDAPI DllCanUnloadNow() {
    return live_objects.load() == 0 && server_locks.load() == 0 ? S_OK : S_FALSE;
}
