#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <msctf.h>
#include <initguid.h>
#include <inputscope.h>
#include <oleauto.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <utility>

namespace {
using Microsoft::WRL::ComPtr;

constexpr CLSID kBridgeClsid{
    0xbe2d82d0, 0xe30d, 0x4baf, {0xa7, 0xe8, 0x44, 0x62, 0x62, 0x60, 0x4b, 0x1f}};
constexpr wchar_t kWindowClass[] = L"HexLiveTyping.Tsf";
constexpr UINT kCancelInactive = WM_APP + 1;
constexpr DWORD kMaxTextBytes = 16 * 1024;

enum class Operation : ULONG_PTR { begin = 1, update, commit, cancel };
enum class Outcome : LRESULT { declined = 0, applied = 1, uncertain = 2 };

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

ComPtr<ITfRange> collapsed_selection(ITfContext* context, TfEditCookie cookie) {
    ComPtr<ITfRange> range;
    TF_SELECTION selection{};
    ULONG count = 0;
    if (FAILED(context->GetSelection(cookie, TF_DEFAULT_SELECTION, 1, &selection, &count)) || count != 1) {
        return range;
    }
    range.Attach(selection.range);
    BOOL empty = FALSE;
    if (!range || FAILED(range->IsEmpty(cookie, &empty)) || !empty) range.Reset();
    return range;
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
                     public ITfThreadMgrEventSink, public ITfThreadFocusSink {
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

    HRESULT STDMETHODCALLTYPE OnCompositionTerminated(TfEditCookie cookie, ITfComposition* composition) override {
        if (ending_composition_ || !composition_ || !same_object(composition_.Get(), composition)) return S_OK;
        try {
            ComPtr<ITfRange> range;
            if (FAILED(composition_->GetRange(&range)) || !range || !same_text(range.Get(), cookie, preview_) ||
                FAILED(range->SetText(cookie, 0, L"", 0))) {
                uncertain_id_ = capture_id_;
            }
        } catch (...) {
            uncertain_id_ = capture_id_;
        }
        clear_capture();
        return S_OK;
    }

    Outcome apply(Operation operation, std::uint64_t id, const std::wstring& text, TfEditCookie cookie) {
        if (id != capture_id_) return Outcome::declined;
        if (operation == Operation::cancel) return cancel(cookie);
        if (focus_lost_ || !still_focused()) return failure();
        if (operation == Operation::begin) {
            anchor_ = collapsed_selection(context_.Get(), cookie);
            if (anchor_ && input_scope_allows_preview(context_.Get(), anchor_.Get(), cookie, target_window_)) {
                return Outcome::applied;
            }
            return Outcome::declined;
        }

        ComPtr<ITfRange> selection = collapsed_selection(context_.Get(), cookie);
        if (!selection || !input_scope_allows_preview(context_.Get(), selection.Get(), cookie, target_window_)) {
            return failure();
        }
        ComPtr<ITfRange> range;
        std::wstring next_preview(text);
        if (!composition_) {
            if (!anchor_ || !same_position(selection.Get(), anchor_.Get(), cookie, TF_ANCHOR_START)) {
                return Outcome::declined;
            }
            ComPtr<ITfContextComposition> composition_api;
            ComPtr<IEnumITfCompositionView> compositions;
            ComPtr<ITfCompositionView> other;
            ULONG count = 0;
            if (FAILED(context_.As(&composition_api)) ||
                FAILED(composition_api->EnumCompositions(&compositions)) || !compositions ||
                FAILED(compositions->Next(1, &other, &count)) || count != 0 ||
                FAILED(composition_api->StartComposition(cookie, anchor_.Get(), this, &composition_)) ||
                !composition_) {
                return Outcome::declined;
            }
        }
        if (FAILED(composition_->GetRange(&range)) || !range || !same_text(range.Get(), cookie, preview_) ||
            !same_position(selection.Get(), range.Get(), cookie, TF_ANCHOR_END)) {
            return failure();
        }

        // Once SetText is called, a failed document callback may have applied
        // part of the transaction. Preserve that distinction for the caller.
        ComPtr<ITfComposition> owned = composition_;
        if (FAILED(range->SetText(cookie, 0, text.data(), static_cast<LONG>(text.size()))) ||
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
            copy->dwData > static_cast<ULONG_PTR>(Operation::cancel)) {
            return Outcome::declined;
        }
        const auto op = static_cast<Operation>(copy->dwData);
        const size_t characters = (copy->cbData - sizeof(std::uint64_t)) / sizeof(wchar_t);
        if ((op == Operation::begin || op == Operation::cancel) && characters) return Outcome::declined;
        std::uint64_t id = 0;
        std::memcpy(&id, copy->lpData, sizeof(id));
        if (!id) return Outcome::declined;
        std::wstring text(characters, L'\0');
        if (characters) {
            const auto* bytes = static_cast<const char*>(copy->lpData);
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

    Outcome request(Operation operation, std::uint64_t id, const std::wstring& text) {
        if (in_request_) return failure();
        if (operation == Operation::begin) {
            if (capture_id_) return Outcome::declined;
            if (!find_focus(document_, context_, target_window_)) {
                clear_capture();
                return Outcome::declined;
            }
            capture_id_ = id;
        } else if (id != capture_id_) {
            return id == uncertain_id_ ? Outcome::uncertain : Outcome::declined;
        } else if (operation != Operation::cancel && (focus_lost_ || !still_focused())) {
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

    Outcome cancel(TfEditCookie cookie) {
        if (!composition_) {
            clear_capture();
            return Outcome::applied;
        }
        const std::uint64_t id = capture_id_;
        ComPtr<ITfRange> range;
        const bool owned = SUCCEEDED(composition_->GetRange(&range)) && range &&
            same_text(range.Get(), cookie, preview_);
        ComPtr<ITfRange> selection;
        if (owned && !focus_lost_ && still_focused()) {
            selection = collapsed_selection(context_.Get(), cookie);
        }
        const bool restore = selection && same_position(selection.Get(), range.Get(), cookie, TF_ANCHOR_END);
        if (!owned || FAILED(range->SetText(cookie, 0, L"", 0)) || capture_id_ != id) {
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
        composition_.Reset();
        anchor_.Reset();
        context_.Reset();
        document_.Reset();
        target_window_ = nullptr;
        capture_id_ = 0;
        focus_lost_ = false;
        preview_.clear();
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
    ComPtr<ITfRange> anchor_;
    ComPtr<ITfComposition> composition_;
    std::wstring preview_;
    std::uint64_t capture_id_ = 0;
    std::uint64_t uncertain_id_ = 0;
    bool focus_lost_ = false;
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
