//! Delivers the latest provisional transcript to the focused Windows text service.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{HWND, POINT};
use windows_sys::Win32::System::DataExchange::COPYDATASTRUCT;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    FindWindowExW, GUITHREADINFO, GetForegroundWindow, GetGUIThreadInfo, GetWindowThreadProcessId,
    HWND_MESSAGE, IsChild, SMTO_ABORTIFHUNG, SMTO_BLOCK, SMTO_ERRORONEXIT, SendMessageTimeoutW,
    WM_COPYDATA, WindowFromPoint,
};

use crate::dictation::live_text::{ClientDemand, client_demand};
use crate::transcription::engine::Partial;

const CLASS: windows::core::PCWSTR = windows::core::w!("HexLiveTyping.Tsf2");
const MAX_UNITS: usize = 8_192;
const MESSAGE_MS: u32 = 65;
const BEGIN: usize = 1;
const UPDATE: usize = 2;
const COMMIT: usize = 3;
const CANCEL: usize = 4;
const PROBE: usize = 5;
const ACCEPT: usize = 6;
const REQUEST_TIMEOUT: Duration = Duration::from_millis(240);
const SELECTION_RECHECK: Duration = Duration::from_millis(120);
const INPUT_SETTLE: Duration = Duration::from_millis(120);

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) struct Focus {
    window: isize,
    control: isize,
    thread: u32,
    process: u32,
}

impl Focus {
    pub(super) fn current() -> Option<Self> {
        let mut info = GUITHREADINFO {
            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        // SAFETY: these calls only read the foreground GUI thread and its focus.
        let window = unsafe { GetForegroundWindow() };
        if window.is_null() {
            return None;
        }
        let foreground_thread = unsafe { GetWindowThreadProcessId(window, std::ptr::null_mut()) };
        if foreground_thread == 0
            || unsafe { GetGUIThreadInfo(foreground_thread, &mut info) } == 0
            || info.hwndFocus.is_null()
        {
            return None;
        }
        let mut process = 0;
        let thread = unsafe { GetWindowThreadProcessId(info.hwndFocus, &mut process) };
        (thread != 0 && process != 0 && unsafe { GetForegroundWindow() } == window).then_some(
            Self {
                window: window as isize,
                control: info.hwndFocus as isize,
                thread,
                process,
            },
        )
    }

    fn bridge(self) -> Option<isize> {
        let mut after = std::ptr::null_mut();
        // SAFETY: the name is a null-terminated UTF-16 constant. The HWNDs
        // are only used with Win32 lookup and bounded SendMessage.
        loop {
            after = unsafe { FindWindowExW(HWND_MESSAGE, after, CLASS.as_ptr(), std::ptr::null()) };
            if after.is_null() {
                return None;
            }
            let mut process = 0;
            let thread = unsafe { GetWindowThreadProcessId(after, &mut process) };
            if thread == self.thread && process == self.process {
                return Some(after as isize);
            }
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Response {
    Declined,
    Applied,
    Uncertain,
    Reanchor,
    SelectionPending,
}

fn send(window: isize, capture: u64, operation: usize, text: &str) -> Response {
    let mut payload = Vec::with_capacity(8 + text.len().min(MAX_UNITS) * 2);
    payload.extend_from_slice(&capture.to_le_bytes());
    for unit in text.encode_utf16() {
        payload.extend_from_slice(&unit.to_le_bytes());
        if payload.len() > 8 + 2 * MAX_UNITS {
            return Response::Declined;
        }
    }
    let packet = COPYDATASTRUCT {
        dwData: operation,
        cbData: payload.len() as u32,
        lpData: payload.as_mut_ptr().cast(),
    };
    let mut result = 0;
    // SAFETY: WM_COPYDATA copies this caller-owned packet across the process
    // boundary. Its buffers stay valid until the synchronous send returns.
    let delivered = unsafe {
        SendMessageTimeoutW(
            window as HWND,
            WM_COPYDATA,
            0,
            (&raw const packet) as isize,
            SMTO_ABORTIFHUNG | SMTO_BLOCK | SMTO_ERRORONEXIT,
            MESSAGE_MS,
            &mut result,
        )
    };
    if delivered == 0 {
        return Response::Uncertain;
    }
    match result {
        0 => Response::Declined,
        1 => Response::Applied,
        3 => Response::Reanchor,
        4 => Response::SelectionPending,
        _ => Response::Uncertain,
    }
}

static EDITOR_INPUT: Mutex<Weak<ManualInput>> = Mutex::new(Weak::new());

#[derive(Default)]
struct InputState {
    armed: bool,
    uncertain: bool,
    settle_by: Option<Instant>,
}

struct ManualInput {
    focus: Focus,
    window: isize,
    id: u64,
    // Only one native request at a time. In particular, a preview must be
    // acknowledged before the hook can hand its text to the editor.
    state: Mutex<InputState>,
}

impl ManualInput {
    fn send(&self, operation: usize, text: &str) -> Response {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.uncertain {
            // The editor may have already processed the input. Retry only
            // relinquishing the span, never cancel or replace its text.
            if Focus::current() != Some(self.focus)
                || send(self.window, self.id, ACCEPT, "") != Response::Applied
            {
                return Response::Uncertain;
            }
            state.uncertain = false;
            state.settle_by = Some(Instant::now() + INPUT_SETTLE);
        }
        let response = send(self.window, self.id, operation, text);
        if operation == UPDATE && response == Response::Applied {
            state.armed = !text.is_empty();
        } else if response == Response::Reanchor || operation == CANCEL || operation == COMMIT {
            state.armed = false;
        }
        response
    }

    fn wait_for_input(&self, deadline: Instant) {
        let settle_by = self
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .settle_by;
        if let Some(settle_by) = settle_by {
            let last_send = deadline - Duration::from_millis(u64::from(MESSAGE_MS));
            std::thread::sleep(
                settle_by
                    .min(last_send)
                    .saturating_duration_since(Instant::now()),
            );
        }
    }
}

/// Runs before the hook forwards an edit or caret movement. Some controls
/// keep returning their cached TSF composition after their editor changes.
pub(in crate::windows) fn before_editor_input(point: Option<POINT>, edits: impl FnOnce() -> bool) {
    let input = EDITOR_INPUT
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .upgrade();
    let Some(input) = input else { return };
    let mut state = input
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !state.armed || !edits() || Focus::current() != Some(input.focus) {
        return;
    }
    if let Some(point) = point {
        // SAFETY: these calls only compare the hit-tested window to the
        // original, still-focused control.
        let hit = unsafe { WindowFromPoint(point) };
        let control = input.focus.control as HWND;
        if hit != control && unsafe { IsChild(control, hit) } == 0 {
            return;
        }
    }
    state.armed = false;
    state.uncertain = send(input.window, input.id, ACCEPT, "") != Response::Applied;
    state.settle_by = Some(Instant::now() + INPUT_SETTLE);
}

pub(super) enum Commit {
    Inserted,
    Paste,
    Copy,
    Uncertain,
    Reanchor(EditPoint),
}

/// The model's complete hypothesis when the editor accepted the preceding
/// span. The editor owns that span from then on, including manual changes.
#[derive(Clone, Default)]
pub(super) struct EditPoint(Option<String>);

impl EditPoint {
    pub(super) fn suffix<'a>(&self, text: &'a str) -> Option<&'a str> {
        let text = text.trim();
        let Some(prefix) = self.0.as_deref().filter(|prefix| !prefix.is_empty()) else {
            return Some(text);
        };
        if let Some(rest) = text.strip_prefix(prefix) {
            // A streaming decoder can emit more pieces of the last word
            // after the editor accepts it. That word stays at its old caret.
            return Some(if rest.starts_with(char::is_whitespace) {
                rest.trim_start()
            } else {
                rest.find(char::is_whitespace)
                    .map_or("", |index| rest[index..].trim_start())
            });
        }

        // A whole-clip fallback can change casing, punctuation or spacing.
        // Require the same leading words before using any of its suffix.
        let mut rest = text;
        for old in prefix.split_whitespace() {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            let (word, after) = rest.split_at(end);
            if word != old {
                let old = old.trim_matches(|c: char| !c.is_alphanumeric());
                let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                if old.is_empty() || old.to_lowercase() != word.to_lowercase() {
                    return None;
                }
            }
            rest = after.trim_start();
        }
        Some(rest)
    }
}

enum Request {
    Prepare(Instant, SyncSender<EditPoint>),
    Commit(String, Instant, SyncSender<Commit>),
    Cancel,
}

#[derive(Default)]
struct Pending {
    preview: Option<String>,
    request: Option<Request>,
}

#[derive(Default)]
struct Mailbox {
    pending: Mutex<Pending>,
    changed: Condvar,
    first_preview: AtomicBool,
}

impl Mailbox {
    fn request(&self, request: Request) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        pending.preview = None;
        pending.request = Some(request);
        self.changed.notify_one();
    }
}

pub(super) struct LiveTyping {
    mailbox: Arc<Mailbox>,
    ended: bool,
    demand: Option<ClientDemand>,
}

impl LiveTyping {
    pub(super) fn begin(focus: Focus) -> Option<Self> {
        if Focus::current() != Some(focus) {
            return None;
        }
        let window = focus.bridge()?;
        let mut id = [0; 8];
        getrandom::fill(&mut id).ok()?;
        let id = u64::from_le_bytes(id).max(1);
        let mailbox = Arc::new(Mailbox::default());
        let worker = mailbox.clone();
        let input = Arc::new(ManualInput {
            focus,
            window,
            id,
            state: Mutex::new(InputState::default()),
        });
        std::thread::Builder::new()
            .name("hex-inline-dictation".into())
            .spawn(move || run(worker, input))
            .ok()?;
        Some(Self {
            mailbox,
            ended: false,
            demand: Some(client_demand()),
        })
    }

    pub(super) fn preview(&self, partial: &Partial) {
        let words = format!("{} {}", partial.stable, partial.tentative);
        let words = words.trim();
        let mut pending = self
            .mailbox
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if pending.request.is_none() && pending.preview.as_deref() != Some(words) {
            pending.preview = Some(words.to_owned());
            self.mailbox.changed.notify_one();
        }
    }

    pub(super) fn take_first_preview(&self) -> bool {
        self.mailbox.first_preview.swap(false, Ordering::AcqRel)
    }

    pub(super) fn release(&mut self) {
        self.demand.take();
    }

    pub(super) fn prepare(&self) -> Option<EditPoint> {
        let (reply, outcome) = mpsc::sync_channel(1);
        self.mailbox
            .request(Request::Prepare(Instant::now() + REQUEST_TIMEOUT, reply));
        outcome.recv_timeout(REQUEST_TIMEOUT).ok()
    }

    pub(super) fn finish(&mut self, text: &str) -> Commit {
        let (reply, outcome) = mpsc::sync_channel(1);
        self.mailbox.request(Request::Commit(
            text.to_owned(),
            Instant::now() + REQUEST_TIMEOUT,
            reply,
        ));
        let outcome = outcome
            .recv_timeout(REQUEST_TIMEOUT)
            .unwrap_or(Commit::Uncertain);
        self.ended = !matches!(outcome, Commit::Reanchor(_));
        outcome
    }
}

impl Drop for LiveTyping {
    fn drop(&mut self) {
        if !self.ended {
            self.mailbox.request(Request::Cancel);
        }
    }
}

fn run(mailbox: Arc<Mailbox>, input: Arc<ManualInput>) {
    let mut editor = Editor {
        input,
        started: false,
        abandoned: false,
        shown: false,
        last_applied: None,
        point: EditPoint::default(),
    };
    editor.begin();
    if editor.started {
        *EDITOR_INPUT
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Arc::downgrade(&editor.input);
    }
    let mut retry = None;
    loop {
        let (request, preview) = {
            let pending = mailbox
                .pending
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let waiting =
                |pending: &mut Pending| pending.preview.is_none() && pending.request.is_none();
            let mut pending = if retry.is_some() {
                mailbox
                    .changed
                    .wait_timeout_while(pending, SELECTION_RECHECK, waiting)
                    .unwrap_or_else(|error| error.into_inner())
                    .0
            } else {
                mailbox
                    .changed
                    .wait_while(pending, waiting)
                    .unwrap_or_else(|error| error.into_inner())
            };
            (pending.request.take(), pending.preview.take())
        };
        match request {
            Some(Request::Cancel) => {
                editor.cancel();
                break;
            }
            Some(Request::Prepare(deadline, reply)) => {
                retry = None;
                if enough_time(deadline) {
                    editor.prepare();
                    let _ = reply.send(editor.point.clone());
                }
            }
            Some(Request::Commit(text, deadline, reply)) => {
                let outcome = editor.finish(&text, deadline);
                let ended = !matches!(outcome, Commit::Reanchor(_));
                if reply.send(outcome).is_err() {
                    if !ended {
                        editor.cancel();
                    }
                    break;
                }
                if ended {
                    break;
                }
            }
            None => {
                let Some(preview) = preview.or(retry.take()) else {
                    continue;
                };
                if editor.preview(&preview, &mailbox) {
                    retry = Some(preview);
                }
            }
        }
    }
}

struct Editor {
    input: Arc<ManualInput>,
    started: bool,
    abandoned: bool,
    shown: bool,
    last_applied: Option<String>,
    point: EditPoint,
}

impl Editor {
    fn send(&self, operation: usize, text: &str) -> Response {
        self.input.send(operation, text)
    }

    fn begin(&mut self) {
        if Focus::current() != Some(self.input.focus) {
            self.abandoned = true;
            return;
        }
        match self.send(BEGIN, "") {
            Response::Applied => self.started = true,
            Response::Declined => {}
            _ => self.abandoned = self.send(CANCEL, "") == Response::Uncertain,
        }
    }

    fn cancel(&mut self) -> bool {
        let started = std::mem::take(&mut self.started);
        !started || self.send(CANCEL, "") == Response::Applied
    }

    fn focused(&mut self) -> bool {
        if Focus::current() == Some(self.input.focus) {
            return true;
        }
        self.cancel();
        self.abandoned = true;
        false
    }

    fn reanchor(&mut self) {
        if let Some(shown) = self.last_applied.take() {
            self.point.0 = Some(shown);
        } else if self.point.0.is_none() {
            self.point.0 = Some(String::new());
        }
    }

    /// Checks for a manual change after the last preview so Output receives
    /// only the speech that can still be replaced.
    fn prepare(&mut self) {
        if self.started && self.focused() && self.send(PROBE, "") == Response::Reanchor {
            self.reanchor();
        }
    }

    fn finish(&mut self, text: &str, deadline: Instant) -> Commit {
        self.input.wait_for_input(deadline);
        if !enough_time(deadline) {
            self.cancel();
            return Commit::Uncertain;
        }
        if self.abandoned || !self.focused() {
            return Commit::Copy;
        }
        if self.point.0.is_none() && !self.shown {
            // An older bridge or an ineligible textbox may never have owned
            // any text. End its empty session before using ordinary paste.
            if self.cancel() && Focus::current() == Some(self.input.focus) {
                return Commit::Paste;
            }
            return Commit::Copy;
        }
        if !self.started {
            return Commit::Copy;
        }
        match self.send(COMMIT, text) {
            Response::Applied => Commit::Inserted,
            Response::Reanchor => {
                self.reanchor();
                Commit::Reanchor(self.point.clone())
            }
            Response::Declined | Response::SelectionPending => {
                self.cancel();
                Commit::Copy
            }
            Response::Uncertain => {
                self.cancel();
                Commit::Uncertain
            }
        }
    }

    /// Returns true while a noncollapsed selection temporarily blocks these
    /// words. The worker retries only with pending speech.
    fn preview(&mut self, text: &str, mailbox: &Mailbox) -> bool {
        if !self.started || !self.focused() {
            return false;
        }
        for _ in 0..2 {
            let Some(suffix) = self.point.suffix(text) else {
                return false;
            };
            if self.last_applied.as_deref() == Some(text)
                || (self.last_applied.is_none() && suffix.is_empty())
            {
                return false;
            }
            match self.send(UPDATE, suffix) {
                Response::Applied => {
                    if !self.shown && !suffix.is_empty() {
                        mailbox.first_preview.store(true, Ordering::Release);
                        self.shown = true;
                    }
                    self.last_applied = Some(text.to_owned());
                    return false;
                }
                Response::Reanchor => self.reanchor(),
                Response::SelectionPending => return true,
                Response::Declined | Response::Uncertain => {
                    self.cancel();
                    self.abandoned = true;
                    return false;
                }
            }
        }
        true
    }
}

fn enough_time(deadline: Instant) -> bool {
    deadline.saturating_duration_since(Instant::now())
        > Duration::from_millis(u64::from(MESSAGE_MS))
}
