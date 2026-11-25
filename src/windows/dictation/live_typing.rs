//! Delivers the latest provisional transcript to the focused Windows text service.

use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::System::DataExchange::COPYDATASTRUCT;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    FindWindowExW, GUITHREADINFO, GetForegroundWindow, GetGUIThreadInfo, GetWindowThreadProcessId,
    HWND_MESSAGE, SMTO_ABORTIFHUNG, SMTO_BLOCK, SMTO_ERRORONEXIT, SendMessageTimeoutW, WM_COPYDATA,
};

use crate::dictation::live_text::{ClientDemand, client_demand};
use crate::transcription::engine::Partial;

const CLASS: windows::core::PCWSTR = windows::core::w!("HexLiveTyping.Tsf");
const MAX_UNITS: usize = 8_192;
const MESSAGE_MS: u32 = 65;
const BEGIN: usize = 1;
const UPDATE: usize = 2;
const COMMIT: usize = 3;
const CANCEL: usize = 4;

#[derive(Clone, Copy, Eq, PartialEq)]
struct Focus {
    window: isize,
    control: isize,
    thread: u32,
}

impl Focus {
    fn current() -> Option<Self> {
        let mut info = GUITHREADINFO {
            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        // SAFETY: these calls only read the foreground GUI thread and its focus.
        let window = unsafe { GetForegroundWindow() };
        if window.is_null()
            || unsafe { GetGUIThreadInfo(0, &mut info) } == 0
            || info.hwndFocus.is_null()
        {
            return None;
        }
        let thread = unsafe { GetWindowThreadProcessId(info.hwndFocus, std::ptr::null_mut()) };
        (thread != 0 && unsafe { GetForegroundWindow() } == window).then_some(Self {
            window: window as isize,
            control: info.hwndFocus as isize,
            thread,
        })
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
            if unsafe { GetWindowThreadProcessId(after, std::ptr::null_mut()) } == self.thread {
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
        _ => Response::Uncertain,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Commit {
    Inserted,
    Paste,
    Copy,
    Uncertain,
}

enum End {
    Commit(String, Instant, SyncSender<Commit>),
    Cancel,
}

#[derive(Default)]
struct Pending {
    preview: Option<String>,
    end: Option<End>,
}

#[derive(Default)]
struct Mailbox {
    pending: Mutex<Pending>,
    changed: Condvar,
}

impl Mailbox {
    fn end(&self, end: End) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        pending.preview = None;
        pending.end = Some(end);
        self.changed.notify_one();
    }
}

pub(super) struct LiveTyping {
    mailbox: Arc<Mailbox>,
    ended: bool,
    demand: Option<ClientDemand>,
}

impl LiveTyping {
    pub(super) fn begin() -> Option<Self> {
        let focus = Focus::current()?;
        let window = focus.bridge()?;
        let mut id = [0; 8];
        getrandom::fill(&mut id).ok()?;
        let id = u64::from_le_bytes(id);
        let mailbox = Arc::new(Mailbox::default());
        let worker = mailbox.clone();
        std::thread::Builder::new()
            .name("hex-inline-dictation".into())
            .spawn(move || run(worker, focus, window, id))
            .ok()?;
        Some(Self {
            mailbox,
            ended: false,
            demand: Some(client_demand()),
        })
    }

    pub(super) fn preview(&self, partial: &Partial) {
        let mut pending = self
            .mailbox
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if pending.end.is_none() {
            pending.preview = Some(
                format!("{} {}", partial.stable, partial.tentative)
                    .trim()
                    .to_owned(),
            );
            self.mailbox.changed.notify_one();
        }
    }

    pub(super) fn release(&mut self) {
        self.demand.take();
    }

    pub(super) fn finish(mut self, text: &str) -> Commit {
        let (reply, outcome) = mpsc::sync_channel(1);
        let timeout = Duration::from_millis(240);
        self.mailbox.end(End::Commit(
            text.to_owned(),
            Instant::now() + timeout,
            reply,
        ));
        self.ended = true;
        outcome.recv_timeout(timeout).unwrap_or(Commit::Uncertain)
    }
}

impl Drop for LiveTyping {
    fn drop(&mut self) {
        if !self.ended {
            self.mailbox.end(End::Cancel);
        }
    }
}

fn run(mailbox: Arc<Mailbox>, focus: Focus, window: isize, id: u64) {
    let mut started = false;
    if Focus::current() == Some(focus) {
        match send(window, id, BEGIN, "") {
            Response::Applied => started = true,
            Response::Uncertain => {
                send(window, id, CANCEL, "");
            }
            Response::Declined => {}
        }
    }
    let mut abandoned = false;
    loop {
        let (end, preview) = {
            let pending = mailbox
                .pending
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let mut pending = mailbox
                .changed
                .wait_while(pending, |pending| {
                    pending.preview.is_none() && pending.end.is_none()
                })
                .unwrap_or_else(|error| error.into_inner());
            (pending.end.take(), pending.preview.take())
        };
        match end {
            Some(End::Cancel) => {
                if started {
                    send(window, id, CANCEL, "");
                }
                break;
            }
            Some(End::Commit(text, deadline, reply)) => {
                let outcome = if Instant::now() >= deadline {
                    if started {
                        send(window, id, CANCEL, "");
                    }
                    Commit::Uncertain
                } else if abandoned {
                    Commit::Copy
                } else if !started {
                    Commit::Paste
                } else if Focus::current() != Some(focus) {
                    send(window, id, CANCEL, "");
                    Commit::Copy
                } else {
                    match send(window, id, COMMIT, &text) {
                        Response::Applied => Commit::Inserted,
                        Response::Declined => {
                            send(window, id, CANCEL, "");
                            Commit::Copy
                        }
                        Response::Uncertain => Commit::Uncertain,
                    }
                };
                let _ = reply.send(outcome);
                break;
            }
            None if started => {
                if Focus::current() != Some(focus) {
                    send(window, id, CANCEL, "");
                    started = false;
                    abandoned = true;
                } else if let Some(preview) = preview {
                    let _ = send(window, id, UPDATE, &preview);
                }
            }
            None => {}
        }
    }
}
