//! The opt-in Windows ARM64 Whisper worker, using a private stdio pipe.

use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, WrapErr, bail, eyre};
use serde::{Deserialize, Serialize};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject,
};

use crate::transcription::SpeechBackend;
use crate::transcription::gguf_session::{chunk_ranges, job_cancel, sample_offset_ms};
use crate::transcription::models::AUTO_LANGUAGE;
use crate::transcription::transcript::{Transcript, TranscriptSegment};

const WORKER_ENV: &str = "HEX_SNAPDRAGON_WORKER";
const PYTHON_ENV: &str = "HEX_SNAPDRAGON_PYTHON";
const LANGUAGE_ENV: &str = "HEX_SNAPDRAGON_LANGUAGE";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
const CLIP_TIMEOUT: Duration = Duration::from_secs(90);
const MAX_REPLY: u64 = 1024 * 1024;
const MAX_FRAMES: usize = 30 * 16_000;

pub(crate) fn configured() -> bool {
    std::env::var_os(WORKER_ENV).is_some_and(|value| !value.is_empty())
}

pub(crate) fn language() -> String {
    std::env::var(LANGUAGE_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| AUTO_LANGUAGE.into())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct Identity {
    pub(crate) model: String,
    pub(crate) language: String,
}

impl Identity {
    pub(crate) fn name(&self) -> &str {
        match self.model.as_str() {
            "whisper-large-v3-turbo" => "Whisper Large v3 Turbo (Qualcomm QNN)",
            _ => &self.model,
        }
    }
}

pub(crate) struct Qnn {
    config: Config,
    worker: Option<Worker>,
    identity: Identity,
    backend: SpeechBackend,
    detected_language: Option<String>,
}

impl Qnn {
    pub(crate) fn load(language: &str) -> Result<Self> {
        let config = Config::read()?;
        let mut worker = Worker::start(&config)?;
        let ready = worker.initialize()?;
        let language = ready.language(language)?;
        tracing::info!(
            model = ready.model,
            backend = ready.backend,
            device = ready.device,
            %language,
            worker_pid = worker.child.id(),
            "loaded Snapdragon transcription model"
        );
        Ok(Self {
            config,
            worker: Some(worker),
            identity: Identity {
                model: ready.model,
                language,
            },
            backend: SpeechBackend {
                kind: ready.backend,
                device: "Qualcomm Hexagon NPU (HTP)".into(),
                crash_fallback: false,
            },
            detected_language: None,
        })
    }

    pub(crate) fn identity(&self) -> &Identity {
        &self.identity
    }

    pub(crate) fn backend(&self) -> Option<&SpeechBackend> {
        self.worker.as_ref().map(|_| &self.backend)
    }

    pub(crate) fn detected_language(&self) -> Option<&str> {
        self.detected_language.as_deref()
    }

    pub(crate) fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript> {
        self.detected_language = None;
        let mut transcript = Transcript::default();
        let mut texts = Vec::new();
        for range in chunk_ranges(samples, MAX_FRAMES) {
            let offset = sample_offset_ms(range.start);
            let samples = &samples[range];
            let audio = AudioFile::create(samples)?;
            self.ensure_worker()?;
            let language = (self.identity.language != AUTO_LANGUAGE)
                .then_some(self.identity.language.as_str());
            let result = match self
                .worker
                .as_mut()
                .expect("initialized worker")
                .request::<Clip>(
                    Request::Transcribe {
                        audio_path: &audio.0,
                        language,
                    },
                    CLIP_TIMEOUT,
                ) {
                Ok(response) => response,
                Err(error) => {
                    self.worker.take();
                    return Err(error);
                }
            };
            if !result.text.trim().is_empty() {
                if self.detected_language.is_none() {
                    self.detected_language = result.language;
                }
                texts.push(result.text.trim().to_owned());
            }
            let duration = sample_offset_ms(samples.len());
            transcript
                .segments
                .extend(result.segments.into_iter().filter_map(|segment| {
                    (segment.start_ms >= 0
                        && segment.end_ms > segment.start_ms
                        && segment.start_ms < duration
                        && !segment.text.trim().is_empty())
                    .then(|| TranscriptSegment {
                        start_ms: offset + segment.start_ms,
                        end_ms: offset + segment.end_ms.min(duration),
                        text: segment.text,
                    })
                }));
        }
        transcript.text = texts.join(" ");
        Ok(transcript)
    }

    fn ensure_worker(&mut self) -> Result<()> {
        if self.worker.is_none() {
            let mut worker = Worker::start(&self.config)?;
            let ready = worker.initialize()?;
            if ready.model != self.identity.model {
                bail!("the Snapdragon worker loaded a different model; restart Hex to use it");
            }
            ready.language(&self.identity.language)?;
            self.worker = Some(worker);
        }
        Ok(())
    }
}

struct Config {
    python: PathBuf,
    script: PathBuf,
}

impl Config {
    fn read() -> Result<Self> {
        if !cfg!(target_arch = "aarch64") {
            bail!("Snapdragon QNN requires an ARM64 build of Hex");
        }
        let script = std::env::var_os(WORKER_ENV)
            .map(PathBuf::from)
            .ok_or_else(|| eyre!("{WORKER_ENV} must point to the Snapdragon worker script"))?;
        let python = match std::env::var_os(PYTHON_ENV) {
            Some(python) => PathBuf::from(python),
            None => PathBuf::from(
                std::env::var_os("LOCALAPPDATA")
                    .ok_or_else(|| eyre!("LOCALAPPDATA or {PYTHON_ENV} is required"))?,
            )
            .join(r"Hex\snapdragon\venv\Scripts\python.exe"),
        };
        if !python.is_absolute() || !python.is_file() {
            bail!("{PYTHON_ENV} must point to an existing ARM64 Python executable");
        }
        if !script.is_absolute() || !script.is_file() {
            bail!("{WORKER_ENV} must point to an existing worker script by absolute path");
        }
        Ok(Self { python, script })
    }
}

#[derive(Deserialize)]
struct Ready {
    model: String,
    backend: String,
    device: String,
    provider: String,
    cpu_fallback: bool,
    languages: Vec<String>,
}

impl Ready {
    fn language(&self, requested: &str) -> Result<String> {
        let requested = requested.trim().to_ascii_lowercase();
        if requested == AUTO_LANGUAGE {
            return Ok(requested);
        }
        let base = requested.split('-').next().unwrap_or(&requested);
        if let Some(language) = self
            .languages
            .iter()
            .find(|language| **language == requested || **language == base)
        {
            return Ok(language.clone());
        }
        bail!(
            "the Snapdragon model {} doesn't support language {requested:?}",
            self.model
        )
    }
}

#[derive(Deserialize)]
struct Clip {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    segments: Vec<Segment>,
}

#[derive(Deserialize)]
struct Segment {
    start_ms: i64,
    end_ms: i64,
    text: String,
}

#[derive(Serialize)]
#[serde(tag = "method", rename_all = "snake_case")]
enum Request<'a> {
    Initialize,
    Transcribe {
        audio_path: &'a Path,
        language: Option<&'a str>,
    },
}

#[derive(Serialize)]
struct Message<'a> {
    id: u64,
    #[serde(flatten)]
    request: Request<'a>,
}

#[derive(Deserialize)]
struct Reply<T> {
    id: u64,
    result: Option<T>,
    error: Option<String>,
}

struct Worker {
    child: Child,
    _job: OwnedHandle,
    input: Option<ChildStdin>,
    replies: Receiver<io::Result<Vec<u8>>>,
    id: u64,
}

impl Worker {
    fn start(config: &Config) -> Result<Self> {
        let job = kill_on_close_job()?;
        let mut child = crate::desktop::background_process::command(&config.python)
            .arg("-u")
            .arg(&config.script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("PYTHONIOENCODING", "utf-8")
            .spawn()
            .wrap_err("could not launch the Snapdragon transcription worker")?;
        // SAFETY: both handles stay open throughout the assignment.
        if unsafe { AssignProcessToJobObject(job.as_raw_handle(), child.as_raw_handle()) } == 0 {
            let error = io::Error::last_os_error();
            let _ = child.kill();
            let _ = child.wait();
            return Err(error).wrap_err("could not manage the Snapdragon worker process");
        }
        let input = child.stdin.take().expect("worker stdin pipe");
        let mut output = BufReader::new(child.stdout.take().expect("worker stdout pipe"));
        let (send, replies) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("snapdragon-worker-output".into())
            .spawn(move || {
                loop {
                    let mut line = Vec::new();
                    let read = (&mut output).take(MAX_REPLY).read_until(b'\n', &mut line);
                    let reply = read.and_then(|_| {
                        if line.is_empty() {
                            Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "worker exited",
                            ))
                        } else if line.last() != Some(&b'\n') {
                            Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "invalid worker reply",
                            ))
                        } else {
                            Ok(line)
                        }
                    });
                    let done = reply.is_err();
                    if send.send(reply).is_err() || done {
                        break;
                    }
                }
            })?;
        Ok(Self {
            child,
            _job: job,
            input: Some(input),
            replies,
            id: 0,
        })
    }

    fn initialize(&mut self) -> Result<Ready> {
        let ready: Ready = self.request(Request::Initialize, STARTUP_TIMEOUT)?;
        if ready.backend != "qnn"
            || ready.device != "htp"
            || ready.provider != "QNNExecutionProvider"
            || ready.cpu_fallback
            || ready.model.trim().is_empty()
            || ready.languages.is_empty()
        {
            bail!("the Snapdragon worker did not initialize a Qualcomm QNN HTP model");
        }
        Ok(ready)
    }

    fn request<T: serde::de::DeserializeOwned>(
        &mut self,
        request: Request<'_>,
        timeout: Duration,
    ) -> Result<T> {
        self.id += 1;
        let request = Message {
            id: self.id,
            request,
        };
        let input = self.input.as_mut().expect("open worker stdin");
        serde_json::to_writer(&mut *input, &request)?;
        input.write_all(b"\n")?;
        input.flush()?;
        let cancel = job_cancel();
        let deadline = Instant::now() + timeout;
        let line = loop {
            if cancel.as_ref().is_some_and(|cancel| cancel.is_cancelled()) {
                bail!("Snapdragon transcription was cancelled");
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("the Snapdragon transcription worker timed out");
            }
            match self
                .replies
                .recv_timeout(remaining.min(Duration::from_millis(100)))
            {
                Ok(line) => break line.wrap_err("lost the Snapdragon transcription worker")?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    bail!("the Snapdragon transcription worker exited");
                }
            }
        };
        let reply: Reply<T> =
            serde_json::from_slice(&line).wrap_err("invalid Snapdragon worker reply")?;
        if reply.id != self.id {
            bail!("the Snapdragon worker replied to a different request");
        }
        match (reply.result, reply.error) {
            (Some(result), None) => Ok(result),
            (None, Some(error)) => Err(eyre!("Snapdragon transcription failed: {error}")),
            _ => Err(eyre!("invalid Snapdragon worker result")),
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // EOF lets the worker release its sessions and flush optional NPU profiles.
        self.input.take();
        let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
        while let Ok(None) = self.child.try_wait()
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn kill_on_close_job() -> io::Result<OwnedHandle> {
    // SAFETY: null attributes and name create a new anonymous job.
    let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if handle.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the new handle is valid and owned here.
    let job = unsafe { OwnedHandle::from_raw_handle(handle) };
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    // SAFETY: the handle is open and the information class reads this structure.
    if unsafe {
        SetInformationJobObject(
            job.as_raw_handle(),
            JobObjectExtendedLimitInformation,
            (&raw const limits).cast(),
            std::mem::size_of_val(&limits) as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(job)
}

struct AudioFile(PathBuf);

impl AudioFile {
    fn create(samples: &[f32]) -> Result<Self> {
        let mut nonce = [0; 16];
        getrandom::fill(&mut nonce)
            .map_err(|error| eyre!("could not name temporary Snapdragon audio: {error}"))?;
        let path = std::path::absolute(std::env::temp_dir())?.join(format!(
            "hex-snapdragon-{:032x}.wav",
            u128::from_ne_bytes(nonce)
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let audio = Self(path);
        let mut wav = hound::WavWriter::new(
            io::BufWriter::new(file),
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for sample in samples {
            wav.write_sample((sample * 32_768.0).round() as i16)?;
        }
        wav.finalize()?;
        Ok(audio)
    }
}

impl Drop for AudioFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
