//! The models owned by the Windows inference thread.

use color_eyre::eyre::Result;

use super::{CancelToken, Partial, StreamFeed};
use crate::transcription::SpeechBackend;
use crate::transcription::gguf_session::with_job_cancel;
use crate::transcription::local_transcriber::{LocalTranscriber, TimedSpan};
use crate::transcription::models::{TranscriptionModelId, TranscriptionSelection};
use crate::transcription::snapdragon::{Identity, Qnn};

pub(crate) enum Model {
    Gguf(Box<LocalTranscriber>),
    Snapdragon(Box<Qnn>),
}

impl Model {
    /// Load the explicit GGUF selection, used for existing reloads.
    pub(super) fn load(selection: &TranscriptionSelection) -> Result<Self> {
        LocalTranscriber::load(selection).map(Into::into)
    }

    pub(crate) fn gguf(&mut self) -> Option<&mut LocalTranscriber> {
        match self {
            Self::Gguf(model) => Some(model),
            Self::Snapdragon(_) => None,
        }
    }

    pub(super) fn gguf_selection(&self) -> Option<&TranscriptionSelection> {
        match self {
            Self::Gguf(model) => model.gguf_selection(),
            Self::Snapdragon(_) => None,
        }
    }

    pub(super) fn model_id(&self) -> Option<TranscriptionModelId> {
        self.gguf_selection().map(|selection| selection.model)
    }

    pub(super) fn snapdragon_identity(&self) -> Option<&Identity> {
        match self {
            Self::Gguf(_) => None,
            Self::Snapdragon(model) => Some(model.identity()),
        }
    }

    pub(super) fn detected_language(&self) -> Option<&str> {
        match self {
            Self::Gguf(_) => None,
            Self::Snapdragon(model) => model.detected_language(),
        }
    }

    pub(super) fn backend(&self) -> Option<&SpeechBackend> {
        match self {
            Self::Gguf(model) => Some(model.backend()),
            Self::Snapdragon(model) => model.backend(),
        }
    }

    pub(super) fn out_of_memory(&self) -> Option<&crate::transcription::gpu_health::Failure> {
        match self {
            Self::Gguf(model) => model.out_of_memory(),
            Self::Snapdragon(_) => None,
        }
    }

    pub(super) fn supports_streaming(&self) -> bool {
        match self {
            Self::Gguf(model) => model.stream_options().is_some(),
            Self::Snapdragon(model) => model.supports_streaming(),
        }
    }

    pub(super) fn refresh_stream(&mut self) {
        if let Some(model) = self.gguf() {
            model.refresh_stream();
        }
    }

    pub(super) fn stream_trailing_silence(&self) -> usize {
        match self {
            Self::Gguf(model) => model.stream_trailing_silence(),
            Self::Snapdragon(_) => 0,
        }
    }

    pub(super) fn stream_requires_closing_mark(&self) -> bool {
        match self {
            Self::Gguf(model) => model.stream_requires_closing_mark(),
            Self::Snapdragon(_) => false,
        }
    }

    pub(super) fn streams_buffered_parakeet(&mut self) -> bool {
        self.gguf()
            .is_some_and(LocalTranscriber::streams_buffered_parakeet)
    }

    pub(super) fn with_stream<R>(
        &mut self,
        cancel: &CancelToken,
        drive: impl FnOnce(&mut dyn StreamFeed) -> R,
    ) -> Result<R, String> {
        match self {
            Self::Gguf(model) => {
                let options = model
                    .stream_options()
                    .cloned()
                    .ok_or("the speech model cannot stream on its device")?;
                model
                    .with_stream(&options, cancel, |stream| drive(stream))
                    .map_err(|error| error.to_string())
            }
            Self::Snapdragon(model) => {
                if cancel.is_cancelled() {
                    return Err("Snapdragon streaming was cancelled".into());
                }
                with_job_cancel(cancel, || model.stream_start())
                    .map_err(|error| error.to_string())?;
                let mut stream = SnapdragonStream {
                    model,
                    cancel,
                    tentative: String::new(),
                    open: true,
                };
                Ok(drive(&mut stream))
            }
        }
    }

    pub(super) fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        match self {
            Self::Gguf(model) => model.transcribe(samples),
            Self::Snapdragon(model) => model.transcribe(samples).map(|result| result.text),
        }
    }

    pub(crate) fn transcribe_words(&mut self, samples: &[f32]) -> Result<(String, Vec<TimedSpan>)> {
        match self {
            Self::Gguf(model) => model.transcribe_words(samples),
            Self::Snapdragon(model) => {
                let result = model.transcribe(samples)?;
                let segments = result
                    .segments
                    .into_iter()
                    .map(|segment| (segment.start_ms, segment.end_ms, segment.text))
                    .collect();
                Ok((result.text, segments))
            }
        }
    }
}

/// A stream owns the worker-side capture until it finishes or is dropped.
struct SnapdragonStream<'a> {
    model: &'a mut Qnn,
    cancel: &'a CancelToken,
    tentative: String,
    open: bool,
}

impl StreamFeed for SnapdragonStream<'_> {
    fn feed(&mut self, samples: &[f32]) -> Result<(), String> {
        if !samples.is_empty() {
            self.tentative = with_job_cancel(self.cancel, || self.model.stream_push(samples))
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<String, String> {
        let text = with_job_cancel(self.cancel, || self.model.stream_finish())
            .map_err(|error| error.to_string())?;
        self.open = false;
        let text = text.trim().to_string();
        Ok(crate::transcription::subtitle_credits::strip(&text).map_or(text, |(kept, _)| kept))
    }

    fn partial(&self) -> Partial {
        Partial {
            stable: String::new(),
            tentative: self.tentative.clone(),
        }
    }
}

impl Drop for SnapdragonStream<'_> {
    fn drop(&mut self) {
        if self.open {
            self.model.stream_cancel();
        }
    }
}

impl From<LocalTranscriber> for Model {
    fn from(model: LocalTranscriber) -> Self {
        Self::Gguf(Box::new(model))
    }
}

impl From<Qnn> for Model {
    fn from(model: Qnn) -> Self {
        Self::Snapdragon(Box::new(model))
    }
}
