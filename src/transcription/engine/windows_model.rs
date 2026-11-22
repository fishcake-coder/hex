//! The models owned by the Windows inference thread.

use color_eyre::eyre::Result;

use crate::transcription::SpeechBackend;
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

    pub(super) fn stream_options(&self) -> Option<&transcribe_cpp::StreamOptions> {
        match self {
            Self::Gguf(model) => model.stream_options(),
            Self::Snapdragon(_) => None,
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
        stream_options: &transcribe_cpp::StreamOptions,
        cancel: &transcribe_cpp::CancelToken,
        drive: impl FnOnce(&mut transcribe_cpp::Stream<'_>) -> R,
    ) -> transcribe_cpp::Result<R> {
        match self.gguf() {
            Some(model) => model.with_stream(stream_options, cancel, drive),
            None => Err(transcribe_cpp::Error::NotImplemented(
                "the Snapdragon worker takes finished audio clips".into(),
            )),
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
