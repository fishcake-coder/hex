//! Access to the engine and its held GGUF model for shared inference.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};

use color_eyre::eyre::Result;

use super::{Model, TranscriptionEngine};
use crate::transcription::models::TranscriptionSelection;
use crate::transcription::transcript::{TimestampLevel, Transcript};

static SHARED: Mutex<Weak<TranscriptionEngine>> = Mutex::new(Weak::new());

impl TranscriptionEngine {
    /// Makes this the engine that meeting transcription and the local API share.
    pub fn share(self: &Arc<Self>) {
        *SHARED.lock().unwrap_or_else(|error| error.into_inner()) = Arc::downgrade(self);
    }

    /// The running app's engine, when one is shared.
    pub fn shared() -> Option<Arc<Self>> {
        SHARED
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .upgrade()
    }
}

/// Whether dictation is capturing or transcribing, so host audio and
/// background work give way to it. macOS counts a capture from its press;
/// Linux and Windows count the shared engine's dictation streams and finishes.
pub(crate) fn dictation_busy() -> bool {
    #[cfg(target_os = "macos")]
    let capturing = crate::parakeet::dictation_busy();
    #[cfg(not(target_os = "macos"))]
    let capturing = false;
    capturing
        || TranscriptionEngine::shared()
            .is_some_and(|engine| engine.dictating.load(Ordering::Acquire))
}

#[cfg(target_os = "macos")]
pub(crate) type HeldModel = crate::transcription::Transcriber;
#[cfg(not(target_os = "macos"))]
pub(crate) type HeldModel = crate::transcription::local_transcriber::LocalTranscriber;

/// The engine's model when it is the GGUF `selection`, never loading or
/// switching one.
pub(crate) fn held<'a>(
    model: &'a mut Model,
    selection: &TranscriptionSelection,
) -> Option<&'a mut HeldModel> {
    #[cfg(target_os = "macos")]
    let model = model
        .active()
        .filter(|model| matches!(model, crate::transcription::Transcriber::Gguf(_)))?;
    #[cfg(target_os = "windows")]
    let model = model.gguf()?;
    model.matches_selection(selection).then_some(model)
}

/// Transcribes a complete 16 kHz clip with a held model, padded as a
/// dictation's clip is.
pub(crate) fn transcribe_clip(model: &mut HeldModel, samples: Vec<f32>) -> Result<String> {
    #[cfg(target_os = "macos")]
    return model.transcribe(samples);
    #[cfg(not(target_os = "macos"))]
    model.transcribe_clip(samples)
}

/// Transcribes a complete 16 kHz clip with a held model and its own native
/// timings, never estimated ones.
pub(crate) fn transcribe_clip_timed(
    model: &mut HeldModel,
    samples: Vec<f32>,
) -> Result<Transcript, String> {
    model
        .transcribe_timed(samples)
        .map_err(|error| format!("{error:#}"))
}

/// The finest native timings a held model gives. Dictation never asks; the
/// local API answers `422` for a level the model cannot produce.
pub(crate) fn timestamp_level(model: &HeldModel) -> TimestampLevel {
    model.timestamps()
}
