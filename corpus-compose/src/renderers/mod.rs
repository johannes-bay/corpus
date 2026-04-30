//! Rendering pipelines for each composition strategy.
//!
//! Each renderer implements the `Renderer` trait and produces a concrete
//! output (file on disk + preview bytes for the UI).

pub mod audio;
pub mod document;
pub mod image;
pub mod manifest;
pub mod mixed;
pub mod timeline;
