use std::path::Path;

use anyhow::{bail, Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// FFI binding to our C++ wrapper
unsafe extern "C" {
    fn kf_detect_key(
        samples: *const f64,
        sample_count: u32,
        channels: u32,
        frame_rate: u32,
    ) -> i32;
}

/// Musical key result from libkeyfinder.
/// Maps to KeyFinder::key_t enum values 0-24.
const KEY_NAMES: &[&str] = &[
    "A",   "Am",   "Bb",  "Bbm",  "B",   "Bm",
    "C",   "Cm",   "Db",  "Dbm",  "D",   "Dm",
    "Eb",  "Ebm",  "E",   "Em",   "F",   "Fm",
    "Gb",  "Gbm",  "G",   "Gm",   "Ab",  "Abm",
    "silence",
];

/// Circle-of-fifths position for each key_t value.
const KEY_COF: &[f64] = &[
    6.0,  // A  major
    1.0,  // A  minor  (relative of C)
    20.0, // Bb major
    15.0, // Bb minor
    10.0, // B  major
    5.0,  // B  minor
    0.0,  // C  major
    19.0, // C  minor
    14.0, // Db major
    13.0, // Db minor
    4.0,  // D  major
    23.0, // D  minor
    18.0, // Eb major
    17.0, // Eb minor
    8.0,  // E  major
    3.0,  // E  minor
    22.0, // F  major
    17.0, // F  minor
    12.0, // Gb major
    7.0,  // Gb minor
    2.0,  // G  major
    21.0, // G  minor
    16.0, // Ab major
    11.0, // Ab minor
    -1.0, // silence
];

// Cap at ~30s of stereo 48kHz to avoid overflowing u32 or feeding
// absurdly large buffers to libkeyfinder's FFT
const MAX_SAMPLE_COUNT: usize = 30 * 48000 * 2; // 2,880,000

#[derive(Debug, Clone)]
pub struct KeyResult {
    pub name: String,
    pub cof_position: f64,
}

/// Detect the musical key of an audio file using libkeyfinder.
/// Decodes via symphonia, then passes samples to libkeyfinder.
/// Analyzes up to `max_seconds` of audio (0 = entire file).
pub fn detect_key(path: &str, max_seconds: f64) -> Result<Option<KeyResult>> {
    let samples = decode_samples(path, max_seconds)?;

    if samples.data.is_empty() {
        return Ok(None);
    }

    classify_key(&samples.data, samples.channels, samples.sample_rate)
}

struct DecodedAudio {
    data: Vec<f64>,
    channels: u32,
    sample_rate: u32,
}

fn decode_samples(path: &str, max_seconds: f64) -> Result<DecodedAudio> {
    let file_path = Path::new(path);
    let file = std::fs::File::open(file_path)
        .with_context(|| format!("Cannot open {path}"))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("Failed to probe {path}"))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("No audio track found")?;

    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u32)
        .unwrap_or(2);
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let track_id = track.id;

    let max_from_seconds = if max_seconds > 0.0 {
        (max_seconds * sample_rate as f64 * channels as f64) as usize
    } else {
        usize::MAX
    };
    let max_samples = max_from_seconds.min(MAX_SAMPLE_COUNT);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create decoder")?;

    let mut all_samples: Vec<f64> = Vec::new();

    loop {
        if all_samples.len() >= max_samples {
            break;
        }

        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                break;
            }
            Err(e) => bail!("Decode error: {e}"),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let num_samples = decoded.capacity() * spec.channels.count();

        let mut sample_buf = SampleBuffer::<f64>::new(decoded.capacity() as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        let samples = sample_buf.samples();
        let remaining = max_samples.saturating_sub(all_samples.len());
        let take = samples.len().min(remaining).min(num_samples);
        all_samples.extend_from_slice(&samples[..take]);
    }

    Ok(DecodedAudio {
        data: all_samples,
        channels,
        sample_rate,
    })
}

fn classify_key(samples: &[f64], channels: u32, sample_rate: u32) -> Result<Option<KeyResult>> {
    if samples.is_empty() {
        return Ok(None);
    }

    // Ensure we don't overflow u32
    let count = samples.len().min(u32::MAX as usize);

    let key_id = unsafe {
        kf_detect_key(
            samples.as_ptr(),
            count as u32,
            channels,
            sample_rate,
        )
    };

    if !(0..24).contains(&key_id) {
        return Ok(None); // SILENCE or error
    }

    let name = KEY_NAMES[key_id as usize].to_string();
    let cof = KEY_COF[key_id as usize];

    Ok(Some(KeyResult {
        name,
        cof_position: cof,
    }))
}
