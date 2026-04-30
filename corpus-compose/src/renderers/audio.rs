//! Audio sequence/layer renderer using symphonia for decoding and hound for WAV writing.
//!
//! Phase 1 capability:
//!   - Decode audio files to PCM via symphonia
//!   - Resample (naive linear) to a common sample rate
//!   - Normalize per-track volume
//!   - Sequence with simple linear crossfades, or layer with per-stem gain/pan
//!   - Output as WAV via hound

use std::fs::File;
use std::path::{Path, PathBuf};

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use tracing::{debug, warn};

use crate::strategy::{AudioMixMode, AudioMixStrategy, CompositionStrategy};
use crate::{
    ComposeError, Composition, CompositionMember, MemberRole, PreviewData, Renderer, Result,
};

/// Audio renderer.
#[derive(Debug, Clone, Default)]
pub struct AudioRenderer {
    pub waveform_samples: usize,
}

impl AudioRenderer {
    pub fn new() -> Self {
        Self { waveform_samples: 256 }
    }

    fn audio_strategy<'a>(&self, comp: &'a Composition) -> Result<&'a AudioMixStrategy> {
        match &comp.strategy {
            CompositionStrategy::AudioMix(m) => Ok(m),
            other => Err(ComposeError::DomainMismatch {
                strategy: other.kind().to_string(),
                domain: "audio".to_string(),
            }),
        }
    }

    fn render_pcm(&self, composition: &Composition) -> Result<RenderedPcm> {
        let strat = self.audio_strategy(composition)?;
        let target_rate = strat.sample_rate.max(8000);

        let mut tracks: Vec<DecodedTrack> = Vec::new();
        for member in &composition.members {
            match decode_to_stereo(&member.artifact.path, target_rate, strat.max_clip_secs) {
                Ok(d) => tracks.push(d),
                Err(e) => {
                    warn!("compose audio: skipping {}: {e}", member.artifact.path);
                }
            }
        }

        if tracks.is_empty() {
            return Err(ComposeError::Empty);
        }

        let pcm = match strat.mode {
            AudioMixMode::Sequence => sequence_tracks(&tracks, strat),
            AudioMixMode::Layer => layer_tracks(&tracks, &composition.members, strat, false),
            AudioMixMode::DuckAndLayer => layer_tracks(&tracks, &composition.members, strat, true),
        };

        Ok(RenderedPcm {
            sample_rate: target_rate,
            samples_lr: pcm,
        })
    }
}

impl Renderer for AudioRenderer {
    type Output = PathBuf;

    fn validate(&self, composition: &Composition) -> Result<()> {
        if composition.members.is_empty() {
            return Err(ComposeError::Empty);
        }
        let _ = self.audio_strategy(composition)?;
        Ok(())
    }

    fn render(&self, composition: &Composition, output_dir: &Path) -> Result<PathBuf> {
        self.validate(composition)?;
        std::fs::create_dir_all(output_dir)?;

        let pcm = self.render_pcm(composition)?;
        let path = output_dir.join(format!("{}.wav", composition.id));
        write_wav(&path, pcm.sample_rate, &pcm.samples_lr)?;
        Ok(path)
    }

    fn preview(&self, composition: &Composition) -> Result<PreviewData> {
        self.validate(composition)?;
        let pcm = self.render_pcm(composition)?;
        let bins = self.waveform_samples.max(32);
        let envelope = downsample_envelope(&pcm.samples_lr, bins);
        Ok(PreviewData::AudioWaveform(envelope))
    }
}

// ---------------------------------------------------------------------------
// PCM container
// ---------------------------------------------------------------------------

struct RenderedPcm {
    sample_rate: u32,
    /// Interleaved L,R,L,R,...
    samples_lr: Vec<f32>,
}

struct DecodedTrack {
    /// Interleaved L,R samples at the renderer target rate.
    samples: Vec<f32>,
    /// Original source path for diagnostic logging.
    #[allow(dead_code)]
    source: String,
}

impl DecodedTrack {
    fn frame_count(&self) -> usize {
        self.samples.len() / 2
    }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

fn decode_to_stereo(path: &str, target_rate: u32, max_secs: f64) -> Result<DecodedTrack> {
    if !Path::new(path).exists() {
        return Err(ComposeError::SourceMissing(path.to_string()));
    }

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| ComposeError::Audio(format!("probe {path}: {e}")))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| ComposeError::Audio(format!("no audio track in {path}")))?;
    let track_id = track.id;
    let src_rate = track.codec_params.sample_rate.unwrap_or(target_rate);
    let src_channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(2)
        .max(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| ComposeError::Audio(format!("decoder {path}: {e}")))?;

    let max_frames_at_src = if max_secs > 0.0 {
        (max_secs * src_rate as f64) as usize
    } else {
        usize::MAX
    };

    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => {
                debug!("packet read error in {path}: {e}");
                break;
            }
        };
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                append_stereo_f32(&audio_buf, src_channels, &mut interleaved);
                let frames = interleaved.len() / 2;
                if frames >= max_frames_at_src {
                    interleaved.truncate(max_frames_at_src * 2);
                    break;
                }
            }
            Err(SymphoniaError::DecodeError(e)) => {
                debug!("decode error in {path}: {e}");
                continue;
            }
            Err(e) => {
                return Err(ComposeError::Audio(format!("decode {path}: {e}")));
            }
        }
    }

    if interleaved.is_empty() {
        return Err(ComposeError::Audio(format!("no samples decoded from {path}")));
    }

    // Resample if needed.
    let resampled = if src_rate != target_rate {
        resample_linear(&interleaved, src_rate, target_rate)
    } else {
        interleaved
    };

    Ok(DecodedTrack {
        samples: resampled,
        source: path.to_string(),
    })
}

/// Convert any AudioBufferRef into stereo interleaved f32 frames and append.
fn append_stereo_f32(buf: &AudioBufferRef<'_>, src_channels: usize, dest: &mut Vec<f32>) {
    use symphonia::core::audio::AudioBufferRef::*;

    macro_rules! handle {
        ($abuf:expr, $convert:expr) => {{
            let frames = $abuf.frames();
            let ch_count = src_channels.max(1);
            for f in 0..frames {
                let l_raw = $abuf.chan(0)[f];
                let r_raw = if ch_count >= 2 {
                    $abuf.chan(1)[f]
                } else {
                    $abuf.chan(0)[f]
                };
                let l = $convert(l_raw);
                let r = $convert(r_raw);
                dest.push(l);
                dest.push(r);
            }
        }};
    }

    match buf {
        F32(b) => handle!(b, |x: f32| x),
        F64(b) => handle!(b, |x: f64| x as f32),
        S8(b) => handle!(b, |x: i8| x as f32 / i8::MAX as f32),
        S16(b) => handle!(b, |x: i16| x as f32 / i16::MAX as f32),
        S24(b) => handle!(b, |x: symphonia::core::sample::i24| {
            x.inner() as f32 / 8_388_607.0
        }),
        S32(b) => handle!(b, |x: i32| x as f32 / i32::MAX as f32),
        U8(b) => handle!(b, |x: u8| (x as f32 - 128.0) / 128.0),
        U16(b) => handle!(b, |x: u16| (x as f32 - 32768.0) / 32768.0),
        U24(b) => handle!(b, |x: symphonia::core::sample::u24| {
            (x.inner() as f32 - 8_388_608.0) / 8_388_608.0
        }),
        U32(b) => handle!(b, |x: u32| {
            (x as f32 - 2_147_483_648.0) / 2_147_483_648.0
        }),
    }
}

/// Naive linear resampler. Adequate for Phase 1 — preserves musical content
/// at the cost of mild aliasing on high-frequency material.
fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.is_empty() {
        return input.to_vec();
    }
    let frames_in = input.len() / 2;
    if frames_in == 0 {
        return Vec::new();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let frames_out = ((frames_in as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(frames_out * 2);

    for i in 0..frames_out {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let i0 = idx.min(frames_in - 1);
        let i1 = (idx + 1).min(frames_in - 1);
        for ch in 0..2 {
            let a = input[i0 * 2 + ch];
            let b = input[i1 * 2 + ch];
            out.push(a + (b - a) * frac);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Sequencing & layering
// ---------------------------------------------------------------------------

fn sequence_tracks(tracks: &[DecodedTrack], strat: &AudioMixStrategy) -> Vec<f32> {
    if tracks.is_empty() {
        return Vec::new();
    }
    let sr = strat.sample_rate;
    let crossfade_frames =
        ((strat.crossfade_ms as f64 / 1000.0) * sr as f64).round() as usize;

    // Normalize each track to peak ~0.95.
    let mut prepared: Vec<Vec<f32>> = tracks.iter().map(|t| normalize(&t.samples, 0.95)).collect();

    let mut out: Vec<f32> = Vec::new();
    out.extend_from_slice(&prepared.remove(0));

    for next in prepared {
        let cf = crossfade_frames.min((out.len() / 2).min(next.len() / 2));
        if cf == 0 {
            out.extend_from_slice(&next);
            continue;
        }

        let out_frames = out.len() / 2;
        let fade_start = out_frames - cf;

        // Crossfade region: linear ramp.
        for f in 0..cf {
            let t = f as f32 / cf as f32;
            for ch in 0..2 {
                let dst_idx = (fade_start + f) * 2 + ch;
                let src = next[f * 2 + ch];
                let dst = out[dst_idx];
                out[dst_idx] = dst * (1.0 - t) + src * t;
            }
        }

        // Append the rest of `next` after the crossfade region.
        if next.len() > cf * 2 {
            out.extend_from_slice(&next[cf * 2..]);
        }
    }

    // Apply a small fade in/out over the whole output.
    apply_endpoints_fade(&mut out, (sr as f64 * 0.5) as usize);
    out
}

fn layer_tracks(
    tracks: &[DecodedTrack],
    members: &[CompositionMember],
    strat: &AudioMixStrategy,
    duck: bool,
) -> Vec<f32> {
    if tracks.is_empty() {
        return Vec::new();
    }
    let max_frames = tracks.iter().map(|t| t.frame_count()).max().unwrap_or(0);
    let mut out: Vec<f32> = vec![0.0; max_frames * 2];

    // Subjects play at full gain; backgrounds duck slightly when ducking.
    for (track, member) in tracks.iter().zip(members.iter()) {
        let stem = strat.stem_roles.get(&member.role).cloned().unwrap_or_default();
        let mut gain = db_to_lin(stem.gain_db);
        if duck && member.role == MemberRole::Background {
            gain *= 0.5;
        }
        if duck && member.role == MemberRole::Subject {
            gain *= 1.2;
        }
        let pan = stem.pan.clamp(-1.0, 1.0) as f32;
        let (gl, gr) = pan_gains(pan);

        let normalized = normalize(&track.samples, 0.9);
        let frames = normalized.len() / 2;
        for f in 0..frames {
            let l = normalized[f * 2] * gain * gl;
            let r = normalized[f * 2 + 1] * gain * gr;
            out[f * 2] += l;
            out[f * 2 + 1] += r;
        }
    }

    // Final normalization to avoid clipping.
    let peak = out.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for s in out.iter_mut() {
            *s *= scale;
        }
    }

    apply_endpoints_fade(&mut out, (strat.sample_rate as f64 * 0.5) as usize);
    out
}

fn db_to_lin(db: f64) -> f32 {
    10f32.powf(db as f32 / 20.0)
}

fn pan_gains(pan: f32) -> (f32, f32) {
    // Equal-power pan.
    let theta = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
    (theta.cos(), theta.sin())
}

fn normalize(samples: &[f32], target_peak: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let peak = samples.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    if peak < 1e-6 {
        return samples.to_vec();
    }
    let scale = target_peak / peak;
    samples.iter().map(|s| s * scale).collect()
}

fn apply_endpoints_fade(samples: &mut [f32], fade_frames: usize) {
    let total_frames = samples.len() / 2;
    if total_frames == 0 {
        return;
    }
    let fade = fade_frames.min(total_frames / 2);
    for f in 0..fade {
        let g = f as f32 / fade as f32;
        samples[f * 2] *= g;
        samples[f * 2 + 1] *= g;
        let from_end = total_frames - 1 - f;
        samples[from_end * 2] *= g;
        samples[from_end * 2 + 1] *= g;
    }
}

fn downsample_envelope(samples: &[f32], bins: usize) -> Vec<f64> {
    if samples.is_empty() || bins == 0 {
        return Vec::new();
    }
    let frames = samples.len() / 2;
    let chunk = (frames / bins).max(1);
    let mut env = Vec::with_capacity(bins);
    for b in 0..bins {
        let start = b * chunk;
        let end = ((b + 1) * chunk).min(frames);
        if start >= end {
            env.push(0.0);
            continue;
        }
        let mut peak = 0.0f32;
        for f in start..end {
            let v = (samples[f * 2].abs() + samples[f * 2 + 1].abs()) * 0.5;
            if v > peak {
                peak = v;
            }
        }
        env.push(peak as f64);
    }
    env
}

// ---------------------------------------------------------------------------
// WAV writing
// ---------------------------------------------------------------------------

fn write_wav(path: &Path, sample_rate: u32, samples_lr: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples_lr {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

