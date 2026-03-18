#ifndef KEYFINDER_WRAPPER_H
#define KEYFINDER_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Returns the key_t enum value (0-24) for the given audio samples.
// samples: interleaved f64 audio data
// sample_count: total number of samples (frames * channels)
// channels: number of audio channels
// frame_rate: sample rate in Hz
// Returns 24 (SILENCE) on error.
int kf_detect_key(const double* samples, unsigned int sample_count,
                  unsigned int channels, unsigned int frame_rate);

#ifdef __cplusplus
}
#endif

#endif
