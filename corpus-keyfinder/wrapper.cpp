#include "wrapper.h"
#include <keyfinder/keyfinder.h>
#include <keyfinder/audiodata.h>

extern "C" int kf_detect_key(const double* samples, unsigned int sample_count,
                              unsigned int channels, unsigned int frame_rate) {
    try {
        KeyFinder::AudioData audio;
        audio.setChannels(channels);
        audio.setFrameRate(frame_rate);
        audio.addToSampleCount(sample_count);

        for (unsigned int i = 0; i < sample_count; i++) {
            audio.setSample(i, samples[i]);
        }

        KeyFinder::KeyFinder kf;
        return static_cast<int>(kf.keyOfAudio(audio));
    } catch (...) {
        return 24; // SILENCE
    }
}
