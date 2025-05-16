import numpy as np
import scipy
import librosa
import pyaudio
import soundfile as sf
import warnings
import argparse
import os
import datetime
from concurrent.futures import ThreadPoolExecutor
import queue
import sys
import tqdm
import traceback

# Constants
FFTSIZE = 2048
HOPSIZE = 1024

CHUNK_SIZE = 4096  # Increased for better frequency resolution
SAMPLE_RATE = 44100
HOPSIZE = CHUNK_SIZE // 4

warnings.simplefilter('ignore')

def griffin_lim(mag_spec, n_iter=20, n_fft=1025, hop_length=HOPSIZE):
    phase_spec = np.exp(1j * np.random.uniform(0, 2*np.pi, mag_spec.shape))
    for _ in range(n_iter):
        wav = librosa.istft(mag_spec * phase_spec, n_fft=n_fft, hop_length=hop_length)
        _, phase_spec = librosa.magphase(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
    return phase_spec

def connect_spectra_smooth(spectrum1, spectrum2, overlap_size=32):
    if overlap_size > min(len(spectrum1), len(spectrum2)):
        raise ValueError("too big overlap_size")

    overlap1 = spectrum1[-overlap_size:]
    overlap2 = spectrum2[:overlap_size]
    level_diff = np.mean(1 + overlap1) / np.mean(1 + overlap2)
    adjusted_spectrum2 = spectrum2 * (level_diff + 4)

    fade_out = np.linspace(1, 0, overlap_size)
    fade_in = np.linspace(0, 1, overlap_size)
    crossfaded = overlap1 * fade_out + adjusted_spectrum2[:overlap_size] * fade_in

    result = np.concatenate([
        spectrum1[:-overlap_size//2],
        crossfaded,
        adjusted_spectrum2[overlap_size//2:]
    ])

    return result

def flatten_spectrum(signal, window_size=6):
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    smoothed_signal = np.convolve(padded_signal, np.ones(window_size)/window_size, mode='valid')
    return smoothed_signal[:len(signal)]

def griffin_lim_rt(mag_spec, n_iter=1, n_fft=FFTSIZE, hop_length=HOPSIZE):
    phase_spec = np.exp(1j * np.random.uniform(0, 2*np.pi, mag_spec.shape))
    for _ in range(n_iter):
        wav = librosa.istft(mag_spec * phase_spec, n_fft=n_fft, hop_length=hop_length)
        _, phase_spec = librosa.magphase(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
    return phase_spec

def xorshift(y=2463534242):
    y ^= (y << 13) & 0xFFFFFFFF
    y ^= (y >> 17) & 0xFFFFFFFF
    y ^= (y << 5) & 0xFFFFFFFF
    return y & 0xFFFFFFFF

def remaster(dat, fs, scale):
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2
    mid_ffted = librosa.stft(mid, n_fft=FFTSIZE, hop_length=HOPSIZE)
    side_ffted = librosa.stft(side, n_fft=FFTSIZE, hop_length=HOPSIZE)
    mid_proc = np.zeros([mid_ffted.shape[0]*scale, mid_ffted.shape[1]], dtype=np.complex128)
    side_proc = np.zeros([mid_ffted.shape[0]*scale, mid_ffted.shape[1]], dtype=np.complex128)

    for i in tqdm.tqdm(range(mid_ffted.shape[1])):
        sample_mid = np.hstack([mid_ffted[:,i], np.zeros(len(mid_ffted[:,i])*(scale-1))])
        sample_side = np.hstack([side_ffted[:,i], np.zeros(len(side_ffted[:,i])*(scale-1))])
        db_mid = librosa.amplitude_to_db(sample_mid)
        db_side = librosa.amplitude_to_db(sample_side)

        # Vectorized operations
        mask = db_mid > db_side
        db_side[mask] += (db_mid[mask] - db_side[mask]) / 2
        mask = (np.arange(len(db_side)) > 32) & (db_side < -6)
        db_side[mask] += (db_mid[mask] - db_side[mask]) / 2

        mask = db_side > db_mid
        db_mid[mask] += (db_side[mask] - db_mid[mask]) / 2
        mask = (np.arange(len(db_mid)) > 32) & (db_mid < -6)
        db_mid[mask] += (db_side[mask] - db_mid[mask])

        # Smoothing
        mask = db_mid < -6
        db_mid[mask] = (np.roll(db_mid, -1)[mask] + np.roll(db_mid, 1)[mask]) / 2
        mask = db_side < -6
        db_side[mask] = (np.roll(db_side, -1)[mask] + np.roll(db_side, 1)[mask]) / 2

        db_mid[0] = db_side[0] = -12
        db_mid[256:] += np.linspace(0, int((db_mid[1]-db_mid[256])/16), len(db_mid)-256)
        db_side[256:] += np.linspace(0, int((db_mid[1]-db_mid[256])/16), len(db_side)-256)

        mid_proc[:,i] = librosa.db_to_amplitude(db_mid) * np.exp(1.j * np.angle(sample_mid))
        side_proc[:,i] = librosa.db_to_amplitude(db_side) * np.exp(1.j * np.angle(sample_side))

    mid = librosa.istft(mid_proc, n_fft=FFTSIZE*scale, hop_length=HOPSIZE*scale)
    side = librosa.istft(side_proc, n_fft=FFTSIZE*scale, hop_length=HOPSIZE*scale)
    return np.array([mid + side, mid - side]).T * scale

def proc_for_compressed(mid, side):
    mid_db = librosa.amplitude_to_db(mid)
    side_db = librosa.amplitude_to_db(side)
    mid, mid_phs = librosa.magphase(mid)
    side, side_phs = librosa.magphase(side)

    for i in tqdm.tqdm(range(mid.shape[1])):
        if np.mean(np.abs(mid_db[:,i])) > 60 or np.mean(np.abs(side_db[:,i])) > 60:
            continue

        mask_mid = (mid_db[:,i] < -12) & (np.min(mid_db[:,i]) > -60)
        mid[:,i][mask_mid] += (np.roll(mid[:,i], -1)[mask_mid] + np.roll(mid[:,i], 1)[mask_mid]) / 8

        mask_side = (side_db[:,i] < -12) & (np.min(side_db[:,i]) > -60)
        side[:,i][mask_side] += (np.roll(side[:,i], -1)[mask_side] + np.roll(side[:,i], 1)[mask_side]) / 8

        mask_side_high = mask_side & (np.arange(len(side[:,i])) > 256)
        side[:,i][mask_side_high] += (mid[:,i][mask_side_high] / 16)

    return mid * np.exp(1.j * np.angle(mid_phs)), side * np.exp(1.j * np.angle(side_phs))

def look_for_audio_input():
    pa = pyaudio.PyAudio()
    devices = [pa.get_device_info_by_index(i)["name"] for i in range(pa.get_device_count())]
    pa.terminate()
    return devices

def hires_playback(dev, dev2):
    fs = 48000
    devs = look_for_audio_input()
    p = pyaudio.PyAudio()
    p2 = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=fs, input=True,
                    input_device_index=devs.index(dev), output_device_index=devs.index(dev2),
                    frames_per_buffer=8192)
    playstream = p2.open(format=pyaudio.paFloat32, channels=2, rate=fs, output=True,
                         output_device_index=devs.index(dev2), frames_per_buffer=8192)
    while True:
        d = stream.read(4096*2)
        data = np.frombuffer(d, dtype=np.float32)
        print(data)
        hf = hfp_2(data.copy(), fs)
        playstream.write(hf.astype(np.float32).tobytes())

class OVERTONE:
    def __init__(self):
        self.width = 2
        self.amplitude = 0
        self.base_freq = 0
        self.slope = []
        self.loop = 0

def hfp(dat, lowpass, fs, compressd_mode=False, use_hpss=True):
    fft_size = FFTSIZE * (1 if not compressd_mode else 2)
    hop_length = HOPSIZE
    lowpass_fft = int((fft_size//2+1) * (lowpass/(fs/2)))
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2
    mid_ffted = librosa.stft(mid, n_fft=fft_size, hop_length=hop_length)
    side_ffted = librosa.stft(side, n_fft=fft_size, hop_length=hop_length)
    mid_mag, mid_phs = librosa.magphase(mid_ffted)
    side_mag, side_phs = librosa.magphase(side_ffted)

    if use_hpss:
        mid_hm, mid_pc = librosa.decompose.hpss(mid_mag, kernel_size=3)
        side_hm, side_pc = librosa.decompose.hpss(side_mag, kernel_size=3)
    else:
        mid_pc, mid_hm = mid_mag, np.zeros_like(mid_mag)
        side_pc, side_hm = side_mag, np.zeros_like(side_mag)

    scale = int(fs / 48000 + 0.555555)

    for i in tqdm.tqdm(range(mid_hm.shape[1])):
        mid_ffted[:,i][lowpass_fft:] = 0
        side_ffted[:,i][lowpass_fft:] = 0

        sample_mid_hm = mid_hm[:,i].copy()
        sample_mid_pc = mid_pc[:,i].copy()
        sample_side_hm = side_hm[:,i].copy()
        sample_side_pc = side_pc[:,i].copy()

        sample_mid_hm[lowpass_fft:] = 0
        sample_mid_pc[lowpass_fft:] = 0
        sample_side_hm[lowpass_fft:] = 0
        sample_side_pc[lowpass_fft:] = 0
        rebuild_mid = np.zeros_like(sample_mid_hm)
        rebuild_noise_mid = np.zeros_like(sample_mid_pc)
        rebuild_side = np.zeros_like(sample_mid_hm)
        rebuild_noise_side = np.zeros_like(sample_mid_pc)

        db_hm_max = scipy.signal.find_peaks(librosa.amplitude_to_db(sample_mid_hm))[0]
        db_hm_max = db_hm_max[db_hm_max > lowpass_fft//2]
        db_pc_max = scipy.signal.find_peaks(librosa.amplitude_to_db(sample_mid_pc))[0]
        db_pc_max = db_pc_max[db_pc_max > lowpass_fft//2]
        db_side_hm_max = scipy.signal.find_peaks(librosa.amplitude_to_db(sample_side_hm))[0]
        db_side_hm_max = db_side_hm_max[db_side_hm_max > lowpass_fft//2]
        db_side_pc_max = scipy.signal.find_peaks(librosa.amplitude_to_db(sample_side_pc))[0]
        db_side_pc_max = db_side_pc_max[db_side_pc_max > lowpass_fft//2]

        # Remove harmonics
        db_hm_max = np.array([f for f in db_hm_max if not any(np.abs(f - k*f) < 1 for k in range(2, fft_size//(2*f)))])
        db_side_hm_max = np.array([f for f in db_side_hm_max if not any(np.abs(f - k*f) < 1 for k in range(2, fft_size//(2*f)))])

        for peaks, sample, rebuild in [(db_hm_max, sample_mid_hm, rebuild_mid),
                                       (db_pc_max, sample_mid_pc, rebuild_noise_mid),
                                       (db_side_hm_max, sample_side_hm, rebuild_side),
                                       (db_side_pc_max, sample_side_pc, rebuild_noise_side)]:
            for j in peaks:
                ot = OVERTONE()
                ot.base_freq = j // 2 if 'hm' in locals() else j
                ot.loop = fft_size // (2 * ot.base_freq)

                harmonics = np.array([sample[ot.base_freq * l] for l in range(ot.loop) if ot.base_freq * l < len(sample)])
                ot.slope = np.fft.fft(np.fft.ifft(harmonics / harmonics[0]) + np.linspace(1, 0, len(harmonics))) / 2

                for k in range(2, 3):
                    if j-k//2 >= 0 and j+k//2 < len(sample) and abs(abs(sample[j-k//2]) - abs(sample[j+k//2])) < 4:
                        ot.width = k
                        break

                ot.power = sample[j-ot.width//2:j+ot.width//2]

                for k in range(1, ot.loop):
                    start = int(ot.base_freq * k - ot.width//2)
                    end = int(ot.base_freq * k + ot.width//2)
                    if start < 0 or end > len(rebuild):
                        break
                    rebuild[start:end] += ot.power * abs(ot.slope[k])

        # Applying the rebuilt spectra
        mid_hm[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_mid[lowpass_fft:] * 2 * np.linspace(1, 3, len(rebuild_mid[lowpass_fft:]))])
        mid_pc[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_mid[lowpass_fft:] / 2 * np.linspace(1, 3, len(rebuild_mid[lowpass_fft:]))])
        side_hm[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_side[lowpass_fft:] * 2 * np.linspace(1, 3, len(rebuild_mid[lowpass_fft:]))])
        side_pc[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_side[lowpass_fft:] / 2 * np.linspace(1, 3, len(rebuild_mid[lowpass_fft:]))])

        # Smoothing
        for spec in [mid_hm[:,i], mid_pc[:,i], side_hm[:,i], side_pc[:,i]]:
            spec[lowpass_fft+1:-1] += (spec[lowpass_fft:-2] + spec[lowpass_fft+2:]) / 6

        mid_hm[:,i] = flatten_spectrum(mid_hm[:,i], window_size=5)
        mid_pc[:,i] = flatten_spectrum(mid_pc[:,i], window_size=12)
        side_hm[:,i] = flatten_spectrum(side_hm[:,i], window_size=5)
        side_pc[:,i] = flatten_spectrum(side_pc[:,i], window_size=12)

        # Mirror lower frequencies
        mid_hm[:,i][lowpass_fft:lowpass_fft*2] = mid_hm[:,i][lowpass_fft:lowpass_fft*2][::-1]
        side_hm[:,i][lowpass_fft:lowpass_fft*2] = side_hm[:,i][lowpass_fft:lowpass_fft*2][::-1]
        mid_pc[:,i][lowpass_fft:lowpass_fft*2] = mid_pc[:,i][lowpass_fft:lowpass_fft*2][::-1]
        side_pc[:,i][lowpass_fft:lowpass_fft*2] = side_pc[:,i][lowpass_fft:lowpass_fft*2][::-1]

        # Scaling
        mid_hm[:,i] /= scale * 4
        mid_pc[:,i] /= scale * 4
        side_hm[:,i] /= scale * 4
        side_pc[:,i] /= scale * 4


        # Apply fade out to high frequencies
        fade_out = np.linspace(1, 0, len(mid_hm[:,i][lowpass_fft:])) ** 2
        for spec in [mid_hm[:,i], mid_pc[:,i], side_hm[:,i], side_pc[:,i]]:
            spec[lowpass_fft:] *= fade_out

    # Phase reconstruction
    mid_phs[lowpass_fft:] = griffin_lim(mid_hm + mid_pc, n_fft=fft_size, hop_length=hop_length)[lowpass_fft:]
    side_phs[lowpass_fft:] = griffin_lim(side_hm + side_pc, n_fft=fft_size, hop_length=hop_length)[lowpass_fft:]

    rebuilt_mid = (mid_hm + mid_pc) * np.exp(1.j * mid_phs)
    rebuilt_side = (side_hm + side_pc) * np.exp(1.j * side_phs)

    if compressd_mode:
        for i in range(mid_ffted.shape[1]):
            mid_ffted[:,i] = connect_spectra_smooth(mid_ffted[:,i][:lowpass_fft], rebuilt_mid[:,i][lowpass_fft:] * 2)
            side_ffted[:,i] = connect_spectra_smooth(side_ffted[:,i][:lowpass_fft], rebuilt_side[:,i][lowpass_fft:] * 2)
        mid_ffted, side_ffted = proc_for_compressed(mid_ffted, side_ffted)
    else:
        for i in range(mid_ffted.shape[1]):
            mid_ffted[:,i] = connect_spectra_smooth(mid_ffted[:,i][:lowpass_fft], rebuilt_mid[:,i][lowpass_fft:])
            side_ffted[:,i] = connect_spectra_smooth(side_ffted[:,i][:lowpass_fft], rebuilt_side[:,i][lowpass_fft:])

    iffted_mid = librosa.istft(mid_ffted, hop_length=hop_length)
    iffted_side = librosa.istft(side_ffted, hop_length=hop_length)

    return np.array([iffted_mid+iffted_side, iffted_mid-iffted_side]).T


class AudioProcessor:
    def __init__(self, input_device, output_device, settings):
        self.input_device = input_device
        self.output_device = output_device
        self.settings = settings
        self.running = False
        self.audio_queue = queue.Queue(maxsize=10)

    def process(self, file):
        dat, fs = sf.read(file)

        if self.settings['noise_reduction']:
            dat = decode(dat, 120)
        if self.settings['remaster']:
            dat = remaster(dat, fs, self.settings['scale'])
        elif self.settings['scale'] != 1:
            dat = remaster(dat, fs, self.settings['scale'])
        if self.settings['high_freq_comp']:
            dat = hfp(dat, self.settings['lowpass'], fs * self.settings['scale'],
                      compressd_mode=self.settings['compressed_mode'])
        output_file = f"{os.path.splitext(file)[0]}_converted.wav"
        sf.write(output_file, dat, fs * self.settings['scale'],
                 format="WAV", subtype=f"PCM_{self.settings['bit_depth']}")
        return output_file

def hfp_2(dat, fs):
    fft_size = FFTSIZE
    dat = np.nan_to_num(dat)
    mid = (dat[::2] + dat[1::2]) / 2
    side = (dat[::2] - dat[1::2]) / 2
    lowpass_fft = 384
    mid_ffted = librosa.stft(mid, n_fft=fft_size, hop_length=fft_size//2)
    side_ffted = librosa.stft(side, n_fft=fft_size, hop_length=fft_size//2)
    mid_mag, mid_phs = librosa.magphase(mid_ffted)
    side_mag, side_phs = librosa.magphase(side_ffted)
    mid_hm, mid_pc = librosa.decompose.hpss(mid_mag, kernel_size=31, mask=True)
    side_hm, side_pc = librosa.decompose.hpss(side_mag, kernel_size=31, mask=True)
    scale = int(fs / 48000 + 0.555555)

    for i in tqdm.tqdm(range(mid_hm.shape[1])):
        sample_mid_hm = mid_hm[:,i].copy()
        sample_mid_pc = mid_pc[:,i].copy()
        sample_side_hm = side_hm[:,i].copy()
        sample_side_pc = side_pc[:,i].copy()

        sample_mid_hm[lowpass_fft:] = 0
        sample_mid_pc[lowpass_fft:] = 0
        sample_side_hm[lowpass_fft:] = 0
        sample_side_pc[lowpass_fft:] = 0

        rebuild_mid = np.zeros_like(sample_mid_hm)
        rebuild_noise_mid = np.zeros_like(sample_mid_pc)
        rebuild_side = np.zeros_like(sample_mid_hm)
        rebuild_noise_side = np.zeros_like(sample_mid_pc)

        for sample, rebuild in [(sample_mid_hm, rebuild_mid),
                                (sample_mid_pc, rebuild_noise_mid),
                                (sample_side_hm, rebuild_side),
                                (sample_side_pc, rebuild_noise_side)]:
            peaks = scipy.signal.find_peaks(sample)[0]
            peaks = peaks[peaks > len(peaks)//2]

            for j in peaks:
                ot = OVERTONE()
                ot.base_freq = j // 2
                ot.loop = fft_size // (2 * ot.base_freq)

                harmonics = np.array([sample[ot.base_freq * l] for l in range(ot.loop) if ot.base_freq * l < len(sample)])
                ot.slope = np.fft.fft(np.fft.ifft(harmonics / harmonics[0]) + np.linspace(1, 0.33333, len(harmonics))) / (22 if 'hm' in locals() else 8)

                for k in range(2, 4, 2):
                    if j-k//2 >= 0 and j+k//2 < len(sample) and abs(abs(sample[j-k//2]) - abs(sample[j+k//2])) < 4:
                        ot.width = k
                        break

                ot.power = sample[j-ot.width//2:j+ot.width//2]

                for k in range(1, ot.loop):
                    start = int(ot.base_freq * (k//2 + 1) - ot.width//2)
                    end = int(ot.base_freq * (k//2 + 1) + ot.width//2)
                    if start < 0 or end > len(rebuild):
                        break
                    rebuild[start:end] = ot.power * abs(ot.slope[k//2])

        # Applying the rebuilt spectra
        mid_hm[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_mid[lowpass_fft:]])
        mid_pc[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_mid[lowpass_fft:]])
        side_hm[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_side[lowpass_fft:]])
        side_pc[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_side[lowpass_fft:]])

        fade_out = np.linspace(1, 0, len(mid_hm[:,i][lowpass_fft:]))
        mid_hm[:,i][lowpass_fft:] *= fade_out
        mid_pc[:,i][lowpass_fft:] *= fade_out
        side_hm[:,i][lowpass_fft:] *= fade_out
        side_pc[:,i][lowpass_fft:] *= fade_out

        # Level matching
        for original, rebuilt in [(sample_mid_hm, rebuild_mid),
                                  (sample_mid_pc, rebuild_noise_mid),
                                  (sample_side_hm, rebuild_side),
                                  (sample_side_pc, rebuild_noise_side)]:
            db_original = librosa.amplitude_to_db(abs(original[lowpass_fft-60:lowpass_fft]))
            db_rebuilt = librosa.amplitude_to_db(abs(rebuilt[lowpass_fft:lowpass_fft+60]))
            diff_db = np.mean(db_original) - np.mean(db_rebuilt)
            diff_db = min(diff_db, 36)
            rebuilt = librosa.db_to_amplitude(librosa.amplitude_to_db(rebuilt) + diff_db - 15 - scale*2)

        # Envelope shaping
        for spec in [mid_hm[:,i], mid_pc[:,i], side_hm[:,i], side_pc[:,i]]:
            env = np.abs(np.fft.fft(np.fft.ifft(spec) + np.linspace(0.5, -0.5, len(spec)))) / 12
            env = np.clip(env, 0, 1)
            spec *= env

    # Phase reconstruction
    mid_phs[lowpass_fft:] = griffin_lim_rt(mid_ffted)[lowpass_fft:]
    side_phs[lowpass_fft:] = griffin_lim_rt(side_ffted)[lowpass_fft:]

    rebuilt_mid = (mid_hm + mid_pc) * np.exp(1.j * mid_phs)
    rebuilt_side = (side_hm + side_pc) * np.exp(1.j * side_phs)

    # Replace very low amplitude parts
    low_amp_mask = librosa.amplitude_to_db(mid_ffted) < -80
    mid_ffted[low_amp_mask] = rebuilt_mid[low_amp_mask]
    side_ffted[low_amp_mask] = rebuilt_side[low_amp_mask]

    mid_ffted[:lowpass_fft] = 0
    side_ffted[:lowpass_fft] = 0

    iffted_mid = librosa.istft(mid_ffted, hop_length=fft_size//2)
    iffted_side = librosa.istft(side_ffted, hop_length=fft_size//2)

    return np.array([iffted_mid+iffted_side, iffted_mid-iffted_side]).T

def get_score(original_file, processed_file, sr, lowpass):
    original_y = (original_file[:,0] + original_file[:,1]) / 2
    processed_y = (processed_file[:,0] + processed_file[:,1]) / 2
    lowpass_fft = int((1024//2+1) * (lowpass/(sr/2)))

    # Ensure same length
    min_length = min(len(original_y), len(processed_y))
    original_y = original_y[:min_length]
    processed_y = processed_y[:min_length]

    original_stft = librosa.stft(original_y)
    processed_stft = librosa.stft(processed_y)

    score = 1
    for i in range(original_stft.shape[1]):
        or_ = original_stft[:,i]
        cv = processed_stft[:,i]
        score = (score + np.corrcoef(or_.real, cv.real)[0][1]) / 2

    return score

def decode(dat, bit):
    bit *= -1
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2

    for channel in [mid, side]:
        ffted = librosa.stft(channel, n_fft=1024)
        for i in tqdm.tqdm(range(ffted.shape[1])):
            mag, phs = librosa.magphase(ffted[:,i])
            db = librosa.amplitude_to_db(mag)
            avg = int(np.mean(db)) - 6

            mask = db < bit
            db[mask] = -120

            smooth_mask = (db == -120) & (np.roll(db, -1) != -120) & (np.roll(db, 1) != -120)
            db[smooth_mask] = (np.roll(db, -1)[smooth_mask] + np.roll(db, 1)[smooth_mask]) / 2

            random_mask = db == -120
            db[random_mask] = np.random.randint(avg-15, avg-8, size=np.sum(random_mask))

            idb = librosa.db_to_amplitude(db)
            idb[-400:] *= np.linspace(1, 0, 400)
            ffted[:,i] = idb * np.exp(1.j * np.angle(phs))

        channel[:] = librosa.istft(ffted, n_fft=1024)

    return np.array([mid+side, mid-side]).T

def prompt_with_default(prompt, default, cast_func=str, choices=None):
    while True:
        if choices:
            prompt_str = f"{prompt} [{'/'.join(str(c) for c in choices)}] (default: {default}): "
        else:
            prompt_str = f"{prompt} (default: {default}): "
        val = input(prompt_str).strip()
        if not val:
            return default
        try:
            val_cast = cast_func(val)
            if choices and val_cast not in choices:
                print(f"Please choose from {choices}.")
                continue
            return val_cast
        except Exception:
            print(f"Invalid input. Please enter a value of type {cast_func.__name__}.")


def interactive_main():
    print("HRAudioWizard CLI - Interactive Mode\n")
    # File selection
    while True:
        files = input("Enter input WAV file paths (comma-separated): ").strip()
        file_list = [f.strip() for f in files.split(',') if f.strip()]
        if file_list:
            break
        print("Please enter at least one file.")

    bit_depth = prompt_with_default("Output bit depth", 24, int, [24, 32, 64])
    scale = prompt_with_default("Sampling rate scale (x1, x2, x4, x8)", 1, int, [1, 2, 4, 8])
    remaster = prompt_with_default("Enable remastering? (y/n)", 'n', str.lower, ['y', 'n']) == 'y'
    noise_reduction = prompt_with_default("Enable noise reduction? (y/n)", 'n', str.lower, ['y', 'n']) == 'y'
    high_freq_comp = prompt_with_default("Enable high frequency compensation? (y/n)", 'n', str.lower, ['y', 'n']) == 'y'
    compressed_mode = prompt_with_default("Enable compressed source mode? (y/n)", 'n', str.lower, ['y', 'n']) == 'y'
    lowpass = prompt_with_default("Lowpass filter frequency (Hz)", 16000, int)

    class Args:
        pass
    args = Args()
    args.files = file_list
    args.bit_depth = bit_depth
    args.scale = scale
    args.remaster = remaster
    args.noise_reduction = noise_reduction
    args.high_freq_comp = high_freq_comp
    args.compressed_mode = compressed_mode
    args.lowpass = lowpass

    main(args)

def main(args):
    settings = {
        'bit_depth': args.bit_depth,
        'scale': args.scale,
        'remaster': getattr(args, 'remaster', False),
        'noise_reduction': getattr(args, 'noise_reduction', False),
        'high_freq_comp': getattr(args, 'high_freq_comp', False),
        'compressed_mode': getattr(args, 'compressed_mode', False),
        'lowpass': getattr(args, 'lowpass', 16000),
    }
    processor = AudioProcessor(input_device=None, output_device=None, settings=settings)
    for file in args.files:
        print(f"Processing {file}...")
        try:
            output_file = processor.process(file)
            print(f"Output written to: {output_file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        interactive_main()
    else:
        parser = argparse.ArgumentParser(description='HRAudioWizard CLI - High Resolution Audio Batch Processor')
        parser.add_argument('files', nargs='+', help='Input WAV files to process')
        parser.add_argument('--bit-depth', type=int, choices=[24, 32, 64], default=24, help='Output bit depth (24, 32, 64)')
        parser.add_argument('--scale', type=int, choices=[1, 2, 4, 8], default=1, help='Sampling rate scale (x1, x2, x4, x8)')
        parser.add_argument('--remaster', action='store_true', help='Enable remastering')
        parser.add_argument('--noise-reduction', action='store_true', help='Enable noise reduction')
        parser.add_argument('--high-freq-comp', action='store_true', help='Enable high frequency compensation')
        parser.add_argument('--compressed-mode', action='store_true', help='Enable compressed source mode')
        parser.add_argument('--lowpass', type=int, default=16000, help='Lowpass filter frequency (Hz)')
        args = parser.parse_args()
        main(args)
