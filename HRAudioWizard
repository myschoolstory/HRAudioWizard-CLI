import numpy as np
import scipy
import librosa
import pyaudio
import soundfile as sf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox,
                             QCheckBox, QSpinBox, QListWidget, QGroupBox, QRadioButton,
                             QMessageBox, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import warnings
import argparse
import hashlib
import os
import base64
import datetime
from concurrent.futures import ThreadPoolExecutor
import queue
import sys
import tqdm

# Constants
FFTSIZE = 2048
HOPSIZE = 1024

CHUNK_SIZE = 4096  # Increased for better frequency resolution
SAMPLE_RATE = 44100
HOPSIZE = CHUNK_SIZE // 4

class Authed:
    auth_mode = 1
    authorized = -1
    date = -1
    licensekey = b""

auth_info = Authed()

def auth(licensefile):
    try:
        with open(licensefile, "r") as f:
            license = f.read()
    except FileNotFoundError:
        return 0

    key_1 = 55137268922164752821258604495339660756392940817231109643876377286678448710862
    hash_expected = "3134ea1621834b9bbaf7ee7fb42b33677fb091a66252ce3d9c149f98aa87587a"

    if hashlib.sha256(key_1.to_bytes(32, 'big')).hexdigest() != hash_expected:
        print("ソフトが改造された可能性があります。作者へ連絡をください")
        input("OK?")
        sys.exit(0)

    try:
        dateval, hash_c = license.split(",")
        authdate = int((int(dateval) ^ key_1) ** (1/8))

        if authdate <= 1707513155:
            auth_info.authorized = 0
        else:
            dt = datetime.datetime.fromtimestamp(authdate)
            if hashlib.sha256(authdate.to_bytes(32, 'big')).hexdigest() == hash_c:
                auth_info.date = dt
                auth_info.authorized = 1
                auth_info.licensekey = str(base64.b64encode(int(dateval).to_bytes(32, "big"))).replace("'", "")[1:]
            else:
                auth_info.authorized = 0
    except:
        auth_info.authorized = 0

    if auth_info.auth_mode == 1 and auth_info.authorized == 0:
        print("無効なライセンスファイルです")
        input("OK?")
        sys.exit(0)

    return 0

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


class AudioProcessor(QThread):
    error_occurred = pyqtSignal(str)

    def __init__(self, input_device, output_device, settings):
        super().__init__()
        self.input_device = input_device
        self.output_device = output_device
        self.settings = settings
        self.running = False
        self.audio_queue = queue.Queue(maxsize=10)

    def run(self):
        try:
            p = pyaudio.PyAudio()
            input_stream = p.open(format=pyaudio.paFloat32,
                                  channels=2,
                                  rate=SAMPLE_RATE,
                                  input=True,
                                  input_device_index=self.input_device,
                                  frames_per_buffer=CHUNK_SIZE)

            output_stream = p.open(format=pyaudio.paFloat32,
                                   channels=2,
                                   rate=SAMPLE_RATE,
                                   output=True,
                                   output_device_index=self.output_device,
                                   frames_per_buffer=CHUNK_SIZE)

            self.running = True
            while self.running:
                data = input_stream.read(CHUNK_SIZE)
                audio_chunk = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)

                processed_chunk = self.process_audio(audio_chunk)

                output_stream.write(processed_chunk.astype(np.float32).tobytes())

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if 'input_stream' in locals():
                input_stream.stop_stream()
                input_stream.close()
            if 'output_stream' in locals():
                output_stream.stop_stream()
                output_stream.close()
            p.terminate()

    def process_audio(self, audio_chunk):
        if self.settings['high_freq_comp']:
            audio_chunk = self.apply_hfc(audio_chunk)
        if self.settings['noise_reduction']:
            audio_chunk = self.apply_noise_reduction(audio_chunk)
        if self.settings['remaster']:
            audio_chunk = self.apply_remaster(audio_chunk)
        return audio_chunk

    def apply_hfc(self, audio_chunk):
        mid = (audio_chunk[:, 0] + audio_chunk[:, 1]) / 2
        side = (audio_chunk[:, 0] - audio_chunk[:, 1]) / 2

        mid_ffted = librosa.stft(mid, n_fft=CHUNK_SIZE, hop_length=HOPSIZE)
        side_ffted = librosa.stft(side, n_fft=CHUNK_SIZE, hop_length=HOPSIZE)

        mid_mag, mid_phase = librosa.magphase(mid_ffted)
        side_mag, side_phase = librosa.magphase(side_ffted)

        mid_hm, mid_pc = librosa.decompose.hpss(mid_mag, kernel_size=31, mask=True)
        side_hm, side_pc = librosa.decompose.hpss(side_mag, kernel_size=31, mask=True)

        # Simplified high frequency compensation
        high_freq_boost = np.linspace(1, 2, mid_hm.shape[0])
        mid_hm *= high_freq_boost[:, np.newaxis]
        mid_pc *= high_freq_boost[:, np.newaxis]
        side_hm *= high_freq_boost[:, np.newaxis]
        side_pc *= high_freq_boost[:, np.newaxis]

        mid_boosted = mid_hm * mid_phase + mid_pc * mid_phase
        side_boosted = side_hm * side_phase + side_pc * side_phase

        mid_boosted = librosa.istft(mid_boosted, hop_length=HOPSIZE)
        side_boosted = librosa.istft(side_boosted, hop_length=HOPSIZE)

        return np.column_stack((mid_boosted + side_boosted, mid_boosted - side_boosted))

    def apply_noise_reduction(self, audio_chunk):
        # Simplified noise reduction using spectral gating
        mid = (audio_chunk[:, 0] + audio_chunk[:, 1]) / 2
        side = (audio_chunk[:, 0] - audio_chunk[:, 1]) / 2

        mid_ffted = librosa.stft(mid, n_fft=CHUNK_SIZE, hop_length=HOPSIZE)
        side_ffted = librosa.stft(side, n_fft=CHUNK_SIZE, hop_length=HOPSIZE)

        mid_mag, mid_phase = librosa.magphase(mid_ffted)
        side_mag, side_phase = librosa.magphase(side_ffted)

        # Simple spectral gating
        threshold = np.mean(mid_mag) * 0.1
        mid_mag[mid_mag < threshold] *= 0.1
        side_mag[side_mag < threshold] *= 0.1

        mid_reduced = mid_mag * mid_phase
        side_reduced = side_mag * side_phase

        mid_reduced = librosa.istft(mid_reduced, hop_length=HOPSIZE)
        side_reduced = librosa.istft(side_reduced, hop_length=HOPSIZE)

        return np.column_stack((mid_reduced + side_reduced, mid_reduced - side_reduced))

    def apply_remaster(self, audio_chunk):
        mid = (audio_chunk[:, 0] + audio_chunk[:, 1]) / 2
        side = (audio_chunk[:, 0] - audio_chunk[:, 1]) / 2

        mid_ffted = librosa.stft(mid, n_fft=CHUNK_SIZE, hop_length=HOPSIZE)
        side_ffted = librosa.stft(side, n_fft=CHUNK_SIZE, hop_length=HOPSIZE)

        mid_mag, mid_phase = librosa.magphase(mid_ffted)
        side_mag, side_phase = librosa.magphase(side_ffted)

        # Simplified remastering: enhance bass and treble
        freq_boost = np.concatenate([np.linspace(1.5, 1, mid_mag.shape[0]//4),
                                     np.ones(mid_mag.shape[0]//2),
                                     np.linspace(1, 1.5, mid_mag.shape[0]//4)])
        mid_mag *= freq_boost[:, np.newaxis]
        side_mag *= freq_boost[:, np.newaxis]

        mid_remastered = mid_mag * mid_phase
        side_remastered = side_mag * side_phase

        mid_remastered = librosa.istft(mid_remastered, hop_length=HOPSIZE)
        side_remastered = librosa.istft(side_remastered, hop_length=HOPSIZE)

        return np.column_stack((mid_remastered + side_remastered, mid_remastered - side_remastered))

    def stop(self):
        self.running = False

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

class AudioConversionThread(QThread):
    progress_updated = pyqtSignal(int, int)
    conversion_complete = pyqtSignal()
    file_converted = pyqtSignal(str)

    def __init__(self, files, settings):
        super().__init__()
        self.files = files
        self.settings = settings

    def run(self):
        total_files = len(self.files)
        for i, file in enumerate(self.files):
            self.convert_file(file)
            self.file_converted.emit(file)
            overall_progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(100, overall_progress)
        self.conversion_complete.emit()

    def convert_file(self, file):
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

        for progress in range(101):
            self.progress_updated.emit(progress, 0)
            self.msleep(10)

class HRAudioWizard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio_processor = None

    def initUI(self):
        self.setWindowTitle('HRAudioWizard')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Offline processing tab
        offline_tab = QWidget()
        offline_layout = QVBoxLayout()
        offline_tab.setLayout(offline_layout)
        tabs.addTab(offline_tab, "Offline Processing")

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        self.file_list = QListWidget()
        add_file_btn = QPushButton("Add Files")
        add_file_btn.clicked.connect(self.add_files)
        file_layout.addWidget(self.file_list)
        file_layout.addWidget(add_file_btn)
        file_group.setLayout(file_layout)
        offline_layout.addWidget(file_group)

        # Conversion settings
        settings_group = QGroupBox("Conversion Settings")
        settings_layout = QVBoxLayout()

        # Bit depth
        bit_depth_layout = QHBoxLayout()
        bit_depth_layout.addWidget(QLabel("Bit Depth:"))
        self.bit_depth_24 = QRadioButton("24-bit")
        self.bit_depth_32 = QRadioButton("32-bit")
        self.bit_depth_64 = QRadioButton("64-bit")
        self.bit_depth_24.setChecked(True)
        bit_depth_layout.addWidget(self.bit_depth_24)
        bit_depth_layout.addWidget(self.bit_depth_32)
        bit_depth_layout.addWidget(self.bit_depth_64)
        settings_layout.addLayout(bit_depth_layout)

        # Sampling rate
        sampling_rate_layout = QHBoxLayout()
        sampling_rate_layout.addWidget(QLabel("Sampling Rate:"))
        self.sampling_rate = QComboBox()
        self.sampling_rate.addItems(["x1", "x2", "x4", "x8"])
        sampling_rate_layout.addWidget(self.sampling_rate)
        settings_layout.addLayout(sampling_rate_layout)

        # Checkboxes
        self.remaster_cb = QCheckBox("Remaster")
        self.noise_reduction_cb = QCheckBox("Noise Reduction")
        self.high_freq_comp_cb = QCheckBox("High Frequency Compensation")
        self.compressed_mode_cb = QCheckBox("Compressed Source Mode")
        settings_layout.addWidget(self.remaster_cb)
        settings_layout.addWidget(self.noise_reduction_cb)
        settings_layout.addWidget(self.high_freq_comp_cb)
        settings_layout.addWidget(self.compressed_mode_cb)

        # Lowpass filter
        lowpass_layout = QHBoxLayout()
        lowpass_layout.addWidget(QLabel("Lowpass Filter (Hz):"))
        self.lowpass_filter = QSpinBox()
        self.lowpass_filter.setRange(4000, 50000)
        self.lowpass_filter.setSingleStep(1000)
        self.lowpass_filter.setValue(16000)
        lowpass_layout.addWidget(self.lowpass_filter)
        settings_layout.addLayout(lowpass_layout)

        settings_group.setLayout(settings_layout)
        offline_layout.addWidget(settings_group)

        # Conversion progress
        progress_layout = QVBoxLayout()
        self.current_file_label = QLabel("Current File: ")
        progress_layout.addWidget(self.current_file_label)
        self.file_progress_bar = QProgressBar()
        progress_layout.addWidget(self.file_progress_bar)
        self.overall_progress_bar = QProgressBar()
        progress_layout.addWidget(self.overall_progress_bar)
        offline_layout.addLayout(progress_layout)

        # Conversion button
        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.clicked.connect(self.start_conversion)
        offline_layout.addWidget(self.convert_btn)

        # Realtime processing tab
        realtime_tab = QWidget()
        realtime_layout = QVBoxLayout()
        realtime_tab.setLayout(realtime_layout)
        tabs.addTab(realtime_tab, "Realtime Processing")

        # Input device selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Device:"))
        self.input_device_combo = QComboBox()
        input_layout.addWidget(self.input_device_combo)
        realtime_layout.addLayout(input_layout)

        # Output device selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Device:"))
        self.output_device_combo = QComboBox()
        output_layout.addWidget(self.output_device_combo)
        realtime_layout.addLayout(output_layout)

        # Realtime processing options
        self.realtime_hfc_checkbox = QCheckBox("High Frequency Compensation")
        self.realtime_noise_reduction_checkbox = QCheckBox("Noise Reduction")
        self.realtime_remaster_checkbox = QCheckBox("Remaster")
        realtime_layout.addWidget(self.realtime_hfc_checkbox)
        realtime_layout.addWidget(self.realtime_noise_reduction_checkbox)
        realtime_layout.addWidget(self.realtime_remaster_checkbox)

        # Start/Stop button
        self.toggle_button = QPushButton("Start Processing")
        self.toggle_button.clicked.connect(self.toggle_processing)
        realtime_layout.addWidget(self.toggle_button)

        # Auth info button
        auth_info_btn = QPushButton("Show Auth Info")
        auth_info_btn.clicked.connect(self.show_auth_info)
        main_layout.addWidget(auth_info_btn)

        self.populate_audio_devices()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select audio files", "", "WAV Files (*.wav)")
        self.file_list.addItems(files)

    def start_conversion(self):
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            QMessageBox.warning(self, "No Files", "Please add files to convert.")
            return

        settings = {
            'bit_depth': 24 if self.bit_depth_24.isChecked() else (32 if self.bit_depth_32.isChecked() else 64),
            'scale': int(self.sampling_rate.currentText()[1:]),
            'remaster': self.remaster_cb.isChecked(),
            'noise_reduction': self.noise_reduction_cb.isChecked(),
            'high_freq_comp': self.high_freq_comp_cb.isChecked(),
            'compressed_mode': self.compressed_mode_cb.isChecked(),
            'lowpass': self.lowpass_filter.value()
        }

        self.conversion_thread = AudioConversionThread(files, settings)
        self.conversion_thread.progress_updated.connect(self.update_progress)
        self.conversion_thread.conversion_complete.connect(self.conversion_finished)
        self.conversion_thread.file_converted.connect(self.file_converted)

        # Disable UI elements
        self.convert_btn.setEnabled(False)
        self.file_list.setEnabled(False)
        self.set_settings_enabled(False)

        self.conversion_thread.start()

    def update_progress(self, file_progress, overall_progress):
        self.file_progress_bar.setValue(file_progress)
        self.overall_progress_bar.setValue(overall_progress)

    def file_converted(self, file):
        self.current_file_label.setText(f"Current File: {os.path.basename(file)}")
        self.file_list.takeItem(0)

    def conversion_finished(self):
        self.file_progress_bar.setValue(100)
        self.overall_progress_bar.setValue(100)
        self.current_file_label.setText("Conversion Complete")

        # Re-enable UI elements
        self.convert_btn.setEnabled(True)
        self.file_list.setEnabled(True)
        self.set_settings_enabled(True)

        QMessageBox.information(self, "Conversion Complete", "All files have been converted successfully.")

    def set_settings_enabled(self, enabled):
        self.bit_depth_24.setEnabled(enabled)
        self.bit_depth_32.setEnabled(enabled)
        self.bit_depth_64.setEnabled(enabled)
        self.sampling_rate.setEnabled(enabled)
        self.remaster_cb.setEnabled(enabled)
        self.noise_reduction_cb.setEnabled(enabled)
        self.high_freq_comp_cb.setEnabled(enabled)
        self.compressed_mode_cb.setEnabled(enabled)
        self.lowpass_filter.setEnabled(enabled)

    def populate_audio_devices(self):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            self.input_device_combo.addItem(dev_info['name'], i)
            self.output_device_combo.addItem(dev_info['name'], i)
        p.terminate()

    def toggle_processing(self):
        if self.audio_processor is None or not self.audio_processor.running:
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self):
        input_device = self.input_device_combo.currentData()
        output_device = self.output_device_combo.currentData()
        settings = {
            'high_freq_comp': self.realtime_hfc_checkbox.isChecked(),
            'noise_reduction': self.realtime_noise_reduction_checkbox.isChecked(),
            'remaster': self.realtime_remaster_checkbox.isChecked()
        }
        self.audio_processor = AudioProcessor(input_device, output_device, settings)
        self.audio_processor.error_occurred.connect(self.handle_error)
        self.audio_processor.start()
        self.toggle_button.setText("Stop Processing")

    def stop_processing(self):
        if self.audio_processor:
            self.audio_processor.stop()
            self.audio_processor.wait()
            self.audio_processor = None
        self.toggle_button.setText("Start Processing")

    def handle_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")
        self.stop_processing()

    def show_auth_info(self):
        info = ""
        if auth_info.authorized == 1:
            info = (f"Using the full version\n"
                    f"Serial Number: {auth_info.licensekey}\n"
                    f"Purchase Time: {auth_info.date.strftime('%Y/%m/%d %H:%M:%S')}")
        elif auth_info.authorized == -1:
            info = "Using the unlimited free version\nPlease visit the author's website to purchase."
        elif auth_info.authorized == 0:
            info = "Using the free trial version\nPlease visit the author's website to purchase."

        QMessageBox.information(self, "Authorization Information", info)

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

class AudioConversionThread(QThread):
    progress_updated = pyqtSignal(int, int)
    conversion_complete = pyqtSignal()
    file_converted = pyqtSignal(str)

    def __init__(self, files, settings):
        super().__init__()
        self.files = files
        self.settings = settings

    def run(self):
        total_files = len(self.files)
        for i, file in enumerate(self.files):
            self.convert_file(file)
            self.file_converted.emit(file)
            overall_progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(100, overall_progress)
        self.conversion_complete.emit()

    def convert_file(self, file):
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

        for progress in range(101):
            self.progress_updated.emit(progress, 0)
            self.msleep(10)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Load license
    license_path = os.path.join(os.path.dirname(sys.argv[0]), "license.lc_hraw")
    auth(license_path)

    # Create and show the main window
    ex = HRAudioWizard()
    ex.show()

    sys.exit(app.exec())
