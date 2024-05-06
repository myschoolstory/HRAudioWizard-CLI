import numpy as np
import scipy
import librosa
import tqdm
import sys
import pyaudio
import random
import soundfile as sf
import PySimpleGUI as sg
import warnings
import argparse
import asyncio
import hashlib
import os
import base64
import matplotlib.pyplot as plt
import matplotlib
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import datetime

#Developed By Eurobeat-Lover @ YH Released from 2022/11/3
#This is still Beta Code, I hope It'll be more excite project than now
#Premium Code

def xorshift(generator, seed=None):
    ret = seed
    def inner():
        nonlocal ret
        if ret is None:
            ret = generator()
        else:
            ret = generator(*ret)
        return ret[-1]
    return inner


def xor32(y=2463534242):
    y = y ^ (y << 13 & 0xFFFFFFFF)
    y = y ^ (y >> 17 & 0xFFFFFFFF)
    y = y ^ (y << 5 & 0xFFFFFFFF)
    return y & 0xFFFFFFFF,

class Authed:
    auth_mode = 1
    authorized = -1
    date = -1
    licensekey = b""

auth_info = Authed()

def auth(licensefile):
    try:
        f = open(licensefile, "r")
    except:
        return 0
    license= f.read()
    f.close()
    try:
        key_1 = 55137268922164752821258604495339660756392940817231109643876377286678448710862
        hash = "3134ea1621834b9bbaf7ee7fb42b33677fb091a66252ce3d9c149f98aa87587a"
    except:
        pass
    if not hashlib.sha256(key_1.to_bytes(32, 'big')).hexdigest() == hash:
        print("ソフトが改造された可能性があります。作者へ連絡をください")
        input("OK?")
        sys.exit(0)
    try:
        dateval, hash_c = license.split(",")
    except:
        pass
    try:
        authdate = int((int(dateval) ^ key_1) ** (1/8))
        if not authdate > 1707513155:
            auth_info.authorized = 0
        dt = datetime.datetime.fromtimestamp(authdate)
        if hashlib.sha256(authdate.to_bytes(32, 'big')).hexdigest() == hash_c:
            auth_info.date = dt
            auth_info.authorized = 1
            auth_info.licensekey = str(base64.b64encode(int(license.split(",")[0]).to_bytes(32, "big"))).replace("'", "")[1:]
        else:
            auth_info.authorized = 0
    except:
        pass
    if auth_info.auth_mode == 1 and auth_info.authorized == 0:
        print("無効なライセンスファイルです")
        input("OK?")
        sys.exit(0)
    return 0

warnings.simplefilter('ignore')

def detect_overtone_peaks(db_hm, lowpass_fft, fft_size, fs):
    energy_threshold = 0.00001 # エネルギーの閾値（dB）
    min_peak_distance = int(fs / fft_size)  # ピーク間の最小距離（サンプル数）

    peaks = []
    for i in range(0, len(db_hm)):
        if np.abs(db_hm[i]) < energy_threshold:
            continue

        # 局所的なピークの検出
        if i > 0 and i < len(db_hm) - 1 and db_hm[i] > db_hm[i-1] and db_hm[i] > db_hm[i+1]:
            # 調波構造の考慮
            if i % min_peak_distance == 0:
                peaks.append(i)

    # 時間的な連続性の考慮
    stable_peaks = []
    for peak in peaks:
        if len(stable_peaks) == 0 or peak - stable_peaks[-1] >= min_peak_distance:
            stable_peaks.append(peak)

    return np.array(stable_peaks)

def remaster(dat, fs, scale):
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2
    mid_ffted = librosa.stft(mid, n_fft=2048, hop_length=1024)
    side_ffted = librosa.stft(side, n_fft=2048, hop_length=1024)
    mid_proc = np.zeros([mid_ffted.shape[0]*scale, mid_ffted.shape[1]], dtype=np.complex128)
    side_proc = np.zeros([mid_ffted.shape[0]*scale, mid_ffted.shape[1]], dtype=np.complex128)
    for i in tqdm.tqdm(range(mid_ffted.shape[1])):
        sample_mid = np.hstack([mid_ffted[:,i], np.zeros(len(mid_ffted[:,i])*(scale-1))])
        sample_side = np.hstack([side_ffted[:,i], np.zeros(len(side_ffted[:,i])*(scale-1))])
        db_mid = librosa.amplitude_to_db(sample_mid)
        db_side = librosa.amplitude_to_db(sample_side)
        for j in range(mid_ffted.shape[0]*scale):
            try:
                if db_mid[j] > db_side[j]:
                    db_side[j] += (db_mid[j] - db_side[j]) / 2
                if j > 128 and db_side[j] < -6:
                    db_side[j] += (db_mid[j] - db_side[j]) / 2
                if db_side[j] > db_mid[j]:
                    db_mid[j] += (db_side[j] - db_mid[j]) / 2
                if j > 128 and db_mid[j] < -6:
                    db_mid[j] += (db_side[j] - db_mid[j]) / 1
            except:
                pass
            try:
                if db_mid[j] < -6:
                    db_mid[j] = (db_mid[j-1] + db_mid[j+1]) / 2
                if db_side[j] < -6:
                    db_side[j] = (db_side[j-1] + db_side[j+1]) / 2
            except:
                pass
            '''if j > 256:
                #db_mid[j-8:j] -= np.hstack([np.linspace(1,8,4), np.linspace(8,1,4)])
                #db_side[j-8:j] -= np.hstack([np.linspace(1,8,4), np.linspace(8,1,4)])
                try:
                    if db_mid[j] < -12:
                        if db_mid[j] < -12 and not np.all(db_mid):
                            db_mid[j] = (db_mid[int(j/2)] - (db_mid[int(j/4)] - db_mid[int(j/2)]))
                        else:
                            db_mid[j] += (db_mid[int(j/2)] - (db_mid[int(j/4)] - db_mid[int(j/2)]))
                    if db_side[j] < -12:
                        if db_side[j] < -12 and not np.all(db_side):
                            db_side[j] = (db_side[int(j/2)] - (db_side[int(j/4)] - db_side[int(j/2)]))
                        else:
                            db_side[j] += (db_side[int(j/2)] - (db_side[int(j/4)] - db_side[int(j/2)]))
                except:
                    pass'''
        #if db_mid[512] > -36:
        #    db_mid[512:] -= np.abs(db_mid[:513][::-1])*-1 * np.linspace(1,3,513) / 3
        #    db_side[512:] -= np.abs(db_side[:513][::-1])*-1 * np.linspace(1,3,513) / 3
        db_mid[0] = -12
        db_side[0] = -12
        db_mid[256:] += np.linspace(0,int((db_mid[1]-db_mid[256])/16),len(db_mid)-256)
        db_side[256:] += np.linspace(0,int((db_mid[1]-db_mid[256])/16),len(db_side)-256)
        mid_proc[:,i] = librosa.db_to_amplitude(db_mid) * np.exp(1.j * np.angle(sample_mid))
        side_proc[:,i] = librosa.db_to_amplitude(db_side) * np.exp(1.j * np.angle(sample_side))
    mid = librosa.istft(mid_proc, n_fft=2048*scale, hop_length=1024)
    side = librosa.istft(side_proc, n_fft=2048*scale, hop_length=1024)
    return np.array([mid + side, mid - side]).T * scale

random32=xorshift(xor32)

def proc_for_compressed(mid, side):
    mid_db = librosa.amplitude_to_db(mid)
    side_db = librosa.amplitude_to_db(side)
    if np.sum(np.abs(mid_db))/len(mid) > 60:
        return mid, side
    if np.sum(np.abs(side_db))/len(side) > 60:
        return mid, side
    for i in range(len(mid)):
        if (mid_db[i]) < -12 and np.min(mid_db) > -60:
            mid[i] += np.random.randn(1)/(2**10) * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=1))
        if (side_db[i]) < -12 and np.min(side_db) > -60:
            side[i] += np.random.randn(1)/(2**10) * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=1))
            if i > 20:
                side[i] += (np.abs(mid[i]) / 4) * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=1))
    return mid, side

def look_for_audio_input():
    dev = []
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        dev.append(pa.get_device_info_by_index(i)["name"])
    pa.terminate()
    return dev

def hires_playback(dev, dev2):
    fs = 48000
    devs = look_for_audio_input()
    p = pyaudio.PyAudio()
    p2 = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,channels=2,rate=fs,input=True,input_device_index = devs.index(dev),output_device_index = devs.index(dev2),frames_per_buffer=8192)
    playstream = p2.open(format=pyaudio.paFloat32,channels=2,rate=fs,output=True,output_device_index = devs.index(dev2),frames_per_buffer=8192)
    while True:
        try:
            d = (stream.read(2048*2))
            data = (np.frombuffer(d, dtype=np.float32))
            hf = hfp_2(data.copy(), fs)
            playstream.write(hf.astype(np.float32).tostring())
        except:
            pass
    return

def estimate_amplitudes(power, num_harmonics):
    # パワースペクトルからケプストラムを計算
    cepstrum = np.fft.ifft(np.log(power), n=num_harmonics).real

    # ケプストラムのリフタリング
    lifter_size = 32
    lifter = np.zeros((num_harmonics))
    lifter[:lifter_size] = 1
    lifter[-lifter_size+1:] = 1
    cepstrum_liftered = cepstrum * lifter

    # リフタリングされたケプストラムからスペクトル包絡を推定
    spec_env = np.exp(np.fft.fft(cepstrum_liftered, n=num_harmonics))
    # 倍音の振幅を抽出
    amplitudes = np.zeros(num_harmonics)
    for k in range(num_harmonics):
        idx = int(np.round((k+1) * len(power) / num_harmonics))
        amplitudes[k] = np.abs(spec_env[idx])

    return amplitudes

# 位相の推定（最小位相推定）
def estimate_phases(slope, num_harmonics):
    # 最小位相応答を計算
    min_phase = np.exp(1j * np.cumsum(slope))

    # 倍音の位相を抽出
    phases = np.angle(min_phase[:num_harmonics])

    return phases

# スペクトル包絡の推定（LPC分析）
def estimate_spectral_envelope(signal, order):
    # LPC係数の推定
    a = librosa.lpc(np.array(signal), order=order)

    # LPC係数からスペクトル包絡を計算
    w, h = freqz(1, a, worN=len(signal))
    spec_env = np.abs(h)

    return spec_env


class OVERTONE:
    width = 2
    amplitude = 0
    base_freq = 0
    slope = []
    loop = 0
    pass

def synthesize_overtones(ot_arr, lowpass_fft, fft_size, fs):
    rebuild_mid = np.zeros(fft_size//2+1, dtype=np.float64)

    for ot in ot_arr:
        base_freq = ot.base_freq
        num_harmonics = int(fft_size / base_freq) * 2

        # 振幅と位相の推定
        amplitudes = estimate_amplitudes(ot.power, num_harmonics=num_harmonics)

        # 倍音構造の考慮
        for k in range(0, num_harmonics-1):
            freq = base_freq * k
            amplitude = amplitudes[k]

            rebuild_mid[int(freq-ot.width/2):int(freq+ot.width/2)+1] += amplitude / 8
    print(rebuild_mid[lowpass_fft:])
    return rebuild_mid

def hfp(dat, lowpass, fs, compressd_mode=False, use_hpss=True):
    fft_size = 2048
    hop_length = 1024
    if compressd_mode == True:
        fft_size *= 1
    lowpass_fft = int((int(fft_size/2+1))*(lowpass/(fs/2)))
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2
    mid_ffted = librosa.stft(mid, n_fft=(fft_size), hop_length=hop_length)
    side_ffted = librosa.stft(side, n_fft=(fft_size), hop_length=hop_length)
    mid_mag, mid_phs = librosa.magphase(mid_ffted)
    side_mag, side_phs = librosa.magphase(side_ffted)
    if use_hpss == True:
        mid_hm, mid_pc = librosa.decompose.hpss(mid_mag, kernel_size=31)
        side_hm, side_pc = librosa.decompose.hpss(side_mag, kernel_size=31)
    else:
        mid_pc, mid_hm = mid_mag, np.zeros(mid_mag.shape)
        side_pc, side_hm = side_mag, np.zeros(mid_mag.shape)
    scale = int(fs / 48000 + 0.555555)
    for i in tqdm.tqdm(range(mid_hm.shape[1])):
        ot_arr = []
        sample_mid_hm = mid_hm[:,i]
        sample_mid_hm[lowpass_fft:] = np.zeros(len(sample_mid_hm[lowpass_fft:]))
        sample_mid_pc = ((mid_pc[:,i]))
        sample_mid_pc[lowpass_fft:] = np.zeros(len(sample_mid_hm[lowpass_fft:]))
        sample_side_hm = side_hm[:,i]
        sample_side_hm[lowpass_fft:] = np.zeros(len(sample_mid_hm[lowpass_fft:]))
        sample_side_pc = ((side_pc[:,i]))
        sample_side_pc[lowpass_fft:] = np.zeros(len(sample_mid_hm[lowpass_fft:]))
        rebuild_mid = np.zeros(len(sample_mid_hm))
        rebuild_noise_mid = np.zeros(len(sample_mid_pc))
        rebuild_side = np.zeros(len(sample_mid_hm))
        rebuild_noise_side = np.zeros(len(sample_mid_pc))
        db_mid_hm = sample_mid_hm
        db_mid_pc = sample_mid_pc
        db_side_hm = sample_side_hm
        db_side_pc = sample_side_pc
        db_hm_max = scipy.signal.find_peaks(db_mid_hm)[0]
        for j in range(len(db_hm_max)):
            try:
                a = db_hm_max[j]
            except:
                break
            for k in range(int(int(fft_size/2+1)/a)):
                if np.abs(a - k*a) < 1:
                    db_hm_max = np.delete(db_hm_max, j)
                    j += 1
                    break
        db_pc_max = scipy.signal.find_peaks(db_mid_pc)[0]
        db_side_hm_max = scipy.signal.find_peaks(db_side_hm)[0]
        for j in range(len(db_side_hm_max)):
            try:
                a = db_side_hm_max[j]
            except:
                break
            for k in range(int(int(fft_size/2+1)/a)):
                if np.abs(a - k*a) < 1:
                    db_side_hm_max = np.delete(db_side_hm_max, j)
                    j += 1
        db_side_pc_max = scipy.signal.find_peaks(db_side_pc)[0]
        for j in db_hm_max:
            ot = OVERTONE()
            ot.base_freq = int(j / 2)
            ot.loop = int(int(fft_size/2+1)/(ot.base_freq/2))
            a = []
            d = 0
            for l in range(ot.loop):
                try:
                    a.append(db_mid_hm[j*l] / db_mid_hm[j])
                    if a[-1] > lowpass_fft and d != 0:
                        d = l
                except:
                    break
            a = np.array(a)
            b = scipy.fft.idst(a)
            b += np.linspace(1,-1,len(b))
            c = scipy.fft.dst(b)
            ot.slope = c / 8 * np.linspace(1,0,len(c))
            for k in range(2,6,2):
                try:
                    if np.abs(np.abs(db_mid_hm[j-int(k/2)]) - np.abs(db_mid_hm[j+int(k/2)])) < 4:
                        ot.width = k
                        break
                except:
                    break
            ot.power = db_mid_hm[j-int(ot.width/2):j+int(ot.width/2)]
            ot_arr.append(ot)
            b = []
            a = []

        for j in ot_arr:
            for k in range(1,j.loop):
                try:
                    rebuild_mid[int(j.base_freq*(int(k))-int(j.width/2)):int(j.base_freq*(int(k))+int(j.width/2))] += j.power * np.abs(j.slope[int(k)])
                except:
                    pass
        ot_arr = []
        for j in db_pc_max:
            ot = OVERTONE()
            ot.base_freq = j
            ot.loop = int(int(fft_size/2+1)/ot.base_freq)
            a = []
            d = 0
            for l in range(ot.loop):
                try:
                    a.append(db_mid_pc[j*l]/db_mid_pc[j])
                    if a[-1] > lowpass_fft and d != 0:
                        d = l
                except:
                    break
            a = np.array(a)
            b = scipy.fft.idst(a)
            b += np.linspace(1,-1,len(b))
            c = scipy.fft.dst(b)
            ot.slope = c / 8
            for k in range(2,4,2):
                try:
                    if np.abs(np.abs(db_mid_pc[j-int(k/2)]) - np.abs(db_mid_pc[j+int(k/2)])) < 4:
                        ot.width = k
                        break
                except:
                    break
            ot.power = db_mid_pc[j-int(ot.width/2):j+int(ot.width/2)]
            ot_arr.append(ot)
            b = []
            a = []
            temp = []

        # ノイズ成分の合成
        for j in ot_arr:
            for k in range(1,j.loop*2):
                try:
                    rebuild_noise_mid[int(j.base_freq*(int(k)+1)-int(j.width/2)):int(j.base_freq*(int(k)+1)+int(j.width/2))] = j.power * np.abs(j.slope[int(k)])
                except:
                    break

        ot_arr = []
        for j in db_side_hm_max:
            ot = OVERTONE()
            ot.base_freq = int(j / 2)
            ot.loop = int(int(fft_size/2+1)/(ot.base_freq/2))
            a = []
            d = 0
            for l in range(ot.loop):
                try:
                    a.append(db_side_hm[j*l] / db_side_hm[j])
                    if a[-1] > lowpass_fft and d != 0:
                        d = l
                except:
                    break
            a = np.array(a)
            b = scipy.fft.idst(a)
            b += np.linspace(1,-1,len(b))
            c = scipy.fft.dst(b)
            ot.slope = c / 8 * np.linspace(1,0,len(c))
            for k in range(2,6,2):
                try:
                    if np.abs(np.abs(db_side_hm[j-int(k/2)]) - np.abs(db_side_hm[j+int(k/2)])) < 4:
                        ot.width = k
                        break
                except:
                    break
            ot.power = db_side_hm[j-int(ot.width/2):j+int(ot.width/2)]
            ot_arr.append(ot)
            b = []
            a = []

        for j in ot_arr:
            for k in range(1,j.loop):
                try:
                    rebuild_side[int(j.base_freq*(int(k))-int(j.width/2)):int(j.base_freq*(int(k))+int(j.width/2))] += j.power * np.abs(j.slope[int(k)])
                except:
                    pass

        ot_arr = []
        for j in db_side_pc_max:
            ot = OVERTONE()
            ot.base_freq = j
            ot.loop = int(int(fft_size/2+1)/ot.base_freq)
            a = []
            d = 0
            for l in range(ot.loop):
                try:
                    a.append(db_side_pc[j*l])
                    if a[-1] > lowpass_fft and d != 0:
                        d = l
                except:
                    break
            a = np.array(a)
            b = scipy.fft.idst(a)
            b += np.linspace(1,-1,len(b))
            c = scipy.fft.dst(b)
            ot.slope = c / 8
            for k in range(2,4,2):
                try:
                    if np.abs(np.abs(db_side_pc[j-int(k/2)]) - np.abs(db_side_pc[j+int(k/2)])) < 4:
                        ot.width = k
                        break
                except:
                    break
            ot.power = db_side_pc[j-int(ot.width/2):j+int(ot.width/2)]
            ot_arr.append(ot)
            b = []
            a = []
            temp = []

        # 側成分のノイズ成分の合成
        for j in ot_arr:
            for k in range(1,j.loop*2):
                try:
                    rebuild_noise_side[int(j.base_freq*(int(k)+1)-int(j.width/2)):int(j.base_freq*(int(k)+1)+int(j.width/2))] = j.power * np.abs(j.slope[int(k)])
                except:
                    pass
        mid_hm[:,i][lowpass_fft:] = rebuild_mid[lowpass_fft:] * 4
        mid_pc[:,i][lowpass_fft:] = rebuild_noise_mid[lowpass_fft:] * 3
        side_hm[:,i][lowpass_fft:] = rebuild_side[lowpass_fft:] * 4
        side_pc[:,i][lowpass_fft:] = rebuild_noise_side[lowpass_fft:] * 3
        try:
            mid_hm[:,i][int(fft_size/4):] *= np.linspace(1,0,len(mid_hm[:,i][int(fft_size/4):]))
            mid_pc[:,i][int(fft_size/4):] *= np.linspace(1,0,len(mid_hm[:,i][int(fft_size/4):]))
            side_hm[:,i][int(fft_size/4):] *= np.linspace(1,0,len(mid_hm[:,i][int(fft_size/4):]))
            side_pc[:,i][int(fft_size/4):] *= np.linspace(1,0,len(mid_hm[:,i][int(fft_size/4):]))
        except:
            pass
        mid_phs[:,i][lowpass_fft:] = np.random.uniform(-np.pi, np.pi, size=len(side_phs[:,i][lowpass_fft:]))
        side_phs[:,i][lowpass_fft:] = np.random.uniform(-np.pi, np.pi, size=len(side_phs[:,i][lowpass_fft:]))
    rebuilded_mid = (mid_hm + mid_pc) * np.exp(1.j * (mid_phs))
    rebuilded_side = (side_hm + side_pc) * np.exp(1.j * (side_phs))
    sf.write("hm.wav", librosa.griffinlim(mid_hm / 4, hop_length=hop_length), fs*scale, format="WAV", subtype="PCM_16")
    sf.write("pc.wav", librosa.griffinlim(mid_pc / 4, hop_length=hop_length), fs*scale, format="WAV", subtype="PCM_16")
    if compressd_mode == True:
        for i in tqdm.tqdm(range((mid_ffted.shape[1]))):
            mid_ffted[:,i][librosa.amplitude_to_db(mid_ffted[:,i])<-12] += rebuilded_mid[:,i][librosa.amplitude_to_db(mid_ffted[:,i])<-12]
            side_ffted[:,i][librosa.amplitude_to_db(side_ffted[:,i])<-12] += rebuilded_side[:,i][librosa.amplitude_to_db(side_ffted[:,i])<-12]
            mid_ffted[:,i], side_ffted[:,i] = proc_for_compressed(mid_ffted[:,i], side_ffted[:,i])
    else:
        mid_ffted[lowpass_fft:] = rebuilded_mid[lowpass_fft:]
        side_ffted[lowpass_fft:] = rebuilded_side[lowpass_fft:]
    iffted_mid = librosa.istft(mid_ffted/scale, hop_length=hop_length)
    iffted_side = librosa.istft(side_ffted/scale, hop_length=hop_length)
    return np.array([iffted_mid+iffted_side, iffted_mid-iffted_side]).T

def hfp_2(dat, fs):
    fft_size = 2048
    dat[np.isnan(dat)] = 0
    dat[np.isinf(dat)] = 0
    mid = (dat[::2] + dat[1::2]) / 2
    side = (dat[::2] - dat[1::2]) / 2
    lowpass_fft = 512
    mid_ffted = librosa.stft(mid, n_fft=(fft_size), hop_length=int(fft_size/1),  window=scipy.signal.windows.cosine(fft_size))
    side_ffted = librosa.stft(side, n_fft=(fft_size), hop_length=int(fft_size/1), window=scipy.signal.windows.cosine(fft_size))
    mid_mag, mid_phs = librosa.magphase(mid_ffted)
    side_mag, side_phs = librosa.magphase(side_ffted)
    mid_hm, mid_pc = librosa.decompose.hpss(mid_mag, kernel_size=63, mask=True)
    side_hm, side_pc = librosa.decompose.hpss(side_mag, kernel_size=63, mask=True)
    scale = int(fs / 48000 + 0.555555)
    for i in (range(mid_hm.shape[1])):
        ot_arr = []
        sample_mid_hm = mid_hm[:,i]
        sample_mid_hm[lowpass_fft:] = np.zeros(int(fft_size/2+1)-lowpass_fft)
        sample_mid_pc = mid_pc[:,i]
        sample_mid_pc[lowpass_fft:] = np.zeros(int(fft_size/2+1)-lowpass_fft)
        sample_side_hm = side_hm[:,i]
        sample_side_hm[lowpass_fft:] = np.zeros(int(fft_size/2+1)-lowpass_fft)
        sample_side_pc = side_pc[:,i]
        sample_side_pc[lowpass_fft:] = np.zeros(int(fft_size/2+1)-lowpass_fft)
        rebuild_mid = np.zeros(len(sample_mid_hm))
        rebuild_noise_mid = np.zeros(len(sample_mid_pc))
        rebuild_side = np.zeros(len(sample_mid_hm))
        rebuild_noise_side = np.zeros(len(sample_mid_pc))
        db_mid_hm = sample_mid_hm
        db_mid_pc = sample_mid_pc
        db_side_hm = sample_side_hm
        db_side_pc = sample_side_pc
        db_hm_max = scipy.signal.find_peaks(db_mid_hm)[0]
        db_hm_max = db_hm_max[db_hm_max>int(lowpass_fft>2)]
        for j in range(len(db_hm_max)):
            try:
                a = db_hm_max[j]
            except:
                break
            for k in range(int(int(fft_size/2+1)/a)):
                if np.abs(a - k*a) < 1:
                    db_hm_max = np.delete(db_hm_max, j)
                    j += 1
                    break
        db_pc_max = scipy.signal.find_peaks(db_mid_pc)[0]
        db_pc_max = db_pc_max[db_pc_max>int(lowpass_fft/2)]
        db_side_hm_max = scipy.signal.find_peaks(db_side_hm)[0]
        db_side_hm_max = db_side_hm_max[db_side_hm_max>int(lowpass_fft/2)]
        for j in range(len(db_side_hm_max)):
            try:
                a = db_side_hm_max[j]
            except:
                break
            for k in range(int(int(fft_size/2+1)/a)):
                if np.abs(a - k*a) < 1:
                    db_side_hm_max = np.delete(db_side_hm_max, j)
                    j += 1
                    break
        db_side_pc_max = scipy.signal.find_peaks(db_side_pc)[0]
        db_side_pc_max = db_side_pc_max[db_side_pc_max>int(lowpass_fft/2)]
        for j in db_hm_max:
            ot = OVERTONE()
            ot.base_freq = j
            ot.loop = int(int(fft_size/2+1)/ot.base_freq) * 2
            a = []
            d = 0
            for l in range(ot.loop):
                a.append(db_mid_hm[j*l] / db_mid_hm[j])
                if a[-1] > lowpass_fft and d != 0:
                    d = l
            a = np.array(a)
            b = scipy.fft.idst(a)
            b += np.linspace(1,-1,len(b))
            c = scipy.fft.dst(b)
            ot.slope = c / 8
            for k in range(2,4,2):
                try:
                    if np.abs(np.abs(db_mid_hm[j-int(k/2)]) - np.abs(db_mid_hm[j+int(k/2)])) < 4:
                        ot.width = k
                        break
                except:
                    break
            ot.power = db_mid_hm[j-int(ot.width/2):j+int(ot.width/2)]
            #print(ot.width)
            ot_arr.append(ot)
            b = []
            a = []
        for j in ot_arr:
            rebuild_mid[j.base_freq-int(j.width/2):j.base_freq+int(j.width/2)] += j.power
            for k in range(1,j.loop*2):
                try:
                    #print(j.amplitude)
                    #print(j.slope)
                    rebuild_mid[int(j.base_freq*(int(k/2)+1)-int(j.width/2)):int(j.base_freq*(int(k/2)+1)+int(j.width/2))] = j.power * np.abs(j.slope[int(k/2)])
                except:
                    break
        ot_arr = []
        for j in db_pc_max:
            ot = OVERTONE()
            ot.base_freq = j
            ot.loop = int(int(fft_size/2+1)/ot.base_freq)
            a = []
            d = 0
            for l in range(ot.loop):
                a.append(db_mid_pc[j*l]/db_mid_pc[j])
                if a[-1] > lowpass_fft and d != 0:
                    d = l
            a = np.array(a)
            b = scipy.fft.idst(a)
            b += np.linspace(1,-1,len(b))
            c = scipy.fft.dst(b)
            ot.slope = c / 8
            for k in range(2,4,2):
                try:
                    if np.abs(np.abs(db_mid_pc[j-int(k/2)]) - np.abs(db_mid_pc[j+int(k/2)])) < 4:
                        ot.width = k
                        break
                except:
                    break
            ot.power = db_mid_pc[j-int(ot.width/2):j+int(ot.width/2)]
            ot_arr.append(ot)
            b = []
            a = []
            temp = []
        for j in ot_arr:
            rebuild_noise_mid[j.base_freq-int(j.width/2):j.base_freq+int(j.width/2)] += j.power
            for k in range(1,j.loop*2):
                try:
                    #print(j.amplitude)
                    #print(j.slope)
                    rebuild_noise_mid[int(j.base_freq*(int(k/2)+1)-int(j.width/2)):int(j.base_freq*(int(k/2)+1)+int(j.width/2))] = j.power * np.abs(j.slope[int(k/2)])
                except:
                    break
        ot_arr = []
        dbed_rebuild = librosa.amplitude_to_db(np.abs(rebuild_mid[lowpass_fft:lowpass_fft+60]))
        dbed_sample = librosa.amplitude_to_db(np.abs(sample_mid_hm[lowpass_fft-60:lowpass_fft]))
        diff_db = np.sum(dbed_sample)/len(dbed_sample) - (np.sum(dbed_rebuild) / len(dbed_rebuild))
        if diff_db > 24:
            diff_db = 24
        mid_hm[:,i] = librosa.db_to_amplitude(librosa.amplitude_to_db(rebuild_mid)+diff_db-15-scale*2)
        dbed_rebuild = librosa.amplitude_to_db(np.abs(rebuild_noise_mid[lowpass_fft:lowpass_fft+60]))
        dbed_sample = librosa.amplitude_to_db(np.abs(sample_mid_pc[lowpass_fft-60:lowpass_fft]))
        diff_db = np.sum(dbed_sample)/len(dbed_sample) - (np.sum(dbed_rebuild) / len(dbed_rebuild))
        if diff_db > 24:
            diff_db = 24
        mid_pc[:,i] = librosa.db_to_amplitude(librosa.amplitude_to_db(rebuild_noise_mid)+diff_db-15-scale*2)
        ot_arr = []
        for j in db_side_hm_max:
            ot = OVERTONE()
            ot.base_freq = j
            ot.loop = int(int(fft_size/2+1)/ot.base_freq)
            a = []
            d = 0
            for l in range(ot.loop):
                a.append(db_side_hm[j*l]/db_side_hm[j])
                if a[-1] > lowpass_fft and d != 0:
                    d = l
            a = np.array(a)
            b = scipy.fft.idst(a)
            b += np.linspace(1,-1,len(b))
            c = scipy.fft.dst(b)
            ot.slope = c / 8
            for k in range(2,4,2):
                try:
                    if np.abs(np.abs(db_side_hm[j-int(k/2)]) - np.abs(db_side_hm[j+int(k/2)])) < 4:
                        ot.width = k
                        break
                except:
                    break
            ot.power = db_side_hm[j-int(ot.width/2):j+int(ot.width/2)]
            ot_arr.append(ot)
            b = []
            a = []
            temp = []
        for j in ot_arr:
            rebuild_side[j.base_freq-int(j.width/2):j.base_freq+int(j.width/2)] += j.power
            for k in range(1,j.loop*2):
                try:
                    #print(j.amplitude)
                    #print(j.slope)
                    rebuild_side[int(j.base_freq*(int(k/2)+1)-int(j.width/2)):int(j.base_freq*(int(k/2)+1)+int(j.width/2))] = j.power * np.abs(j.slope[int(k/2)])
                except:
                    break
        ot_arr = []
        for j in db_side_pc_max:
            ot = OVERTONE()
            ot.base_freq = j
            ot.loop = int(int(fft_size/2+1)/ot.base_freq)
            a = []
            d = 0
            for l in range(ot.loop):
                a.append(db_side_pc[j*l])
                if a[-1] > lowpass_fft and d != 0:
                    d = l
            a = np.array(a)
            b = scipy.fft.idst(a)
            b += np.linspace(1,-1,len(b))
            c = scipy.fft.dst(b)
            #print(c)
            ot.slope = c / 8
            for k in range(2,4,2):
                try:
                    if np.abs(np.abs(db_side_pc[j-int(k/2)]) - np.abs(db_side_pc[j+int(k/2)])) < 4:
                        ot.width = k
                        break
                except:
                    break
            ot.power = db_side_pc[j-int(ot.width/2):j+int(ot.width/2)]
            ot_arr.append(ot)
            b = []
            a = []
            temp = []
        for j in ot_arr:
            rebuild_noise_side[j.base_freq-int(j.width/2):j.base_freq+int(j.width/2)] += j.power
            for k in range(1,j.loop*2):
                try:
            	    rebuild_noise_side[int(j.base_freq*(int(k/2)+1)-int(j.width/2)):int(j.base_freq*(int(k/2)+1)+int(j.width/2))] = j.power * np.abs(j.slope[int(k/2)])
                except:
                    pass
        dbed_rebuild = librosa.amplitude_to_db(np.abs(rebuild_side[lowpass_fft:lowpass_fft+60]))
        dbed_sample = librosa.amplitude_to_db(np.abs(sample_side_hm[lowpass_fft-60:lowpass_fft]))
        diff_db = np.sum(dbed_sample)/len(dbed_sample) - (np.sum(dbed_rebuild) / len(dbed_rebuild))
        if diff_db > 24:
            diff_db = 24
        side_hm[:,i] = librosa.db_to_amplitude(librosa.amplitude_to_db(rebuild_side)+diff_db-15-scale*2)
        dbed_rebuild = librosa.amplitude_to_db(np.abs(rebuild_noise_side[lowpass_fft:lowpass_fft+60]))
        dbed_sample = librosa.amplitude_to_db(np.abs(sample_side_pc[lowpass_fft-60:lowpass_fft]))
        diff_db = np.sum(dbed_sample)/len(dbed_sample) - (np.sum(dbed_rebuild) / len(dbed_rebuild))
        if diff_db > 24:
            diff_db = 24
        side_pc[:,i] = librosa.db_to_amplitude(librosa.amplitude_to_db(rebuild_noise_side)+diff_db-15-scale*2)
        env_mid_hm_ = np.abs(scipy.fft.dst(scipy.fft.idst(sample_mid_hm) + np.linspace(0.5,-0.5,len(sample_mid_hm))) / 6)
        env_mid_pc_ = np.abs(scipy.fft.dst(scipy.fft.idst(sample_side_pc) + np.linspace(0.5,-0.5,len(sample_mid_hm))) / 6)
        env_side_hm_ = np.abs(scipy.fft.dst(scipy.fft.idst(sample_mid_pc) + np.linspace(0.5,-0.5,len(sample_mid_hm))) / 6)
        env_side_pc_ = np.abs(scipy.fft.dst(scipy.fft.idst(sample_side_pc) + np.linspace(0.5,-0.5,len(sample_mid_hm))) / 6)
        for k in range(lowpass_fft, int(fft_size/2+1)-2):
            if mid_hm[:,i][k+1] < 0.01:
                mid_hm[:,i][k+1] += (mid_hm[:,i][k] + mid_hm[:,i][k+2])
            if mid_pc[:,i][k+1] < 0.01:
                mid_pc[:,i][k+1] += (mid_pc[:,i][k] + mid_pc[:,i][k+2])
            if side_hm[:,i][k+1] < 0.01:
                side_hm[:,i][k+1] += (side_hm[:,i][k] + side_hm[:,i][k+2])
            if side_pc[:,i][k+1] < 0.01:
                side_pc[:,i][k+1] += (side_pc[:,i][k] + side_pc[:,i][k+2])
        env_mid_hm_[env_mid_hm_>2] = 2
        env_mid_pc_[env_mid_pc_>2] = 2
        env_side_hm_[env_side_hm_>2] = 2
        env_side_pc_[env_side_pc_>2] = 2
        mid_hm[:,i] *= env_mid_hm_
        mid_pc[:,i] *= env_mid_pc_
        side_hm[:,i] *= env_side_hm_
        side_pc[:,i] *= env_side_pc_
        mid_hm[:,i][int(fft_size/4):] *= np.linspace(1,0,int(fft_size/4+1))
        mid_pc[:,i][int(fft_size/4):] *= np.linspace(1,0,int(fft_size/4+1)) * 2
        side_hm[:,i][int(fft_size/4):] *= np.linspace(1,0,int(fft_size/4+1))
        side_pc[:,i][int(fft_size/4):] *= np.linspace(1,0,int(fft_size/4+1)) * 2
        mid_phs[:,i][lowpass_fft:] = np.random.uniform( -1, 1, int(fft_size/2+1)-lowpass_fft ) + 1j * np.random.uniform( -1, 1, int(fft_size/2+1)-lowpass_fft )
        side_phs[:,i][lowpass_fft:] = np.random.uniform( -1, 1, int(fft_size/2+1)-lowpass_fft ) + 1j * np.random.uniform( -1, 1, int(fft_size/2+1)-lowpass_fft )
    rebuilded_mid = (mid_hm + mid_pc) * np.exp(1.j * np.angle(mid_phs))
    rebuilded_side = (side_hm + side_pc)/2 * np.exp(1.j * np.angle(side_phs))
    rebuilded_mid[:lowpass_fft] = 0
    rebuilded_side[:lowpass_fft] = 0
    rebuilded_mid[:lowpass_fft] = 0
    rebuilded_side[:lowpass_fft] = 0
    iffted_mid = librosa.istft(rebuilded_mid, hop_length=int(fft_size/2),  window=scipy.signal.windows.cosine(fft_size))
    iffted_side = librosa.istft(rebuilded_side, hop_length=int(fft_size/2),  window=scipy.signal.windows.cosine(fft_size))
    return np.array([iffted_mid+iffted_side, iffted_mid-iffted_side]).T* 8

def get_score(original_file,補完後_file,sr, lowpass):
    original_y, original_sr = (original_file[:,0]+original_file[:,1])/2, sr
    補完後_y, 補完後_sr = (補完後_file[:,0]+補完後_file[:,1])/2, sr
    lowpass_fft = int((int(1024/2+1))*(lowpass/(sr/2)))
  # 音声の長さを確認
    if len(original_y) > len(補完後_y):
        original_y = original_y[:len(補完後_y)]
    else:
        補完後_y = 補完後_y[:len(original_y)]

  # スペクトログラムを計算
    original_stft = librosa.stft(original_y)
    補完後_stft = librosa.stft(補完後_y)
    score = 1
    for i in tqdm.tqdm(range(original_stft.shape[1])):
        or_ = original_stft[:,i]
        cv = 補完後_stft[:,i]
        score += np.corrcoef(or_.real, cv.real)[0][1]
        score /= 2
  # スペクトログラムの差を計算

    return score

def decode(dat, bit):
    bit *= -1
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2
    ffted = librosa.stft(mid, n_fft=1024)
    for i in tqdm.tqdm(range((ffted.shape[1]))):
        sample = ffted[:,i]
        mag, phs = librosa.magphase(sample)
        db = librosa.amplitude_to_db(mag)
        avg = int(sum(db) / len(db)) - 6
        #print(db)
        for j in range(len(db)):
            if db[j] < bit:
                db[j] = -120
        for j in range(len(db)):
            if db[j] == -120:
                try:
                    if db[j-1] != -120 and db[j+1] != -120:
                        db[j] = (db[j-1] + db[j+1]) / 2
                except:
                    pass
                else:
                    db[j] = random.randint(avg-15, avg-8)
        idb = librosa.db_to_amplitude(db)
        #print(db)
        idb[-400:] *= np.linspace(1,0,400)
        ffted[:,i] = idb * np.exp(1.j * np.angle(phs))
    rem_m = librosa.istft(ffted, n_fft=1024)
    ffted = librosa.stft(side, n_fft=1024)
    for i in tqdm.tqdm(range((ffted.shape[1]))):
        sample = ffted[:,i]
        mag, phs = librosa.magphase(sample)
        db = librosa.amplitude_to_db(mag)
        avg = int(sum(db) / len(db)) - 6
        #print(db)
        for j in range(len(db)):
            if db[j] < bit:
                db[j] = -90
        for j in range(len(db)):
            if db[j] == -90:
                try:
                    if db[j-1] != -90 and db[j+1] != -90:
                        db[j] = (db[j-1] + db[j+1]) / 2
                except:
                    pass
                else:
                    db[j] = random.randint(avg-15, avg-8)
        idb = librosa.db_to_amplitude(db)
        #print(db)
        idb[-400:] *= np.linspace(1,0,400)
        ffted[:,i] = idb * np.exp(1.j * np.angle(phs))
    rem_s = librosa.istft(ffted, n_fft=1024)
    #for i in range(len(hm)):
    #    try:
    #        hm[i] += rem[i]
    #    except:
    #        break
    return np.array([rem_m+rem_s, rem_m-rem_s]).T

def main():
    parser = argparse.ArgumentParser(description='HRAudioWizard Ver.0.03β rev1 (アップサンプラー)')
    parser.add_argument('--input', '-i', help='入力ファイルを指定してください')
    parser.add_argument('--output', '-o', help='出力ファイルを指定してください')
    parser.add_argument('--depth', '-d', help='出力ビット数（16, 24, 32）', default=24)
    parser.add_argument('--hfc', '-hfc', help='高音補完します', default=0)
    parser.add_argument('--remaster', '-rem', help='可聴域内のリマスターをします。', default=0)
    parser.add_argument('--bitextent', '-bt', help='ビット幅を拡張します。', default=0)
    parser.add_argument('--lowpass', '-lps', help='ローパスフィルター', default=16000)
    parser.add_argument('--scale', '-sc', help='アップサンプルスケール', default=1)
    parser.add_argument('--compressd_mode', '-cm', help='圧縮音源向けモード', default=0)
    parser.add_argument('--score', '-se', help='補完後スコアを表示', default=0)
    args = parser.parse_args()
    scale = int(args.scale)
    dat, fs = sf.read(args.input)
    if args.score == "1":
        moto_dat = dat
    if args.bitextent == "1":
        dat = decode(dat, 120)
    if args.remaster == "1":
        dat = remaster(dat, fs, scale)
    elif args.scale != "1":
        dat = remaster(dat, fs, scale)
    if args.hfc == "1":
        if args.compressd_mode == "1":
            dat = hfp(dat, int(args.lowpass), fs*scale, compressd_mode=True)
        else:
            dat = hfp(dat, int(args.lowpass), fs*scale)
    if args.depth == "16":
        sf.write(args.output, dat, fs*scale, format="WAV", subtype="PCM_16")
    if args.depth == "24":
        sf.write(args.output, dat, fs*scale, format="WAV", subtype="PCM_24")
    if args.depth == "32":
        sf.write(args.output, dat, fs*scale, format="WAV", subtype="PCM_32")
    if args.score == "1":
        score = get_score(moto_dat, dat, fs, int(args.lowpass))
        print(f"音質スコア: {score}")
    return 0

def gui():
    try:
        layout = [
                    [sg.Text("音声ファイルをアップコンバートします")],     # パート 2 - レイアウト
                    [[sg.Text("ファイル名")],
                    [sg.InputText(sys.argv[1]), sg.FileBrowse()]],
                    [[sg.Text("ビット数")],
                    [sg.Radio('24bit', 'bit')],
                    [sg.Radio('32bit', 'bit')],
                    [sg.Radio('64bit', 'bit')]],
                    [[sg.Text("サンプリングレート")],
                    [sg.Radio('x1', 'sr')],
                    [sg.Radio('x2', 'sr')],
                    [sg.Radio('x4', 'sr')],
                    [sg.Radio('x8', 'sr')]],
                    [[sg.Checkbox('リマスター')],
                    [sg.Checkbox('量子化ノイズ削減')],
                    [sg.Checkbox('高域補完')],
                    [sg.Checkbox('圧縮音源モード')]],
                    [[sg.Text("ローパスフィルタ")],
                    [sg.Spin(["14000", "15000", "16000", "17000", "18000", "20000", "6000", "7000", "8000", "9000", "10000", "11000", "12000", "13000", "30000", "50000"])]],
                    [sg.ProgressBar(100)],
                    [sg.Button('OK')],
                    [[sg.Text("再生音をアップコンバートします")],
                    [sg.Combo(look_for_audio_input(), key='dev')],
                    [sg.Combo(look_for_audio_input(), key='dev2')],
                    [sg.Button('Start')]],
                    [sg.Button('AuthInfo')]
        ]
    except:
        layout = [
                    [sg.Text("音声ファイルをアップコンバートします")],     # パート 2 - レイアウト
                    [[sg.Text("ファイル名")],
                    [sg.InputText(), sg.FileBrowse()]],
                    [[sg.Text("ビット数")],
                    [sg.Radio('24bit', 'bit')],
                    [sg.Radio('32bit', 'bit')],
                    [sg.Radio('64bit', 'bit')]],
                    [[sg.Text("サンプリングレート")],
                    [sg.Radio('x1', 'sr')],
                    [sg.Radio('x2', 'sr')],
                    [sg.Radio('x4', 'sr')],
                    [sg.Radio('x8', 'sr')]],
                    [[sg.Checkbox('リマスター')],
                    [sg.Checkbox('量子化ノイズ削減')],
                    [sg.Checkbox('高域補完')],
                    [sg.Checkbox('圧縮音源モード')]],
                    [[sg.Text("ローパスフィルタ")],
                    [sg.Spin(["14000", "15000", "16000", "17000", "18000", "20000", "6000", "7000", "8000", "9000", "10000", "11000", "12000", "13000", "30000", "50000"])]],
                    [sg.ProgressBar(100)],
                    [sg.Button('OK')],
                    [[sg.Text("再生音をアップコンバートします")],
                    [sg.Combo(look_for_audio_input(), key='dev')],
                    [sg.Combo(look_for_audio_input(), key='dev2')],
                    [sg.Button('Start')]],
                    [sg.Button('AuthInfo')]
        ]
    window = sg.Window('HRAudioWizard', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == '終了':
            break
        if event == "OK":
            scale = 0
            if len(sys.argv) > 1:
                input = sys.argv[1]
            input = values[0]
            if input == "":
                sg.popup('ファイル名を入力してください')
                continue
            input = input
            if not input.split(".")[-1] == "wav":
                sg.popup('この種類のファイルは対応していません。')
                continue
            output = "".join(input.split(".")[:-1]) + "_hraw.wav"
            lowpass = int(values[12])
            dat, fs = sf.read(input)
            if auth_info.authorized == 0:
                sg.popup("フリートライアルで体験できるのは、200000サンプルのみの変換です\n正規版を購入しましょう！")
                dat = dat[:,200000]
            if values[4] != False:
                scale = 1
            if values[5] != False:
                scale = 2
            if values[6] != False:
                scale = 4
            if values[7] != False:
                scale = 8
            if scale == 0:
                sg.popup('アップサンプルスケールを設定してください')
                continue
            if values[8] != False:
                dat = decode(dat, 120)
            if values[9] != False:
                dat = remaster(dat, fs, scale)
            else:
                dat = remaster(dat, fs, scale)
            if values[10] != False:
                if values[11] != False:
                    dat = hfp(dat, lowpass, fs*scale, compressd_mode=True)
                else:
                    dat = hfp(dat, lowpass, fs*scale, compressd_mode=False)
            if values[1] != False:
                sf.write(output, dat, fs*scale, format="WAV", subtype="PCM_24")
            elif values[2] != False:
                sf.write(output, dat, fs*scale, format="WAV", subtype="PCM_32")
            elif values[3] != False:
                sf.write(output, dat, fs*scale, format="WAV", subtype="PCM_64")
            else:
                sg.popup('ビット深度を設定してください')
        elif event == "Start":
            if auth_info.authorized == 1:
                hires_playback(values['dev'], values['dev2'])
            else:
                sg.popup("認証されていません")
        elif event == "AuthInfo":
            str = ""
            if auth_info.authorized == 1:
                str += "製品版を使用しています\n"
                str += "シリアルナンバー; " + (auth_info.licensekey) + "\n"
                str += "購入時間; " + (auth_info.date.strftime('%Y/%m/%d %H:%M:%S'))
            elif auth_info.authorized == -1:
                str += "制限の無い無料版を使用しています\n"
                str += "お買い求めは、作者のHPまで"
            elif auth_info.authorized == 0:
                str += "フリートライアル版を使用しています\n"
                str += "お買い求めは、作者のHPまで"
            sg.popup(str)
    return

if __name__ == '__main__':
    max_workers = os.cpu_count()
    auth((sys.argv[0]).replace(sys.argv[0].split("\\")[-1], "") + "license.lc_hraw")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if len(sys.argv) > 1:
            if sys.argv[1][0] == "-":
                main()
            else:
                gui()
        else:
            gui()
