import librosa
import numpy as np
import sys
import json

BASE = "/workspace/AudioEditingCode/pc_extractions/audioldm2-music/MDDBGospel_8secs/pmt_A_recording_of_ryhtmic_clapping,_a_women_singing,_and_drums_and_guitar_playing.__neg__"
files = {
    "orig": f"{BASE}/orig.wav",
    "recon_noPC": f"{BASE}/sNone_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788367761.wav",
    "a2": f"{BASE}/sNone_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788367761_driftgens/pc3_drift150--1_spts120_it50_shiftednpTrue_a2.0.wav",
    "a-2": f"{BASE}/sNone_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788367761_driftgens/pc3_drift150--1_spts120_it50_shiftednpTrue_a-2.0.wav",
}

sr_target = 16000
results = {}

for name, path in files.items():
    y, sr = librosa.load(path, sr=sr_target, mono=True)

    # F0 tracking (pYIN), typical vocal range
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr,
        frame_length=1024, hop_length=256
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    frac_voiced = float(np.mean(voiced_flag)) if len(voiced_flag) else 0.0

    # Vibrato: detrend f0 (remove slow note-level motion), measure oscillation in 4-8Hz band
    vibrato_rms = None
    vibrato_rate_hz = None
    if len(voiced_f0) > 20:
        f0_full = f0.copy()
        # interpolate nan for continuous analysis where voiced
        idx = np.arange(len(f0_full))
        good = ~np.isnan(f0_full)
        if good.sum() > 20:
            f0_interp = np.interp(idx, idx[good], f0_full[good])
            f0_cents = 1200 * np.log2(f0_interp / np.nanmedian(f0_interp[good]))
            frame_rate = sr / 256  # hop_length
            # remove slow trend with a simple high-pass via diff of moving average
            win = max(3, int(frame_rate / 2))  # ~0.5s smoothing window
            kernel = np.ones(win) / win
            slow = np.convolve(f0_cents, kernel, mode='same')
            fast = f0_cents - slow
            fast_voiced = fast[good]
            vibrato_rms = float(np.sqrt(np.mean(fast_voiced ** 2)))
            # dominant oscillation rate via FFT of fast component (voiced region only, contiguous approx)
            spec = np.abs(np.fft.rfft(fast_voiced - fast_voiced.mean()))
            freqs = np.fft.rfftfreq(len(fast_voiced), d=1.0 / frame_rate)
            band = (freqs >= 3) & (freqs <= 9)
            if band.any() and spec[band].sum() > 0:
                vibrato_rate_hz = float(freqs[band][np.argmax(spec[band])])

    # Harmonic-percussive split energy
    y_harm, y_perc = librosa.effects.hpss(y)
    rms_harm = float(np.sqrt(np.mean(y_harm ** 2)))
    rms_perc = float(np.sqrt(np.mean(y_perc ** 2)))
    rms_total = float(np.sqrt(np.mean(y ** 2)))

    # low-frequency energy fraction (<150Hz, rhythmic/drum/bass region)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs_stft < 150
    vocal_mask = (freqs_stft >= 300) & (freqs_stft <= 3400)
    low_energy_frac = float(S[low_mask].sum() / S.sum())
    vocal_band_energy_frac = float(S[vocal_mask].sum() / S.sum())

    results[name] = {
        "frac_voiced": frac_voiced,
        "median_f0_hz": float(np.nanmedian(voiced_f0)) if len(voiced_f0) else None,
        "vibrato_rms_cents": vibrato_rms,
        "vibrato_rate_hz": vibrato_rate_hz,
        "rms_harmonic": rms_harm,
        "rms_percussive": rms_perc,
        "rms_total": rms_total,
        "percussive_frac_of_total": rms_perc / rms_total if rms_total > 0 else None,
        "low_freq(<150Hz)_energy_frac": low_energy_frac,
        "vocal_band(300-3400Hz)_energy_frac": vocal_band_energy_frac,
    }

print(json.dumps(results, indent=2))

# Direct spectral diff between a2 and a-2 to see where the biggest change concentrates
y2, sr = librosa.load(files["a2"], sr=sr_target, mono=True)
ym2, sr = librosa.load(files["a-2"], sr=sr_target, mono=True)
n = min(len(y2), len(ym2))
S2 = np.abs(librosa.stft(y2[:n], n_fft=2048, hop_length=512))
Sm2 = np.abs(librosa.stft(ym2[:n], n_fft=2048, hop_length=512))
diff = np.mean(np.abs(S2 - Sm2), axis=1)  # avg abs diff per freq bin
freqs_stft = librosa.fft_frequencies(sr=sr_target, n_fft=2048)
top_bins = np.argsort(diff)[::-1][:15]
print("\nTop 15 frequency bins by |a2 - a-2| spectral difference (Hz : avg magnitude diff):")
for b in sorted(top_bins):
    pass
for b in top_bins:
    print(f"  {freqs_stft[b]:7.1f} Hz : {diff[b]:.4f}")

# energy of the diff split by band
low_mask = freqs_stft < 150
vocal_mask = (freqs_stft >= 300) & (freqs_stft <= 3400)
mid_high_mask = freqs_stft > 3400
print(f"\ndiff energy <150Hz: {diff[low_mask].sum():.4f}")
print(f"diff energy 300-3400Hz (vocal): {diff[vocal_mask].sum():.4f}")
print(f"diff energy >3400Hz: {diff[mid_high_mask].sum():.4f}")
print(f"total diff energy: {diff.sum():.4f}")
