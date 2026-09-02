import librosa
import numpy as np
import json

BASE = "/workspace/AudioEditingCode/pc_extractions_masked/audioldm2-music/MDDBGospel_16sec/pmt_A_recording_of_ryhtmic_clapping,_a_women_singing,_and_drums_and_guitar_playing.__neg__"
files = {
    "orig": f"{BASE}/orig.wav",
    "recon_noPC": f"{BASE}/sNone_p128-256_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788369572.wav",
    "a2": f"{BASE}/sNone_p128-256_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788369572_driftgens/pc3_drift150--1_spts120_it50_shiftednpTrue_a2.0.wav",
    "a-2": f"{BASE}/sNone_p128-256_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788369572_driftgens/pc3_drift150--1_spts120_it50_shiftednpTrue_a-2.0.wav",
}

sr_target = 16000
results = {}

# Also compute metrics restricted to the masked window (5-10s) specifically,
# since the edit was only meant to act there.
mask_start_sec, mask_end_sec = 5.0, 10.0

for name, path in files.items():
    y, sr = librosa.load(path, sr=sr_target, mono=True)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr,
        frame_length=1024, hop_length=256
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    frac_voiced = float(np.mean(voiced_flag)) if len(voiced_flag) else 0.0

    frame_rate = sr / 256
    mask_frame_start = int(mask_start_sec * frame_rate)
    mask_frame_end = int(mask_end_sec * frame_rate)
    f0_masked = f0[mask_frame_start:mask_frame_end]
    voiced_masked = voiced_flag[mask_frame_start:mask_frame_end]
    voiced_f0_masked = f0_masked[voiced_masked & ~np.isnan(f0_masked)]

    vibrato_rms = None
    if len(voiced_f0) > 20:
        idx = np.arange(len(f0))
        good = ~np.isnan(f0)
        if good.sum() > 20:
            f0_interp = np.interp(idx, idx[good], f0[good])
            f0_cents = 1200 * np.log2(f0_interp / np.nanmedian(f0_interp[good]))
            win = max(3, int(frame_rate / 2))
            kernel = np.ones(win) / win
            slow = np.convolve(f0_cents, kernel, mode='same')
            fast = f0_cents - slow
            vibrato_rms = float(np.sqrt(np.mean(fast[good] ** 2)))

    y_harm, y_perc = librosa.effects.hpss(y)
    rms_harm = float(np.sqrt(np.mean(y_harm ** 2)))
    rms_perc = float(np.sqrt(np.mean(y_perc ** 2)))
    rms_total = float(np.sqrt(np.mean(y ** 2)))

    # masked-window-only percussive fraction (samples, not frames)
    s0, s1 = int(mask_start_sec * sr), int(mask_end_sec * sr)
    y_perc_masked = y_perc[s0:s1]
    y_harm_masked = y_harm[s0:s1]
    rms_perc_masked = float(np.sqrt(np.mean(y_perc_masked ** 2))) if len(y_perc_masked) else None
    rms_harm_masked = float(np.sqrt(np.mean(y_harm_masked ** 2))) if len(y_harm_masked) else None
    rms_total_masked = float(np.sqrt(np.mean(y[s0:s1] ** 2))) if s1 > s0 else None

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs_stft < 150
    vocal_mask = (freqs_stft >= 300) & (freqs_stft <= 3400)
    low_energy_frac = float(S[low_mask].sum() / S.sum())
    vocal_band_energy_frac = float(S[vocal_mask].sum() / S.sum())

    results[name] = {
        "frac_voiced_full": frac_voiced,
        "median_f0_hz_full": float(np.nanmedian(voiced_f0)) if len(voiced_f0) else None,
        "median_f0_hz_masked_5to10s": float(np.nanmedian(voiced_f0_masked)) if len(voiced_f0_masked) else None,
        "vibrato_rms_cents_full": vibrato_rms,
        "rms_harmonic_full": rms_harm,
        "rms_percussive_full": rms_perc,
        "percussive_frac_of_total_full": rms_perc / rms_total if rms_total > 0 else None,
        "percussive_frac_of_total_masked_5to10s": rms_perc_masked / rms_total_masked if rms_total_masked else None,
        "low_freq(<150Hz)_energy_frac": low_energy_frac,
        "vocal_band(300-3400Hz)_energy_frac": vocal_band_energy_frac,
    }

print(json.dumps(results, indent=2))

y2, sr = librosa.load(files["a2"], sr=sr_target, mono=True)
ym2, sr = librosa.load(files["a-2"], sr=sr_target, mono=True)
n = min(len(y2), len(ym2))
S2 = np.abs(librosa.stft(y2[:n], n_fft=2048, hop_length=512))
Sm2 = np.abs(librosa.stft(ym2[:n], n_fft=2048, hop_length=512))
diff = np.mean(np.abs(S2 - Sm2), axis=1)
freqs_stft = librosa.fft_frequencies(sr=sr_target, n_fft=2048)
top_bins = np.argsort(diff)[::-1][:15]
print("\nTop 15 frequency bins by |a2 - a-2| spectral difference (full signal):")
for b in top_bins:
    print(f"  {freqs_stft[b]:7.1f} Hz : {diff[b]:.4f}")

# Same but restricted to the masked 5-10s window in time
hop = 512
t0, t1 = int(mask_start_sec * sr / hop), int(mask_end_sec * sr / hop)
S2m = np.abs(librosa.stft(y2[:n], n_fft=2048, hop_length=hop))[:, t0:t1]
Sm2m = np.abs(librosa.stft(ym2[:n], n_fft=2048, hop_length=hop))[:, t0:t1]
diff_masked = np.mean(np.abs(S2m - Sm2m), axis=1)
top_bins_masked = np.argsort(diff_masked)[::-1][:15]
print("\nTop 15 frequency bins by |a2 - a-2| spectral difference (masked window 5-10s ONLY):")
for b in top_bins_masked:
    print(f"  {freqs_stft[b]:7.1f} Hz : {diff_masked[b]:.4f}")

low_mask = freqs_stft < 150
vocal_mask = (freqs_stft >= 300) & (freqs_stft <= 3400)
print(f"\n[full signal] diff energy <150Hz: {diff[low_mask].sum():.4f} | vocal(300-3400Hz): {diff[vocal_mask].sum():.4f} | total: {diff.sum():.4f}")
print(f"[masked 5-10s] diff energy <150Hz: {diff_masked[low_mask].sum():.4f} | vocal(300-3400Hz): {diff_masked[vocal_mask].sum():.4f} | total: {diff_masked.sum():.4f}")
