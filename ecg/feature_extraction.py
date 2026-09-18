import os
import numpy as np
import scipy.io
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

FEATURE_NAMES = [
    # --- Basic Amplitude & Statistical Features (16) ---
    "mean",                   # Signal mean amplitude
    "std",                    # Standard deviation of signal
    "min",                    # Minimum signal amplitude
    "max",                    # Maximum signal amplitude
    "range",                  # Peak-to-peak amplitude range (max - min)
    "median",                 # Signal median amplitude
    "rms",                    # Root Mean Square of signal
    "energy",                 # Total signal energy (sum of squared amplitudes)
    "p05",                    # 5th percentile amplitude
    "p25",                    # 25th percentile amplitude (Q1)
    "p75",                    # 75th percentile amplitude (Q3)
    "p95",                    # 95th percentile amplitude
    "iqr",                    # Interquartile range (p75 - p25)
    "skewness",               # Signal skewness (asymmetry measure)
    "kurtosis",               # Signal kurtosis (tailedness measure)
    "mad",                    # Mean Absolute Deviation
    
    # --- Time-Domain Variability & Derivative Features (5) ---
    "zero_crossings",         # Number of zero crossings
    "first_diff_mean",        # Mean of absolute 1st derivative (waveform velocity)
    "first_diff_std",         # Standard deviation of 1st derivative
    "second_diff_mean",       # Mean of absolute 2nd derivative (acceleration)
    "second_diff_std",        # Standard deviation of 2nd derivative
    
    # --- Peak & HRV (Heart Rate Variability) Features (11) ---
    "num_peaks",              # Count of detected R-peaks / prominent peaks
    "mean_peak_prominence",   # Mean prominence of detected peaks
    "std_peak_prominence",    # Standard deviation of peak prominence
    "mean_rr_interval",       # Mean distance between consecutive peaks (samples)
    "std_rr_interval",        # Standard deviation of RR-intervals (SDNN equivalent)
    "min_rr_interval",        # Minimum RR-interval duration
    "max_rr_interval",        # Maximum RR-interval duration
    "rr_range",               # Range of RR intervals (max_rr - min_rr)
    "rr_ratio",               # Ratio of max_rr to min_rr
    "rmssd",                  # Root Mean Square of Successive RR Differences
    "pnn50",                  # Percentage of successive RR diffs > 50 samples (~138ms)
    
    # --- Frequency-Domain (FFT & Power Spectral Density) Features (7) ---
    "spectral_centroid",      # Center of mass of power spectrum
    "spectral_entropy",       # Spectral entropy (frequency dispersion measure)
    "dominant_freq",          # Frequency with maximum spectral power
    "low_freq_power",         # Spectral power in Low Frequency band (0.04 - 0.15 Hz)
    "high_freq_power",        # Spectral power in High Frequency band (0.15 - 0.40 Hz)
    "lf_hf_ratio",            # Ratio of Low-Frequency to High-Frequency power
    "total_spectral_power",   # Total spectral power across all frequencies
    
    # --- Signal Complexity & Autocorrelation Features (4) ---
    "amplitude_entropy",      # Shannon entropy of binned amplitude distribution
    "autocorr_lag1",          # Autocorrelation at lag 1 sample
    "autocorr_lag10",         # Autocorrelation at lag 10 samples
    "autocorr_lag50"          # Autocorrelation at lag 50 samples (~138ms)
]

def extract_features_from_signal(signal, fs=360.0):
    """
    Extracts 43 comprehensive ECG features from a 1D z-score normalized array of 3,600 samples.
    """
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    val_range = max_val - min_val
    median_val = float(np.median(signal))
    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    energy_val = float(np.sum(signal ** 2))

    p05 = float(np.percentile(signal, 5))
    p25 = float(np.percentile(signal, 25))
    p75 = float(np.percentile(signal, 75))
    p95 = float(np.percentile(signal, 95))
    iqr_val = p75 - p25

    skew_val = float(skew(signal))
    kurt_val = float(kurtosis(signal))
    mad_val = float(np.mean(np.abs(signal - mean_val)))
    zero_crossings = float(np.sum(np.diff(signal > 0) != 0))

    # 1st and 2nd derivatives
    diff1 = np.diff(signal)
    diff2 = np.diff(signal, n=2)
    diff1_mean = float(np.mean(np.abs(diff1)))
    diff1_std = float(np.std(diff1))
    diff2_mean = float(np.mean(np.abs(diff2)))
    diff2_std = float(np.std(diff2))

    # Peak detection & HRV metrics
    peaks, properties = find_peaks(signal, distance=100, prominence=0.5)
    num_peaks = len(peaks)

    if num_peaks > 0:
        prominences = properties.get("prominences", np.array([0.0]))
        mean_prom = float(np.mean(prominences))
        std_prom = float(np.std(prominences)) if len(prominences) > 1 else 0.0
    else:
        mean_prom = 0.0
        std_prom = 0.0

    if num_peaks > 1:
        rr_intervals = np.diff(peaks).astype(np.float32)
        mean_rr = float(np.mean(rr_intervals))
        std_rr = float(np.std(rr_intervals))
        min_rr = float(np.min(rr_intervals))
        max_rr = float(np.max(rr_intervals))
        rr_range = max_rr - min_rr
        rr_ratio = max_rr / (min_rr + 1e-8)

        if len(rr_intervals) > 1:
            rr_diffs = np.diff(rr_intervals)
            rmssd = float(np.sqrt(np.mean(rr_diffs ** 2)))
            # pNN50: percentage of successive RR interval differences > 50 samples (~138ms)
            pnn50 = float(100.0 * np.sum(np.abs(rr_diffs) > 50.0) / len(rr_diffs))
        else:
            rmssd = 0.0
            pnn50 = 0.0
    else:
        mean_rr = 0.0
        std_rr = 0.0
        min_rr = 0.0
        max_rr = 0.0
        rr_range = 0.0
        rr_ratio = 1.0
        rmssd = 0.0
        pnn50 = 0.0

    # Frequency-Domain Features via FFT
    rfft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    psd = rfft_vals ** 2
    psd_sum = np.sum(psd) + 1e-12
    psd_norm = psd / psd_sum

    spectral_centroid = float(np.sum(freqs * psd_norm))
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    dominant_freq = float(freqs[np.argmax(psd)])

    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    lf_power = float(np.sum(psd[lf_mask]))
    hf_power = float(np.sum(psd[hf_mask]))
    lf_hf_ratio = float(lf_power / (hf_power + 1e-8))
    total_spectral_power = float(psd_sum)

    # Amplitude Entropy & Autocorrelation
    hist, _ = np.histogram(signal, bins=20, density=True)
    hist = hist[hist > 0]
    amplitude_entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

    def autocorr(lag):
        if lag >= len(signal):
            return 0.0
        c = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    ac_lag1 = autocorr(1)
    ac_lag10 = autocorr(10)
    ac_lag50 = autocorr(50)

    features = [
        mean_val, std_val, min_val, max_val, val_range, median_val,
        rms_val, energy_val, p05, p25, p75, p95, iqr_val,
        skew_val, kurt_val, mad_val,
        zero_crossings, diff1_mean, diff1_std, diff2_mean, diff2_std,
        float(num_peaks), mean_prom, std_prom,
        mean_rr, std_rr, min_rr, max_rr, rr_range, rr_ratio, rmssd, pnn50,
        spectral_centroid, spectral_entropy, dominant_freq,
        lf_power, hf_power, lf_hf_ratio, total_spectral_power,
        amplitude_entropy, ac_lag1, ac_lag10, ac_lag50
    ]

    return np.array(features, dtype=np.float32)

def extract_dataset_features():
    input_path = os.path.join("processed_data", "ecg_dataset.npz")
    output_path = os.path.join("processed_data", "features_dataset.npz")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed dataset not found at '{input_path}'. Run preprocess.py first.")

    print("=" * 70)
    print("           ECG EXTENDED FEATURE EXTRACTION (43 FEATURES)           ")
    print("=" * 70)

    data = np.load(input_path)
    X_train_raw = data['X_train']
    X_val_raw = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    class_names = data['class_names']

    print(f"\nLoaded preprocessed signals from '{input_path}':")
    print(f" - Raw X_train shape: {X_train_raw.shape}")
    print(f" - Raw X_val shape  : {X_val_raw.shape}")

    print(f"\nExtracting {len(FEATURE_NAMES)} extended features per signal...")

    X_train_feats = np.zeros((X_train_raw.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(X_train_raw.shape[0]):
        X_train_feats[i] = extract_features_from_signal(X_train_raw[i])

    X_val_feats = np.zeros((X_val_raw.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(X_val_raw.shape[0]):
        X_val_feats[i] = extract_features_from_signal(X_val_raw[i])

    print("\n--- EXTENDED FEATURE MATRIX SUMMARY ---")
    print(f"Total Features Extracted    : {len(FEATURE_NAMES)}")
    print(f"Training Feature Matrix Shape: {X_train_feats.shape}")
    print(f"Validation Feature Matrix    : {X_val_feats.shape}")

    np.savez_compressed(
        output_path,
        X_train=X_train_feats,
        X_val=X_val_feats,
        y_train=y_train,
        y_val=y_val,
        class_names=class_names,
        feature_names=np.array(FEATURE_NAMES)
    )

    print(f"\nSaved extended features successfully to '{output_path}'")
    print("=" * 70)
    return FEATURE_NAMES, X_train_feats.shape, X_val_feats.shape

if __name__ == "__main__":
    extract_dataset_features()
