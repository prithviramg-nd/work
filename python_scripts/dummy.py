import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Synthetic input
time_points = np.arange(0, 25)
signal_slow = 1000 + np.random.normal(0, 7, 25)
bend_duration = 6
bend_amplitude = 50
signal_slow[5:5+bend_duration] += np.linspace(0, bend_amplitude, bend_duration)
signal_slow[5+bend_duration:] += bend_amplitude
signal_slow = [806.0, 789.0, 806.0, 822.0, 812.0, 802.0, 816.0, 805.0, 800.0, 805.0, 838.0, 851.0, 837.0, 830.0, 852.0, 871.0, 834.0, 800.0, 835.0, 845.0, 819.0, 785.0, 791.0, 827.0, 833.0, 805.0, 782.0, 815.0, 842.0, 840.0, 838.0, 820.0]

def moving_median(signal: np.ndarray, kernel: int = 3) -> np.ndarray:
    """
    Simple moving median filter.
    Args:
        signal (np.ndarray): 1D numpy array
        kernel (int, optional): odd integer size of the median filter. Defaults to 3.
    Returns:
        np.ndarray: filtered signal of same length
    """    
    assert kernel % 2 == 1
    h = kernel // 2
    padded = np.pad(signal, (h, h), mode='edge')
    out = np.empty_like(signal)
    for i in range(len(signal)):
        out[i] = np.median(padded[i:i+kernel])
    return out

def moving_mean(signal: np.ndarray, kernel: int = 5) -> np.ndarray:
    """
    Simple moving mean filter.
    Args:
        signal (np.ndarray): 1D numpy array
        kernel (int, optional): size of the mean filter. Defaults to 5.
    Returns:
        np.ndarray: filtered signal of same length
    """
    if kernel <= 1:
        return signal.copy()
    kernel = int(kernel)
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    out = (cumsum[kernel:] - cumsum[:-kernel]) / float(kernel)
    # pad start/end to keep same length (replicate edges)
    left = np.full((kernel//2,), out[0])
    right = np.full((len(signal) - len(out) - left.size,), out[-1]) if len(signal) - len(out) - left.size > 0 else np.array([])
    return np.concatenate([left, out, right])

def haar_decompose(signal: np.ndarray, levels: int) -> tuple:
    """
    Perform Haar wavelet decomposition.
    Args:
        signal (np.ndarray): 1D numpy array
        levels (int): number of decomposition levels
    Returns:
        tuple: (cA, d_list) where
            cA (np.ndarray): approximation coefficients at level 'levels'
            d_list (list): detail coefficients [D_level, D_level-1, ..., D1]
    """
    y = signal.copy().astype(np.float64)
    d_list = []
    for lev in range(levels):
        if y.size % 2 == 1:  # handle odd length by padding last element
            y = np.append(y, y[-1])
        a = (y[0::2] + y[1::2]) / 2.0
        d = (y[0::2] - y[1::2]) / 2.0
        d_list.append(d)
        y = a  # next round on approximation
    cA = y  # final approx at coarsest level
    d_list = d_list[::-1] # reverse to [D_level, D_level-1, ..., D1]
    return cA, d_list

def haar_reconstruct(cA: np.ndarray, d_list: list) -> np.ndarray:
    """
    Performs Haar wavelet reconstruction.
    Args:
        cA (np.ndarray): Approximation coefficients
        d_list (list): Detail coefficients [D_level, D_level-1, ..., D1]
    Returns:
        np.ndarray: Reconstructed signal
    """    
    a = cA.copy()
    levels = len(d_list)
    for i in range(levels):
        d = d_list[i]
        # lengths must match: len(a) == len(d)
        # If d is shorter due to padding during decomposition, we handle it by repeating last element
        m = len(d)
        if len(a) != m:
            # if lengths mismatch, pad a or d appropriately (this shouldn't normally happen
            # if decomposition followed same rules). We pad by repeating last values.
            if len(a) < m:
                a = np.pad(a, (0, m - len(a)), 'edge')
            else:
                d = np.pad(d, (0, len(a) - m), 'edge')
            m = len(d)
        # produce s of length 2*m
        s = np.empty(2*m, dtype=np.float64)
        s[0::2] = a + d
        s[1::2] = a - d
        a = s  # becomes next-level approximation
    return a[:]

def reconstruct_detail_full(signal: np.ndarray, levels: int, detail_level_index: int) -> np.ndarray:
    """
    Reconstruct a specific detail level from the original signal.
    Args:
        signal (np.ndarray): Original input signal
        levels (int): Number of decomposition levels used
        detail_level_index (int): Which detail to reconstruct (1..levels), 1=fastest (D1), levels=coarsest (D_level)
    Returns:
        np.ndarray: Reconstructed detail signal
    """
    cA, d_list = haar_decompose(signal, levels)
    # d_list is [D_level, D_level-1, ..., D1]
    # we want to set zeros for all details except the requested one.
    zeroed_d_list = []
    for i, d in enumerate(d_list):
        # i runs 0..levels-1 where 0->D_level, last->D1
        level_num = levels - i  # mapping to intuitive level number
        if level_num == detail_level_index:
            zeroed_d_list.append(d.copy())
        else:
            zeroed_d_list.append(np.zeros_like(d))
    # set coarse approx to zeros so only detail contribution remains
    zero_cA = np.zeros_like(cA)
    recon = haar_reconstruct(zero_cA, zeroed_d_list)
    # If recon length > original (due to padding), trim.
    return recon[:len(signal)]

def first_consecutive_run(indices: np.ndarray, min_run: int) -> int | None:
    """
    Find the first index in 'indices' where at least 'min_run' consecutive integers occur
    Args:
        indices (np.ndarray): 1D array of integer indices
        min_run (int): minimum consecutive run length
    Returns:
        int or None: starting index of first run, or None if not found
    """
    if indices.size == 0:
        return None
    start = indices[0]
    run_len = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            run_len += 1
        else:
            # reset
            if run_len >= min_run:
                return indices[i-run_len]
            run_len = 1
    return indices[len(indices)-run_len] if run_len >= min_run else None

def detect_bend(signal: np.ndarray,
                levels: int = 3,
                median_k: int = 3,
                mean_k: int = 9,
                weight_d2: float = 0.7,
                mad_k: float = 6.0,
                min_run: int = 2) -> dict:
    """
    Detect bend start in a 1D signal using wavelet-based method.
    Args:
        signal (np.ndarray): 1D numpy array of the signal
        levels (int, optional): Number of Haar wavelet decomposition levels. Defaults to 3
        median_k (int, optional): Kernel size for moving median filter. Defaults to 3.
        mean_k (int, optional): Kernel size for moving mean filter. Defaults to 9
        weight_d2 (float, optional): Weight for D2 detail in energy computation. Defaults to 0.7.
        mad_k (float, optional): Multiplier for MAD thresholding. Defaults to 6
        min_run (int, optional): Minimum consecutive frames above threshold to confirm detection. Defaults to 2.
    Returns:
        dict: {
            'smoothed': np.ndarray,
            'd1_full': np.ndarray,
            'd2_full': np.ndarray,
            'energy': np.ndarray,
            'threshold': float,
            'median': float,
            'mad': float,
            'candidates': np.ndarray,
            'detected_start': int or None
        }
    """
    # small median then moving mean smoothing
    s_med = moving_median(signal, kernel=median_k)
    s_smooth = moving_mean(s_med, kernel=mean_k)
    
    # wavelet details reconstructed to full length
    d1_full = np.abs(reconstruct_detail_full(s_smooth, levels=levels, detail_level_index=1))
    d2_full = np.abs(reconstruct_detail_full(s_smooth, levels=levels, detail_level_index=2)) if levels >= 2 else np.zeros_like(d1_full)

    # d1_full = d1_full[d1_full > 0]  # focus on positive changes (downward bends)
    # d2_full = d2_full[d2_full > 0]  # focus on positive changes (downward bends)

    # energy = abs details combined
    energy = d1_full + weight_d2 * d2_full

    # robust threshold via MAD
    med = np.median(energy)
    mad = np.median(np.abs(energy - med))
    thr = med + mad_k * mad
    
    # candidate indices above threshold
    cand = np.where(energy > thr)[0]
    
    start_frame = first_consecutive_run(cand, min_run) 
    
    return {
        'smoothed': s_smooth,
        'd1_full': d1_full,
        'd2_full': d2_full,
        'energy': energy,
        'threshold': thr,
        'median': med,
        'mad': mad,
        'candidates': cand,
        'detected_start': start_frame
    }

result = detect_bend(signal_slow,
                     levels=3,
                     median_k=3,
                     mean_k=9,      # wider mean for more smoothing on noisy cameras
                     weight_d2=0.7,
                     mad_k=9.0,     # increase to reduce false positives under high noise
                     min_run=2)     # why 2, because head bends last multiple frames, so should be safe

print("Detected start frame:", result['detected_start'])
print("Candidate indices (first 20):", result['candidates'][:20])
print("Threshold:", result['threshold'], "median:", result['median'], "MAD:", result['mad'])

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.title("Original and Smoothed Signal")
plt.plot(signal_slow, label='Original (noisy)')
plt.plot(result['smoothed'], label='Smoothed')
if result['detected_start'] is not None:
    plt.axvline(result['detected_start'], color='k', linestyle='--', label=f'Detected start @ {result['detected_start']}')
plt.legend()
plt.ylabel('Y position')

plt.subplot(3,1,2)
plt.title("Reconstructed Wavelet Detail Components (full length)")
plt.plot(result['d1_full'], label='|D1|')
plt.plot(result['d2_full'], label='|D2|', alpha=0.7)
plt.legend()
plt.ylabel('Detail magnitude')

plt.subplot(3,1,3)
plt.title("Combined Wavelet Energy + Threshold")
plt.plot(result['energy'], label='combined energy')
plt.axhline(result['threshold'], linestyle='--', label='threshold')
if result['detected_start'] is not None:
    plt.axvline(result['detected_start'], color='k', linestyle='--')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Energy')
plt.tight_layout()

if result['detected_start'] is not None:
    idx = result['detected_start']
    left = max(0, idx-3)
    right = min(len(signal_slow)-1, idx+3)
    slope = (result['smoothed'][right] - result['smoothed'][left]) / (right-left+1e-12) # approximate local slope
    print("Local slope at detection (approx):", slope)
else:
    print("No bend detected.")

plt.savefig('head_bend_detection_improved.png')

'''
import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt

# ---------- Synthetic signal (your modified noisy version) ----------
time_points = np.arange(0, 60)
signal_slow = 1000 + np.random.normal(0, 7, 60)  # high noise
bend_duration = 6
bend_amplitude = 30
signal_slow[20:20+bend_duration] += np.linspace(0, bend_amplitude, bend_duration)
signal_slow[20+bend_duration] += bend_amplitude

# ---------- Preprocessing: strong smoothing + optional median filter ----------
y = signal_slow.copy()
# median filter to remove spikes, then Savitzky-Golay for smooth derivative preservation
y_med = medfilt(y, kernel_size=3)                 # removes occasional spikes
y_smooth = savgol_filter(y_med, window_length=11, polyorder=2)  # stronger smoothing

# ---------- Wavelet decomposition ----------
wavelet = 'haar'
level = 3
coeffs = pywt.wavedec(y_smooth, wavelet, level=level)  
# coeffs layout: [cA_level, cD_level, ..., cD2, cD1]
# We'll reconstruct detail components at full signal length so indices map to frames
def reconstruct_detail(coeffs, detail_level_index):
    """Reconstruct a single detail component as a full-length signal.
    detail_level_index: 1 -> D1 (fastest), 2 -> D2, ... up to level
    """
    # Create zeroed list same shape as coeffs
    coeffs_zeroed = [np.zeros_like(c) for c in coeffs]
    # coeffs structure: cA, cD_level, ..., cD2, cD1
    # index of cD1 is -1, cD2 is -2, ... cDk is -(k)
    coeffs_zeroed[-detail_level_index] = coeffs[-detail_level_index].copy()
    # Reconstruct
    recon = pywt.waverec(coeffs_zeroed, wavelet)
    # waverec may produce length slightly different due to padding; trim/pad to original length
    recon = recon[:len(y_smooth)]
    if recon.shape[0] < len(y_smooth):
        recon = np.pad(recon, (0, len(y_smooth) - recon.shape[0]), 'constant')
    return recon

# Reconstruct D1 and D2 (D1 = fastest)
d1 = reconstruct_detail(coeffs, detail_level_index=1)
d2 = reconstruct_detail(coeffs, detail_level_index=2)

# Use absolute energy (magnitude) of details
energy_d1 = np.abs(d1)
energy_d2 = np.abs(d2)

# Combine energies (helps detect a slightly slower bend that shows across scales)
combined_energy = energy_d1 + 0.7 * energy_d2  # weight can be tuned

# ---------- Robust threshold using Median Absolute Deviation ----------
def mad_threshold(signal, k=6.0):
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    # convert MAD to an approximate std if desired: sigma ≈ 1.4826*MAD (for Gaussian)
    # But we can simply use k*MAD as robust threshold
    thr = med + k * mad
    return thr, med, mad

# Tuning note: with higher noise increase k (e.g., 6 or 8). Lower noise use 3.
k = 10.0
thr, med_val, mad_val = mad_threshold(combined_energy, k=k)

# Find candidate frames where energy exceeds threshold
candidates = np.where(combined_energy > thr)[0]

# Post-filter candidates: require contiguous run to avoid single-frame spikes
def find_burst_start(idx_array, min_run=2):
    """Return earliest index where at least min_run consecutive frames are above threshold"""
    if len(idx_array) == 0:
        return None
    # group consecutive indices
    runs = []
    start = idx_array[0]
    prev = idx_array[0]
    run = [start]
    for i in idx_array[1:]:
        if i == prev + 1:
            run.append(i)
        else:
            runs.append(run)
            run = [i]
        prev = i
    runs.append(run)
    # find first run long enough
    for r in runs:
        if len(r) >= min_run:
            return r[0]
    # fallback: return first index
    return idx_array[0]

bend_start_frame = find_burst_start(candidates, min_run=2)

print("Detected candidate indices (first 20):", candidates[:20])
print("Mapped bend start frame (original timeline):", bend_start_frame)
print("Threshold (combined energy):", thr, "median:", med_val, "MAD:", mad_val)

# ---------- Simple sanity check: also check slope around detected frame ----------
def get_local_slope(signal, idx, window=3):
    # central difference
    left = max(0, idx - window)
    right = min(len(signal)-1, idx + window)
    slope = (signal[right] - signal[left]) / (right - left + 1e-12)
    return slope

if bend_start_frame is not None:
    local_slope = get_local_slope(y_smooth, bend_start_frame, window=3)
    print("Local slope at detected frame:", local_slope)

# ---------- Plotting ----------
plt.figure(figsize=(10,8))

plt.subplot(3,1,1)
plt.title("Original and Smoothed Signal")
plt.plot(y, label='Original (noisy)')
plt.plot(y_smooth, label='Smoothed')
if bend_start_frame is not None:
    plt.axvline(bend_start_frame, color='k', linestyle='--', label=f'Detected start @ {bend_start_frame}')
plt.legend()
plt.ylabel('Y position')

plt.subplot(3,1,2)
plt.title("Reconstructed Wavelet Detail Components (full length)")
plt.plot(energy_d1, label='|D1|')
plt.plot(energy_d2, label='|D2|', alpha=0.7)
plt.legend()
plt.ylabel('Detail magnitude')

plt.subplot(3,1,3)
plt.title("Combined Wavelet Energy + Threshold")
plt.plot(combined_energy, label='combined energy')
plt.axhline(thr, linestyle='--', label='threshold')
if bend_start_frame is not None:
    plt.axvline(bend_start_frame, color='k', linestyle='--')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Energy')

plt.tight_layout()
plt.savefig('head_bend_detection_improved.png')
'''