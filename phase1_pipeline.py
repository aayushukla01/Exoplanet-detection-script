import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time as t_module
from scipy.stats import binned_statistic
from astropy.timeseries import BoxLeastSquares

# ─────────────────────────────────────────────
# STEP 1: SEARCH & DOWNLOAD (FAST PARALLEL VERSION)
# ─────────────────────────────────────────────
print("📡 Step 1: Searching MAST archive for TESS light curves...")
target_tic = "TIC 261136679"
raw_numeric_id = "".join(filter(str.isdigit, target_tic))
formatted_tic = f"TIC {raw_numeric_id}"
print(f"🔍 Querying MAST for: {formatted_tic}")

# Filter for standard 2-minute cadence (exptime=120) to avoid buggy 20-second files
search_result = lk.search_lightcurve(formatted_tic, author="SPOC", exptime=120)
if len(search_result) == 0:
    print("🔄 SPOC 2-min empty. Broadening search...")
    search_result = lk.search_lightcurve(formatted_tic, exptime=120)
if len(search_result) == 0:
    raise RuntimeError(f"No 2-minute light curve products found for {formatted_tic}.")

print(f"📥 Found {len(search_result)} products. Downloading in parallel...")
cache_dir = os.path.abspath("./lk_cache")

# We can safely use download_all because we filtered out the buggy 20-second files!
lc_collection = search_result.download_all(download_dir=cache_dir)
if lc_collection is None or len(lc_collection) == 0:
    raise RuntimeError("Failed to download light curves.")

sector_lcs = []
for lc in lc_collection:
    try:
        lc_n = lc.normalize()
        lc_q = lc_n[lc_n.quality == 0]
        nan_mask = np.isnan(lc_q.time.value) | np.isnan(lc_q.flux.value)
        lc_q = lc_q[~nan_mask]
        if len(lc_q) < 100:
            continue
            
        # 1. MASK SECTOR EDGES (Remove first and last 0.75 days to eliminate sector systematics)
        t = lc_q.time.value
        t_start, t_end = t.min(), t.max()
        edge_mask = (t > t_start + 0.75) & (t < t_end - 0.75)
        lc_q = lc_q[edge_mask]
        
        if len(lc_q) < 100:
            continue
            
        # 2. MASK MID-SECTOR DOWNLINK GAP
        t = lc_q.time.value
        diffs = np.diff(t)
        if len(diffs) > 0:
            max_gap_idx = np.argmax(diffs)
            # If the gap is larger than 12 hours (0.5 days), it's the downlink gap
            if diffs[max_gap_idx] > 0.5:
                gap_center = t[max_gap_idx] + 0.5 * diffs[max_gap_idx]
                gap_mask = np.abs(t - gap_center) > 0.5
                lc_q = lc_q[gap_mask]

        # Flare filter per sector
        med = np.median(lc_q.flux.value)
        std = np.std(lc_q.flux.value)
        mask = lc_q.flux.value < (med + 3 * std)
        lc_q = lc_q[mask]
        
        # 3. DETREND WITH LARGER WINDOW (1.5 days) AND ITERATIVE OUTLIER REJECTION
        # 1.5 days is ~1080 points in TESS 2-min cadence. Using 1001 (odd number).
        window = 1001
        if window > len(lc_q) // 2:
            window = (len(lc_q) // 4) * 2 + 1
        window = max(window, 101)
        
        # niters=3 and sigma=3 ignores the transit signal during fitting to avoid flattening it
        flat, trend = lc_q.flatten(window_length=window, return_trend=True, niters=3, sigma=3)
        sector_lcs.append((flat, trend))
    except Exception as e:
        print(f"  ⚠️ Sector skipped: {e}")
        continue

if len(sector_lcs) == 0:
    raise RuntimeError("No sectors were successfully processed.")

# Combine all clean sectors
all_time     = np.concatenate([s[0].time.value for s in sector_lcs])
all_flux     = np.concatenate([s[0].flux.value for s in sector_lcs])
all_trend    = np.concatenate([s[1].flux.value for s in sector_lcs])
all_raw_plot = all_trend * all_flux

# Sort arrays chronologically
sort_idx       = np.argsort(all_time)
time           = all_time[sort_idx]
flattened_flux = all_flux[sort_idx]
trend_flux     = all_trend[sort_idx]
raw_flux_plot  = all_raw_plot[sort_idx]
baseline       = time.max() - time.min()

print(f"✅ Sectors processed: {len(sector_lcs)}")
print(f"   Baseline : {baseline:.2f} days | Points: {len(time)}")
# ─────────────────────────────────────────────
# STEP 2: BIN LIGHT CURVE (10-minute cadence for fast search)
# ─────────────────────────────────────────────
print("\n🧹 Step 2: Binning light curve to 10-minute cadence...")
bin_size_days = 10.0 / 1440.0
nbins = int((time.max() - time.min()) / bin_size_days)

bin_medians, bin_edges, _ = binned_statistic(time, flattened_flux, statistic='median', bins=nbins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

nan_mask = np.isnan(bin_medians)
binned_time = bin_centers[~nan_mask]
binned_flux = bin_medians[~nan_mask]

# ─────────────────────────────────────────────
# STEP 3: VISUALISE PRE-PROCESSING
# ─────────────────────────────────────────────
print("\n📊 Step 3: Plotting preprocessing results...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.scatter(time, raw_flux_plot, s=1, color='black', label='Cleaned Flux')
ax1.plot(time, trend_flux, color='red', linewidth=2, label='Stellar Trend')
ax1.set_ylabel("Raw Relative Flux")
ax1.set_title(f"Stellar Trend Identification — {target_tic}")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

ax2.scatter(time, flattened_flux, s=1, color='blue', alpha=0.3, label='Preprocessed Flux')
ax2.scatter(binned_time, binned_flux, s=3, color='orange', label='Binned Flux (10-min)')
ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7)
ax2.set_ylabel("Normalized Flux")
ax2.set_xlabel("Time (TBJD)")
ax2.set_title("Final Preprocessed Light Curve (Baseline = 1.0)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# STEP 4: BLS SIGNAL DETECTION
# ─────────────────────────────────────────────
print("\n🔍 Step 4: BLS Signal Detection...")
print("   Scanning binned light curve (this will take ~5-15 seconds)...")

t0 = t_module.time()
bls = BoxLeastSquares(binned_time, binned_flux)
period_grid = np.linspace(0.5, 30, 15000)
durations   = np.linspace(0.03, 0.20, 20)

bls_power = bls.power(period_grid, durations)
best_idx  = np.argmax(bls_power.power)
elapsed_time = t_module.time() - t0
print(f"✅ Scanning complete! Elapsed time: {elapsed_time:.2f} seconds.")

top_indices = np.argsort(bls_power.power)[-5:][::-1]
candidate_periods = []

print("\nTop 5 BLS Candidates:")
for idx in top_indices:
    candidate_periods.append(round(float(bls_power.period[idx]), 4))
    print(
        f"  Period={bls_power.period[idx]:.4f} d | "
        f"Power={bls_power.power[idx]:.6f} | "
        f"Depth={bls_power.depth[idx]*100:.4f}%"
    )

best_period   = float(np.array(bls_power.period[best_idx]).item())
best_t0       = float(np.array(bls_power.transit_time[best_idx]).item())
transit_depth = float(np.array(bls_power.depth[best_idx]).item())
best_duration = float(np.array(bls_power.duration[best_idx]).item())
peak_power    = float(np.array(bls_power.power[best_idx]).item())

# ─────────────────────────────────────────────
# STEP 5: SNR & SCORING
# ─────────────────────────────────────────────
noise_sigma = np.std(binned_flux - np.median(binned_flux))

phases_temp = (
    (binned_time - best_t0 + 0.5 * best_period) % best_period - 0.5 * best_period
)
n_points   = int(np.sum(np.abs(phases_temp) < best_duration / 2))
snr        = float(transit_depth * np.sqrt(max(n_points, 1)) / noise_sigma)

baseline   = time.max() - time.min()
n_transits = max(1, round(baseline / best_period))

power_threshold = max(0.02 / len(sector_lcs), 0.0001)

snr_score     = np.clip(snr / 15, 0, 1)
power_score   = np.clip(peak_power / power_threshold, 0, 1)
transit_score = np.clip(n_transits / 3, 0, 1)
depth_score   = np.clip(0.025 / max(transit_depth, 1e-6), 0, 1)

confidence = round(
    100 * (0.40 * snr_score + 0.25 * power_score +
           0.20 * transit_score + 0.15 * depth_score), 2
)

reliability = "High" if n_transits >= 3 else "Medium" if n_transits == 2 else "Low"

# Calculate the duty cycle (ratio of duration to period)
duty_cycle = best_duration / best_period

# Check if the period aligns with TESS spacecraft orbits (13.7 d or 27.4 d)
is_tess_systematic = (
    np.abs(best_period - 13.7) < 0.20 or
    np.abs(best_period - 27.4) < 0.35
)

# Flag as systematic if it is too slow, has a wide duty cycle, or matches TESS orbits
is_systematic = (
    (best_duration * 24 > 12.0) or 
    (duty_cycle > 0.08 and best_period > 3.0) or 
    is_tess_systematic
)

# ─────────────────────────────────────────────
# FINAL CLASSIFICATION HEURISTICS
# ─────────────────────────────────────────────

if is_systematic:
    classification = "Quiet/Variable Star (TESS Systematic/Activity)"
elif snr > 8.0 and n_transits >= 2 and confidence >= 70:
    # If the signal is real and physical, classify by depth
    if transit_depth < 0.025:
        classification = "Exoplanet Host Star (Likely Planet Detected)"
    else:
        classification = "Eclipsing Binary Star System"
elif n_transits < 2:
    classification = "Single Transit Candidate - More Data Needed"
else:
    classification = "Quiet/Variable Star (No Planets Detected)"

print(f"\n🎯 PHASE 2 RESULTS")
print(f"  Period       : {best_period:.4f} days")
print(f"  SNR          : {snr:.2f}")
print(f"  Depth        : {transit_depth*100:.4f}%")
print(f"  Duration     : {best_duration*24:.2f} hr")
print(f"  BLS Power    : {peak_power:.4f}")
print(f"  Transit Count: {n_transits}  →  Reliability: {reliability}")
print(f"  Classification: {classification}")
print(f"  Confidence   : {confidence:.2f}%")

# ─────────────────────────────────────────────
# STEP 6: BLS PERIODOGRAM PLOT
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(bls_power.period, bls_power.power)
plt.xlabel("Period (days)"); plt.ylabel("BLS Power")
plt.title("BLS Periodogram"); plt.grid(alpha=0.3)
plt.show()

# ─────────────────────────────────────────────
# STEP 7: PHASE-FOLD PLOT
# ─────────────────────────────────────────────
phases = (time - best_t0 + 0.5 * best_period) % best_period - 0.5 * best_period

plt.figure(figsize=(10, 5))
plt.scatter(phases, flattened_flux, s=2, color="royalblue", alpha=0.2, label="Folded Data")

bin_edges  = np.linspace(-3 * best_duration, 3 * best_duration, 50)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
binned_flux_plot = [
    np.median(flattened_flux[(phases >= bin_edges[i]) & (phases < bin_edges[i+1])])
    if np.any((phases >= bin_edges[i]) & (phases < bin_edges[i+1])) else 1.0
    for i in range(len(bin_edges) - 1)
]
plt.plot(bin_centers, binned_flux_plot, color="red", linewidth=3, label="Binned Signal")
plt.title(f"Phase-Folded Signal — {target_tic}", fontsize=12)
plt.xlabel("Time from Transit Center (days)"); plt.ylabel("Normalized Flux")
plt.axhline(1.0, color="black", linestyle="--", alpha=0.6)
plt.xlim(-3 * best_duration, 3 * best_duration)
plt.grid(True, alpha=0.25)
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# STEP 8: SAVE RESULTS
# ─────────────────────────────────────────────
results = pd.DataFrame([{
    "TIC": target_tic,
    "Classification": classification,
    "Reliability": reliability,
    "Confidence_percent": confidence,
    "Period_days": round(best_period, 4),
    "Depth_percent": round(transit_depth * 100, 4),
    "Duration_hours": round(best_duration * 24, 2),
    "SNR": round(snr, 2),
    "BLS_Power": round(peak_power, 4),
    "Estimated_Transit_Count": n_transits,
    "Observation_Baseline_Days": round(baseline, 2),
    "Total_Data_Points": len(time),
    "Top_Candidate_Periods": str(candidate_periods)
}])

filename = "detection_results.csv"
results.to_csv(filename, mode="a", header=not os.path.exists(filename), index=False)
print(f"✅ Results saved to {filename}")