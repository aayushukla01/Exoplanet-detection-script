import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time as t_module
import glob
from scipy.stats import binned_statistic
from astropy.timeseries import BoxLeastSquares

# ─────────────────────────────────────────────
# SELF-HEALING DOWNLOAD ROUTINE
# ─────────────────────────────────────────────
def download_with_self_healing(search_result, cache_dir, target_tic):
    """Downloads light curves, automatically detecting and cleaning up corrupt FITS files on failure."""
    try:
        return search_result.download_all(download_dir=cache_dir)
    except Exception as e:
        err_msg = str(e)
        is_corrupt = (
            "corrupt" in err_msg.lower() or 
            "not recognized as a supported data product" in err_msg.lower() or
            "error in reading data product" in err_msg.lower()
        )
        if is_corrupt:
            import shutil
            numeric_id = "".join(filter(str.isdigit, target_tic))
            mast_dir = os.path.join(cache_dir, "mastDownload")
            if os.path.exists(mast_dir) and numeric_id:
                for root, dirs, files in os.walk(mast_dir, topdown=False):
                    for name in files:
                        if numeric_id in name:
                            try:
                                os.remove(os.path.join(root, name))
                            except Exception:
                                pass
                    for name in dirs:
                        if numeric_id in name:
                            try:
                                shutil.rmtree(os.path.join(root, name))
                            except Exception:
                                pass
            # Retry download after self-healing cache cleanup
            return search_result.download_all(download_dir=cache_dir)
        else:
            raise e


# ─────────────────────────────────────────────
# TARGET INPUT SELECTOR & AUTO-DETECTION
# ─────────────────────────────────────────────
print("🚀 TESS Exoplanet Search Pipeline")
print("1. Process a single star by entering a TIC ID")
print("2. Batch process stars from a CSV target list")
choice = input("Select input mode (1 or 2): ").strip()

tic_list = []
is_batch = False

if choice == "2":
    is_batch = True
    
    # Auto-detect target CSV/GZ files in the current folder (excluding our results file)
    detected_catalogs = glob.glob("*.csv") + glob.glob("*.csv.gz")
    detected_catalogs = [f for f in detected_catalogs if f != "detection_results.csv"]
    
    csv_path = ""
    if len(detected_catalogs) == 1:
        csv_path = detected_catalogs[0]
        print(f"📂 Auto-detected target catalog: '{csv_path}'")
    elif len(detected_catalogs) > 1:
        print("\nMultiple target catalogs detected:")
        for idx, f in enumerate(detected_catalogs):
            print(f"  {idx + 1}. {f}")
        selection = input("Select a catalog file (enter number) or press Enter to type path: ").strip()
        if selection:
            try:
                csv_path = detected_catalogs[int(selection) - 1]
            except (ValueError, IndexError):
                pass
    
    # Fallback to manual path input if auto-detect found nothing or was skipped
    if not csv_path:
        csv_path = input("Enter path to the CSV/CSV.GZ file: ").strip()
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find file at: {csv_path}")
        
    print(f"Reading target file: {csv_path}...")
    try:
        if csv_path.endswith('.gz'):
            df = pd.read_csv(csv_path, compression='gzip', header=None)
        else:
            df = pd.read_csv(csv_path, header=None)
        
        # If standard TIC catalog, sort by TESS Magnitude (column 28) to run brightest stars first
        if df.shape[1] >= 29:
            try:
                df[28] = pd.to_numeric(df[28], errors='coerce')
                df = df.sort_values(by=28, ascending=True)
                print("💡 Standard TESS Catalog detected. Auto-sorting targets by brightness (TESS Magnitude) to prioritize stars with active data!")
            except Exception:
                pass
            
        # Read the first column and extract numeric TIC IDs
        raw_ids = df[0].astype(str).tolist()
        for val in raw_ids:
            num_id = "".join(filter(str.isdigit, val))
            if num_id:
                tic_list.append(f"TIC {num_id}")
                
        print(f"Found {len(tic_list)} target TIC IDs in file.")
    except Exception as e:
        raise RuntimeError(f"Error reading target CSV file: {e}")
        
    limit_input = input("How many targets would you like to process? (Leave blank to process all): ").strip()
    if limit_input:
        try:
            limit = int(limit_input)
            tic_list = tic_list[:limit]
            print(f"Limiting run to the first {limit} targets.")
        except ValueError:
            print("Invalid number. Processing all targets.")
else:
    # Option 1: Single star target
    target_tic_input = input("Enter target TIC ID (default: TIC 261136679): ").strip()
    if not target_tic_input:
        target_tic_input = "TIC 261136679"
    tic_list = [target_tic_input]

print(f"\n🚀 Starting pipeline run for {len(tic_list)} target(s)...")

# Create a folder to save plots if running in batch mode
if is_batch:
    os.makedirs("./plots", exist_ok=True)
    print("📁 Plots will be saved automatically to the './plots' directory.")

# ─────────────────────────────────────────────
# PIPELINE LOOP
# ─────────────────────────────────────────────
for idx, current_tic in enumerate(tic_list):
    print(f"\n=====================================================================")
    print(f"📡 Target {idx+1}/{len(tic_list)}: {current_tic}")
    print(f"=====================================================================")
    
    try:
        raw_numeric_id = "".join(filter(str.isdigit, current_tic))
        formatted_tic = f"TIC {raw_numeric_id}"
        
        # ─────────────────────────────────────────────
        # STEP 1: SEARCH & DOWNLOAD (FAST PARALLEL VERSION)
        # ─────────────────────────────────────────────
        print("📡 Step 1: Searching MAST archive for TESS light curves...")
        search_result = lk.search_lightcurve(formatted_tic, author="SPOC", exptime=120)
        if len(search_result) == 0:
            print("🔄 SPOC 2-min empty. Broadening search to all available light curves (FFIs)...")
            search_result = lk.search_lightcurve(formatted_tic)
        if len(search_result) == 0:
            print(f"  ⚠️ Skipping {formatted_tic}: No light curves found.")
            continue

        # Limit to first 3 sectors for speed
        if len(search_result) > 3:
            print("⚡ Limiting query to 3 sectors to optimize processing speed...")
            search_result = search_result[:3]

        print(f"📥 Found {len(search_result)} products. Downloading in parallel...")
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lk_cache")

        lc_collection = download_with_self_healing(search_result, cache_dir, formatted_tic)
        if lc_collection is None or len(lc_collection) == 0:
            print(f"  ⚠️ Skipping {formatted_tic}: Download failed.")
            continue

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
                window = 1001
                if window > len(lc_q) // 2:
                    window = (len(lc_q) // 4) * 2 + 1
                window = max(window, 101)
                
                flat, trend = lc_q.flatten(window_length=window, return_trend=True, niters=3, sigma=3)
                sector_lcs.append((flat, trend))
            except Exception as e:
                print(f"  ⚠️ Sector skipped: {e}")
                continue

        if len(sector_lcs) == 0:
            print(f"  ⚠️ Skipping {formatted_tic}: No sectors successfully processed.")
            continue

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
        print("🧹 Step 2: Binning light curve to 10-minute cadence...")
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
        print("📊 Step 3: Generating preprocessing plot...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.scatter(time, raw_flux_plot, s=1, color='black', label='Cleaned Flux')
        ax1.plot(time, trend_flux, color='red', linewidth=2, label='Stellar Trend')
        ax1.set_ylabel("Raw Relative Flux")
        ax1.set_title(f"Stellar Trend Identification — {formatted_tic}")
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
        if is_batch:
            plt.savefig(f"./plots/{formatted_tic}_preprocessing.png", dpi=150)
            plt.close()
        else:
            plt.show()

        # ─────────────────────────────────────────────
        # STEP 4: BLS SIGNAL DETECTION
        # ─────────────────────────────────────────────
        print("🔍 Step 4: BLS Signal Detection...")
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

        # Vetting: Stellar Variability Check (to detect rapid pulsators like Delta Scuti stars)
        # Discard the lowest 20% of flux values to completely ignore transits during noise estimate
        sorted_flux = np.sort(flattened_flux)
        out_of_transit = sorted_flux[int(len(flattened_flux) * 0.20):]
        std_out = np.std(out_of_transit)
        local_noise = np.std(np.diff(flattened_flux)) / np.sqrt(2)
        variability_ratio = std_out / local_noise if local_noise > 0 else 1.0
        
        is_pulsator = (variability_ratio > 1.8) and (best_period < 3.0)

        # ─────────────────────────────────────────────
        # FINAL CLASSIFICATION HEURISTICS
        # ─────────────────────────────────────────────
        if is_systematic or is_pulsator:
            classification = "Quiet/Variable Star (No Planets Detected)"
        elif snr > 8.0 and n_transits >= 2 and confidence >= 70:
            if transit_depth < 0.025:
                classification = "Exoplanet Host Star (Likely Planet Detected)"
            else:
                classification = "Eclipsing Binary Star System"
        elif n_transits < 2:
            classification = "Single Transit Candidate - More Data Needed"
        else:
            classification = "Quiet/Variable Star (No Planets Detected)"

        print(f"\n🎯 PHASE 2 RESULTS ({formatted_tic})")
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
        plt.title(f"BLS Periodogram — {formatted_tic}")
        plt.grid(alpha=0.3)
        if is_batch:
            plt.savefig(f"./plots/{formatted_tic}_periodogram.png", dpi=150)
            plt.close()
        else:
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
        plt.title(f"Phase-Folded Signal — {formatted_tic}", fontsize=12)
        plt.xlabel("Time from Transit Center (days)"); plt.ylabel("Normalized Flux")
        plt.axhline(1.0, color="black", linestyle="--", alpha=0.6)
        plt.xlim(-3 * best_duration, 3 * best_duration)
        plt.grid(True, alpha=0.25)
        plt.legend(loc="lower left")
        plt.tight_layout()
        if is_batch:
            plt.savefig(f"./plots/{formatted_tic}_folded.png", dpi=150)
            plt.close()
        else:
            plt.show()

        # ─────────────────────────────────────────────
        # STEP 8: SAVE RESULTS
        # ─────────────────────────────────────────────
        results = pd.DataFrame([{
            "TIC": formatted_tic,
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
        
        # Post-analysis cache cleanup (delete downloaded FITS files for this target to keep disk 100% clean)
        try:
            numeric_id = "".join(filter(str.isdigit, formatted_tic))
            mast_dir = os.path.join(cache_dir, "mastDownload")
            if os.path.exists(mast_dir) and numeric_id:
                import shutil
                for root, dirs, files in os.walk(mast_dir, topdown=False):
                    for name in files:
                        if numeric_id in name:
                            try:
                                os.remove(os.path.join(root, name))
                            except Exception:
                                pass
                    for name in dirs:
                        if numeric_id in name:
                            try:
                                shutil.rmtree(os.path.join(root, name))
                            except Exception:
                                pass
        except Exception:
            pass
        
    except Exception as target_error:
        print(f"  ❌ Failed to process {current_tic}: {target_error}")
        continue

print("\n🎉 Pipeline run complete!")
if is_batch:
    print(f"📁 All plots saved inside the './plots/' directory.")