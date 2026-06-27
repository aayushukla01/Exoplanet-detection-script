import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time as t_module
from scipy.stats import binned_statistic
from astropy.timeseries import BoxLeastSquares

# Set page config for a premium wide dashboard layout
st.set_page_config(
    page_title="TESS Exoplanet Discovery Portal",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (Dark Theme styling, rounded corners, custom hover states)
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #FF9E2A 0%, #FF4B4B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #B0B3B8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E2022;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-title {
        color: #9AA0A6;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CORE BLS & SCORING PIPELINE
# ─────────────────────────────────────────────
def run_bls_and_score(time, flattened_flux, raw_flux_plot, trend_flux, target_name, baseline, n_sectors=1):
    """
    Runs the Box Least Squares (BLS) algorithm on the time and flux data,
    scores the signal, and applies classification heuristics.
    """
    # --- BIN LIGHT CURVE (10-minute cadence) ---
    bin_size_days = 10.0 / 1440.0
    nbins = int((time.max() - time.min()) / bin_size_days)
    bin_medians, bin_edges, _ = binned_statistic(time, flattened_flux, statistic='median', bins=nbins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    nan_mask = np.isnan(bin_medians)
    binned_time = bin_centers[~nan_mask]
    binned_flux = bin_medians[~nan_mask]

    # --- PRE-PROCESSING PLOT ---
    fig_pre, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.scatter(time, raw_flux_plot, s=1, color='black', label='Cleaned Flux')
    ax1.plot(time, trend_flux, color='red', linewidth=2, label='Stellar Trend')
    ax1.set_ylabel("Raw Relative Flux")
    ax1.set_title(f"Stellar Trend Identification — {target_name}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(time, flattened_flux, s=1, color='blue', alpha=0.15, label='Preprocessed Flux')
    ax2.scatter(binned_time, binned_flux, s=3, color='orange', label='Binned Flux (10-min)')
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel("Normalized Flux")
    ax2.set_xlabel("Time (days)")
    ax2.set_title("Final Preprocessed Light Curve (Baseline = 1.0)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- BLS SIGNAL DETECTION ---
    bls = BoxLeastSquares(binned_time, binned_flux)
    period_grid = np.linspace(0.5, 30, 15000)
    durations   = np.linspace(0.03, 0.20, 20)
    bls_power   = bls.power(period_grid, durations)
    
    best_idx  = np.argmax(bls_power.power)
    best_period   = float(np.array(bls_power.period[best_idx]).item())
    best_t0       = float(np.array(bls_power.transit_time[best_idx]).item())
    transit_depth = float(np.array(bls_power.depth[best_idx]).item())
    best_duration = float(np.array(bls_power.duration[best_idx]).item())
    peak_power    = float(np.array(bls_power.power[best_idx]).item())

    top_indices = np.argsort(bls_power.power)[-5:][::-1]
    candidate_periods = [round(float(bls_power.period[idx]), 4) for idx in top_indices]

    # --- SNR & SCORING ---
    noise_sigma = np.std(binned_flux - np.median(binned_flux))
    phases_temp = ((binned_time - best_t0 + 0.5 * best_period) % best_period - 0.5 * best_period)
    n_points   = int(np.sum(np.abs(phases_temp) < best_duration / 2))
    snr        = float(transit_depth * np.sqrt(max(n_points, 1)) / noise_sigma)
    n_transits = max(1, round(baseline / best_period))

    power_threshold = max(0.02 / n_sectors, 0.0001)
    snr_score     = np.clip(snr / 15, 0, 1)
    power_score   = np.clip(peak_power / power_threshold, 0, 1)
    transit_score = np.clip(n_transits / 3, 0, 1)
    depth_score   = np.clip(0.025 / max(transit_depth, 1e-6), 0, 1)

    confidence = round(
        100 * (0.40 * snr_score + 0.25 * power_score +
               0.20 * transit_score + 0.15 * depth_score), 2
    )
    reliability = "High" if n_transits >= 3 else "Medium" if n_transits == 2 else "Low"

    # Physics flags
    duty_cycle = best_duration / best_period
    is_tess_systematic = (np.abs(best_period - 13.7) < 0.20 or np.abs(best_period - 27.4) < 0.35)
    is_systematic = ((best_duration * 24 > 12.0) or (duty_cycle > 0.08 and best_period > 3.0) or is_tess_systematic)

    # Vetting: Stellar Variability Check (to detect rapid pulsators like Delta Scuti stars)
    # Discard the lowest 20% of flux values to completely ignore transits during noise estimate
    sorted_flux = np.sort(flattened_flux)
    out_of_transit = sorted_flux[int(len(flattened_flux) * 0.20):]
    std_out = np.std(out_of_transit)
    local_noise = np.std(np.diff(flattened_flux)) / np.sqrt(2)
    variability_ratio = std_out / local_noise if local_noise > 0 else 1.0
    
    is_pulsator = (variability_ratio > 1.8) and (best_period < 3.0)

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

    # --- BLS PERIODOGRAM ---
    fig_bls = plt.figure(figsize=(10, 4))
    plt.plot(bls_power.period, bls_power.power, color='crimson')
    plt.xlabel("Period (days)"); plt.ylabel("BLS Power")
    plt.title(f"Box Least Squares Periodogram — {target_name}")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    # --- PHASE-FOLD PLOT ---
    phases = (time - best_t0 + 0.5 * best_period) % best_period - 0.5 * best_period
    fig_fold = plt.figure(figsize=(10, 4))
    plt.scatter(phases, flattened_flux, s=2, color="royalblue", alpha=0.15, label="Folded Data")

    bin_edges  = np.linspace(-3 * best_duration, 3 * best_duration, 50)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_flux_plot = [
        np.median(flattened_flux[(phases >= bin_edges[i]) & (phases < bin_edges[i+1])])
        if np.any((phases >= bin_edges[i]) & (phases < bin_edges[i+1])) else 1.0
        for i in range(len(bin_edges) - 1)
    ]
    plt.plot(bin_centers, binned_flux_plot, color="red", linewidth=3, label="Binned Signal")
    plt.title(f"Phase-Folded Signal — {target_name}")
    plt.xlabel("Time from Transit Center (days)"); plt.ylabel("Normalized Flux")
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.6)
    plt.xlim(-3 * best_duration, 3 * best_duration)
    plt.grid(True, alpha=0.2)
    plt.legend(loc="lower left")
    plt.tight_layout()

    metrics = {
        "TIC": target_name,
        "Period": best_period,
        "SNR": snr,
        "Depth": transit_depth * 100,
        "Duration": best_duration * 24,
        "BLS Power": peak_power,
        "Transit Count": n_transits,
        "Reliability": reliability,
        "Classification": classification,
        "Confidence": confidence,
        "Baseline": baseline,
        "Data Points": len(time),
        "Candidates": candidate_periods
    }

    return metrics, fig_pre, fig_bls, fig_fold


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
            import re
            import shutil
            # Extract absolute FITS file paths from the error message (both Linux and Windows)
            fits_paths = re.findall(r'(/[^\s]+\.fits|[a-zA-Z]:\\[^\s]+\.fits)', err_msg)
            
            cleaned_any = False
            for path in fits_paths:
                path = path.strip().rstrip('.').rstrip(',')
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        cleaned_any = True
                        parent_dir = os.path.dirname(path)
                        if os.path.exists(parent_dir):
                            shutil.rmtree(parent_dir)
                    except Exception:
                        pass
            
            # Fallback: if regex didn't extract path, clear the target's folders by TIC ID
            if not cleaned_any:
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
# MAST SEARCH WORKFLOW
# ─────────────────────────────────────────────
def analyze_mast_target(target_tic, progress_bar=None, status_text=None, max_sectors=3):
    """Query MAST and run BLS on TESS light curves."""
    raw_numeric_id = "".join(filter(str.isdigit, target_tic))
    formatted_tic = f"TIC {raw_numeric_id}"
    
    if status_text:
        status_text.text(f"📡 Step 1: Querying MAST archive for {formatted_tic}...")
    if progress_bar:
        progress_bar.progress(10)
        
    search_result = lk.search_lightcurve(formatted_tic, author="SPOC", exptime=120)
    if len(search_result) == 0:
        if status_text:
            status_text.text("🔄 SPOC 2-min empty. Broadening search to all available light curves (FFIs)...")
        search_result = lk.search_lightcurve(formatted_tic)
    if len(search_result) == 0:
        raise RuntimeError(f"No light curve products found for {formatted_tic}.")
        
    # Limit sectors for speed
    if len(search_result) > max_sectors:
        search_result = search_result[:max_sectors]
    
    if status_text:
        status_text.text(f"📥 Found {len(search_result)} sectors. Downloading target light curves...")
    if progress_bar:
        progress_bar.progress(25)
        
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lk_cache")
    lc_collection = download_with_self_healing(search_result, cache_dir, formatted_tic)
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
                
            # Mask sector edges
            t = lc_q.time.value
            t_start, t_end = t.min(), t.max()
            edge_mask = (t > t_start + 0.75) & (t < t_end - 0.75)
            lc_q = lc_q[edge_mask]
            if len(lc_q) < 100:
                continue
                
            # Mask mid-sector downlink gap
            t = lc_q.time.value
            diffs = np.diff(t)
            if len(diffs) > 0:
                max_gap_idx = np.argmax(diffs)
                if diffs[max_gap_idx] > 0.5:
                    gap_center = t[max_gap_idx] + 0.5 * diffs[max_gap_idx]
                    gap_mask = np.abs(t - gap_center) > 0.5
                    lc_q = lc_q[gap_mask]

            # Flare filter
            med = np.median(lc_q.flux.value)
            std = np.std(lc_q.flux.value)
            mask = lc_q.flux.value < (med + 3 * std)
            lc_q = lc_q[mask]
            
            # Detrending (1.5-day window)
            window = 1001
            if window > len(lc_q) // 2:
                window = (len(lc_q) // 4) * 2 + 1
            window = max(window, 101)
            
            flat, trend = lc_q.flatten(window_length=window, return_trend=True, niters=3, sigma=3)
            sector_lcs.append((flat, trend))
        except Exception:
            continue

    if len(sector_lcs) == 0:
        raise RuntimeError("No sectors were successfully processed.")

    all_time     = np.concatenate([s[0].time.value for s in sector_lcs])
    all_flux     = np.concatenate([s[0].flux.value for s in sector_lcs])
    all_trend    = np.concatenate([s[1].flux.value for s in sector_lcs])
    all_raw_plot = all_trend * all_flux

    sort_idx       = np.argsort(all_time)
    time           = all_time[sort_idx]
    flattened_flux = all_flux[sort_idx]
    trend_flux     = all_trend[sort_idx]
    raw_flux_plot  = all_raw_plot[sort_idx]
    baseline       = time.max() - time.min()

    if status_text:
        status_text.text("🧹 Running Box Least Squares periodic analysis...")
    if progress_bar:
        progress_bar.progress(50)

    results = run_bls_and_score(time, flattened_flux, raw_flux_plot, trend_flux, formatted_tic, baseline, len(sector_lcs))
    
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
        
    return results


# ─────────────────────────────────────────────
# MAIN STREAMLIT APP LAYOUT
# ─────────────────────────────────────────────
st.markdown("<div class='main-title'>🪐 TESS Exoplanet Discovery Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze stellar light curves and detect periodic exoplanetary transit signals</div>", unsafe_allow_html=True)

# Sidebar config
st.sidebar.header("🛠️ Pipeline Controls")
input_mode = st.sidebar.radio(
    "Select Target Input Mode", 
    [
        "🔍 Search MAST by TIC ID", 
        "📈 Upload Custom Light Curve (CSV/TXT)", 
        "📋 Batch Process TIC Target List (CSV)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Performance Optimization")
max_sectors = st.sidebar.slider(
    "Max Sectors to Download", 
    min_value=1, 
    max_value=15, 
    value=3, 
    help="Limiting sectors dramatically speeds up downloads and computations (e.g. 10s vs 6 mins) while still finding all short/medium period planets."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Astronomical Presets")
st.sidebar.info("• Min Cadence: 120s\n• Min Orbit Scan: 0.5d\n• Max Orbit Scan: 30.0d\n• Detrend Window: 1.5d")

# ─────────────────────────────────────────────
# MODE 1: SEARCH MAST BY TIC ID
# ─────────────────────────────────────────────
if input_mode == "🔍 Search MAST by TIC ID":
    st.subheader("Analyze a Target from MAST Archive")
    
    col_input, col_info = st.columns([2, 3])
    with col_input:
        target_input = st.text_input("Enter TESS Input Catalog (TIC) ID:", value="TIC 261136679", placeholder="e.g. TIC 261136679")
        run_btn = st.button("🚀 Run Pipeline Scan", use_container_width=True)
        
    with col_info:
        st.markdown("""
        **How it works:**
        1. Queries the **MAST archive** to download light curves.
        2. Filters orbital gap systematics and stellar flares.
        3. Normalizes and detrends using a 1.5-day Savitzky-Golay filter.
        4. Runs a **Box Least Squares (BLS)** algorithm to find transiting planet candidates.
        """)

    if run_btn:
        prog_bar = st.progress(0)
        status = st.empty()
        
        try:
            with st.spinner("Analyzing light curves..."):
                start_time = t_module.time()
                metrics, fig_pre, fig_bls, fig_fold = analyze_mast_target(target_input, prog_bar, status, max_sectors=max_sectors)
                elapsed = t_module.time() - start_time
                
            st.success(f"Successfully processed target in {elapsed:.2f} seconds!")
            
            # Display results summary in cards
            st.markdown("### 🎯 Classification & Signal Summary")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Classification</div>
                    <div class='metric-value' style='font-size: 1.25rem; color:#FFA726;'>{metrics['Classification']}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #29B6F6;'>
                    <div class='metric-title'>Orbital Period</div>
                    <div class='metric-value'>{metrics['Period']:.4f} days</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #66BB6A;'>
                    <div class='metric-title'>Transit Depth</div>
                    <div class='metric-value'>{metrics['Depth']:.4f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #AB47BC;'>
                    <div class='metric-title'>Confidence Score</div>
                    <div class='metric-value'>{metrics['Confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Secondary Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal SNR", f"{metrics['SNR']:.2f}")
            c2.metric("Transit Duration", f"{metrics['Duration']:.2f} hours")
            c3.metric("Transit Count", f"{metrics['Transit Count']} transits ({metrics['Reliability']})")
            c4.metric("Observation Baseline", f"{metrics['Baseline']:.1f} days ({metrics['Data Points']} points)")
            
            # Download Analysis Results
            download_df = pd.DataFrame([{
                "TIC": metrics["TIC"],
                "Classification": metrics["Classification"],
                "Reliability": metrics["Reliability"],
                "Confidence_percent": metrics["Confidence"],
                "Period_days": round(metrics["Period"], 4),
                "Depth_percent": round(metrics["Depth"], 4),
                "Duration_hours": round(metrics["Duration"], 2),
                "SNR": round(metrics["SNR"], 2),
                "BLS_Power": round(metrics["BLS Power"], 4),
                "Estimated_Transit_Count": metrics["Transit Count"],
                "Observation_Baseline_Days": round(metrics["Baseline"], 2),
                "Total_Data_Points": metrics["Data Points"],
                "Top_Candidate_Periods": str(metrics["Candidates"])
            }])
            csv_data = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Analysis Results (CSV)",
                data=csv_data,
                file_name=f"{metrics['TIC'].replace(' ', '_')}_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Displays plots in neat tabs
            tab1, tab2, tab3 = st.tabs(["📊 Preprocessing & Detrending", "📈 BLS Periodogram Grid", "🪐 Phase-Folded Transit Profile"])
            
            with tab1:
                st.pyplot(fig_pre)
            with tab2:
                st.pyplot(fig_bls)
            with tab3:
                st.pyplot(fig_fold)
                
            # Save results locally too
            res_df = pd.DataFrame([{
                "TIC": metrics["TIC"],
                "Classification": metrics["Classification"],
                "Reliability": metrics["Reliability"],
                "Confidence_percent": metrics["Confidence"],
                "Period_days": round(metrics["Period"], 4),
                "Depth_percent": round(metrics["Depth"], 4),
                "Duration_hours": round(metrics["Duration"], 2),
                "SNR": round(metrics["SNR"], 2),
                "BLS_Power": round(metrics["BLS Power"], 4),
                "Estimated_Transit_Count": metrics["Transit Count"],
                "Observation_Baseline_Days": round(metrics["Baseline"], 2),
                "Total_Data_Points": metrics["Data Points"],
                "Top_Candidate_Periods": str(metrics["Candidates"])
            }])
            res_df.to_csv("detection_results.csv", mode="a", header=not os.path.exists("detection_results.csv"), index=False)
            
        except Exception as err:
            st.error(f"❌ Error scanning {target_input}: {err}")
            st.exception(err)

# ─────────────────────────────────────────────
# MODE 2: UPLOAD CUSTOM LIGHT CURVE (CSV/TXT)
# ─────────────────────────────────────────────
elif input_mode == "📈 Upload Custom Light Curve (CSV/TXT)":
    st.subheader("Analyze an Uploaded Light Curve File")
    
    st.markdown("""
    Upload a CSV, TSV, or text file containing time and flux measurements of any star. 
    The file can be from other telescopes, simulations, or pre-extracted TESS databases.
    """)
    
    uploaded_lc = st.file_uploader("Upload Light Curve File (CSV, TSV, TXT):", type=["csv", "tsv", "txt"])
    
    if uploaded_lc is not None:
        try:
            # Read file (auto-detecting separator)
            if uploaded_lc.name.endswith(".tsv") or uploaded_lc.name.endswith(".txt"):
                df_lc = pd.read_csv(uploaded_lc, sep=None, engine='python')
            else:
                df_lc = pd.read_csv(uploaded_lc)
                
            st.success("File uploaded successfully!")
            st.markdown("### Data Preview (First 5 Rows)")
            st.dataframe(df_lc.head(5))
            
            # Column selector dropdowns
            cols = df_lc.columns.tolist()
            
            # Helper to guess time/flux columns
            def guess_column(cols, keys):
                for col in cols:
                    if any(k in col.lower() for k in keys):
                        return col
                return cols[0]
                
            col_sel_time, col_sel_flux, col_sel_name = st.columns(3)
            with col_sel_time:
                time_col = st.selectbox(
                    "Select Time Column (X-axis, in days):", 
                    cols, 
                    index=cols.index(guess_column(cols, ["time", "t", "bjd", "hjd", "days"]))
                )
            with col_sel_flux:
                flux_col = st.selectbox(
                    "Select Flux Column (Y-axis):", 
                    cols, 
                    index=cols.index(guess_column(cols, ["flux", "f", "sap", "pdcsap", "relative", "light"]))
                )
            with col_sel_name:
                custom_name = st.text_input("Enter Star/Target Name:", value="Uploaded Target")
                
            apply_detrend = st.checkbox("Apply 1.5-day Stellar Trend Flattening?", value=True)
            
            run_lc_btn = st.button("🚀 Analyze Uploaded Light Curve", use_container_width=True)
            
            if run_lc_btn:
                # Convert values to floats
                time_arr = pd.to_numeric(df_lc[time_col], errors='coerce').values
                flux_arr = pd.to_numeric(df_lc[flux_col], errors='coerce').values
                
                # Filter out NaNs
                valid_mask = ~np.isnan(time_arr) & ~np.isnan(flux_arr)
                time_arr = time_arr[valid_mask]
                flux_arr = flux_arr[valid_mask]
                
                if len(time_arr) < 100:
                    st.error("❌ Not enough valid data points (minimum 100 required). Please check your column selections.")
                else:
                    prog_bar = st.progress(10)
                    status = st.empty()
                    
                    with st.spinner("Analyzing custom light curve..."):
                        # Chronological sorting
                        sort_idx = np.argsort(time_arr)
                        time = time_arr[sort_idx]
                        flux = flux_arr[sort_idx]
                        baseline = time.max() - time.min()
                        
                        prog_bar.progress(30)
                        status.text("🧹 Preparing data...")
                        
                        # Apply detrending if checked
                        if apply_detrend:
                            status.text("🧹 Flattening stellar activity trends...")
                            # Create a dummy lightkurve object to utilize its flatten method
                            lc_obj = lk.LightCurve(time=time, flux=flux)
                            window = 1001
                            if window > len(lc_obj) // 2:
                                window = (len(lc_obj) // 4) * 2 + 1
                            window = max(window, 101)
                            
                            flat, trend = lc_obj.flatten(window_length=window, return_trend=True, niters=3, sigma=3)
                            flattened_flux = flat.flux.value
                            trend_flux = trend.flux.value
                            raw_flux_plot = trend_flux * flattened_flux
                        else:
                            flattened_flux = flux
                            trend_flux = np.ones_like(flux)
                            raw_flux_plot = flux
                        
                        prog_bar.progress(50)
                        status.text("🔍 Running Box Least Squares (BLS) periodic scan...")
                        
                        # Run the core BLS search and scoring
                        metrics, fig_pre, fig_bls, fig_fold = run_bls_and_score(
                            time, flattened_flux, raw_flux_plot, trend_flux, custom_name, baseline, n_sectors=1
                        )
                        
                        prog_bar.progress(100)
                        status.text("✅ Analysis complete!")
                        
                    st.success("Successfully analyzed your uploaded light curve!")
                    
                    # Display results summary in cards
                    st.markdown("### 🎯 Classification & Signal Summary")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-title'>Classification</div>
                            <div class='metric-value' style='font-size: 1.25rem; color:#FFA726;'>{metrics['Classification']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class='metric-card' style='border-left-color: #29B6F6;'>
                            <div class='metric-title'>Orbital Period</div>
                            <div class='metric-value'>{metrics['Period']:.4f} days</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class='metric-card' style='border-left-color: #66BB6A;'>
                            <div class='metric-title'>Transit Depth</div>
                            <div class='metric-value'>{metrics['Depth']:.4f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"""
                        <div class='metric-card' style='border-left-color: #AB47BC;'>
                            <div class='metric-title'>Confidence Score</div>
                            <div class='metric-value'>{metrics['Confidence']}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Secondary Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Signal SNR", f"{metrics['SNR']:.2f}")
                    c2.metric("Transit Duration", f"{metrics['Duration']:.2f} hours")
                    c3.metric("Transit Count", f"{metrics['Transit Count']} transits ({metrics['Reliability']})")
                    c4.metric("Observation Baseline", f"{metrics['Baseline']:.1f} days ({metrics['Data Points']} points)")
                    
                    # Download Analysis Results
                    download_df = pd.DataFrame([{
                        "TIC": metrics["TIC"],
                        "Classification": metrics["Classification"],
                        "Reliability": metrics["Reliability"],
                        "Confidence_percent": metrics["Confidence"],
                        "Period_days": round(metrics["Period"], 4),
                        "Depth_percent": round(metrics["Depth"], 4),
                        "Duration_hours": round(metrics["Duration"], 2),
                        "SNR": round(metrics["SNR"], 2),
                        "BLS_Power": round(metrics["BLS Power"], 4),
                        "Estimated_Transit_Count": metrics["Transit Count"],
                        "Observation_Baseline_Days": round(metrics["Baseline"], 2),
                        "Total_Data_Points": metrics["Data Points"],
                        "Top_Candidate_Periods": str(metrics["Candidates"])
                    }])
                    csv_data = download_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Analysis Results (CSV)",
                        data=csv_data,
                        file_name=f"{metrics['TIC'].replace(' ', '_')}_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Displays plots in neat tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Preprocessing & Detrending", "📈 BLS Periodogram Grid", "🪐 Phase-Folded Transit Profile"])
                    
                    with tab1:
                        st.pyplot(fig_pre)
                    with tab2:
                        st.pyplot(fig_bls)
                    with tab3:
                        st.pyplot(fig_fold)
                        
        except Exception as e:
            st.error(f"Error parsing custom light curve file: {e}")
            st.exception(e)

# ─────────────────────────────────────────────
# MODE 3: BATCH CSV UPLOAD
# ─────────────────────────────────────────────
elif input_mode == "📋 Batch Process TIC Target List (CSV)":
    st.subheader("Process a Catalog of Targets")
    
    st.markdown("""
    Upload a CSV catalog containing a list of target TIC IDs. The script will automatically process
    them in a loop, save the transit graphs, and provide a downloadable results spreadsheet.
    """)
    
    uploaded_file = st.file_uploader("Upload a CSV file of target IDs (zipped .csv.gz is also supported):", type=["csv", "gz"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".gz"):
                df_targets = pd.read_csv(uploaded_file, compression="gzip", header=None)
            else:
                df_targets = pd.read_csv(uploaded_file, header=None)
                
            # If standard TIC catalog, sort by TESS Magnitude (column 28) to run brightest stars first
            if df_targets.shape[1] >= 29:
                try:
                    df_targets[28] = pd.to_numeric(df_targets[28], errors='coerce')
                    df_targets = df_targets.sort_values(by=28, ascending=True)
                    st.info("💡 Standard TESS Catalog detected. Auto-sorting stars by brightness (TESS Magnitude) to prioritize processing targets with active data!")
                except Exception:
                    pass
                
            # Read first column
            raw_ids = df_targets[0].astype(str).tolist()
            tic_list = []
            for val in raw_ids:
                num_id = "".join(filter(str.isdigit, val))
                if num_id:
                    tic_list.append(f"TIC {num_id}")
                    
            st.success(f"Successfully loaded {len(tic_list)} targets from uploaded file!")
            
            # Setup limits
            limit = st.number_input("Limit run to first N targets (recommended for long runs):", min_value=1, max_value=len(tic_list), value=min(5, len(tic_list)))
            selected_tics = tic_list[:limit]
            
            run_batch_btn = st.button(f"🚀 Begin Batch Processing ({limit} targets)", use_container_width=True)
            
            if run_batch_btn:
                # Progress bars
                overall_progress = st.progress(0)
                overall_status = st.empty()
                target_status = st.empty()
                target_progress = st.progress(0)
                
                # Placeholder table to display results on the fly
                results_table_placeholder = st.empty()
                
                # Plot expander so plots don't clog up screen
                plots_expander = st.expander("📈 View Generated Plots for Detections", expanded=False)
                
                batch_results = []
                
                # Create local plots folder
                os.makedirs("./plots", exist_ok=True)
                
                for idx, tic in enumerate(selected_tics):
                    overall_status.markdown(f"**Target {idx+1}/{len(selected_tics)}:** Processing `{tic}`...")
                    overall_progress.progress(int((idx / len(selected_tics)) * 100))
                    
                    try:
                        metrics, fig_pre, fig_bls, fig_fold = analyze_mast_target(tic, target_progress, target_status, max_sectors=max_sectors)
                        
                        # Add results
                        batch_results.append({
                            "TIC": metrics["TIC"],
                            "Classification": metrics["Classification"],
                            "Confidence": f"{metrics['Confidence']}%",
                            "Period": f"{metrics['Period']:.4f} days",
                            "Depth": f"{metrics['Depth']:.4f}%",
                            "SNR": f"{metrics['SNR']:.2f}"
                        })
                        
                        # Display updated table
                        results_table_placeholder.table(pd.DataFrame(batch_results))
                        
                        # Save plots locally
                        fig_pre.savefig(f"./plots/{metrics['TIC']}_preprocessing.png", dpi=150)
                        fig_bls.savefig(f"./plots/{metrics['TIC']}_periodogram.png", dpi=150)
                        fig_fold.savefig(f"./plots/{metrics['TIC']}_folded.png", dpi=150)
                        
                        # Add to UI expander
                        with plots_expander:
                            st.markdown(f"#### Target: {metrics['TIC']} — {metrics['Classification']}")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.pyplot(fig_pre)
                            with c2:
                                st.pyplot(fig_fold)
                            st.markdown("---")
                            
                        # Save row to detection_results.csv
                        row_df = pd.DataFrame([{
                            "TIC": metrics["TIC"],
                            "Classification": metrics["Classification"],
                            "Reliability": metrics["Reliability"],
                            "Confidence_percent": metrics["Confidence"],
                            "Period_days": round(metrics["Period"], 4),
                            "Depth_percent": round(metrics["Depth"], 4),
                            "Duration_hours": round(metrics["Duration"], 2),
                            "SNR": round(metrics["SNR"], 2),
                            "BLS_Power": round(metrics["BLS Power"], 4),
                            "Estimated_Transit_Count": metrics["Transit Count"],
                            "Observation_Baseline_Days": round(metrics["Baseline"], 2),
                            "Total_Data_Points": metrics["Data Points"],
                            "Top_Candidate_Periods": str(metrics["Candidates"])
                        }])
                        row_df.to_csv("detection_results.csv", mode="a", header=not os.path.exists("detection_results.csv"), index=False)
                        
                        # Free up matplotlib memory
                        plt.close('all')
                        
                    except Exception as batch_err:
                        st.warning(f"⚠️ Failed to process {tic}: {batch_err}")
                        continue
                
                overall_progress.progress(100)
                overall_status.success("🎉 Batch processing complete!")
                target_status.empty()
                target_progress.empty()
                
                # Download Button for the complete csv
                if os.path.exists("detection_results.csv"):
                    with open("detection_results.csv", "rb") as f:
                        st.download_button(
                            label="⬇️ Download Full Results Catalog (CSV)",
                            data=f,
                            file_name="detection_results.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error loading file: {e}")
