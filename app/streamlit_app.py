"""
Maximum Entropy Presence Prediction Application
Streamlit Launcher for MaxEnt modeling with Python and R integration
"""
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add app directories to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR / "python"))
sys.path.insert(0, str(APP_DIR / "utils"))

from coordinate_transformer import CoordinateTransformer
from maxent_core import MaxEntModel
from presence_processor import PresenceProcessor
from r_bridge import check_r_environment, install_r_packages
from raster_processor import RasterProcessor
from report_generator import ReportGenerator, compute_prediction_stats, compute_roc_data

# Page configuration
st.set_page_config(
    page_title="MaxEnt Presence Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1B5E20;
        padding: 0.5rem 0;
        border-bottom: 1px solid #C8E6C9;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #E8F5E9;
        color: #1B4332;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .info-box strong, .info-box p, .info-box span {
        color: #0F2E1C;
    }
    .warning-box {
        background-color: #FFF3E0;
        color: #7C2D12;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box strong, .warning-box p, .warning-box span {
        color: #7C2D12;
    }
    .success-box {
        background-color: #E8F5E9;
        color: #1B4332;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box strong, .success-box p, .success-box span {
        color: #0F2E1C;
    }
    .error-box {
        background-color: #FFEBEE;
        color: #7F1D1D;
        border-left: 4px solid #F44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .error-box strong, .error-box p, .error-box span {
        color: #7F1D1D;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #388E3C;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        color: #1F2937;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "results" not in st.session_state:
    st.session_state.results = None
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp(prefix="maxent_")
if "r_env" not in st.session_state:
    st.session_state.r_env = None


def _guess_column_index(cols, candidates, fallback=0):
    lower_cols = [c.strip().lower() for c in cols]
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols.index(cand)
    for cand in candidates:
        for idx, col_name in enumerate(lower_cols):
            if cand in col_name:
                return idx
    return min(fallback, max(0, len(cols) - 1))


def render_upload_tab(use_wgs84):
    """Render the data upload tab"""
    st.markdown('<div class="section-header">📁 Environmental Raster Upload</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="info-box">
    <strong>Supported formats:</strong> GeoTIFF (.tif, .tiff), ESRI ASCII (.asc)<br>
    <strong>Maximum file size:</strong> 2 GB per file<br>
    <strong>Requirements:</strong> All rasters must have compatible extent, resolution, and CRS
    </div>
    """,
        unsafe_allow_html=True,
    )

    raster_files = st.file_uploader(
        "Upload Environmental Rasters",
        type=["tif", "tiff", "asc"],
        accept_multiple_files=True,
        help="Select multiple raster files (up to 2 GB per file). Hold Ctrl/Cmd to select multiple files.",
    )

    raster_dir = Path(st.session_state.temp_dir) / "rasters"
    if raster_files:
        st.success(f"✅ {len(raster_files)} raster file(s) uploaded")
        raster_info = []
        for f in raster_files:
            raster_info.append(
                {
                    "File Name": f.name,
                    "Size (MB)": round(f.size / (1024 * 1024), 2),
                    "Type": Path(f.name).suffix.lstrip(".").upper(),
                }
            )
        st.dataframe(pd.DataFrame(raster_info), width="stretch")

        # Refresh temp rasters directory so removed uploads do not linger
        if raster_dir.exists():
            shutil.rmtree(raster_dir)
        raster_dir.mkdir(parents=True, exist_ok=True)

        for f in raster_files:
            f.seek(0)
            dest_path = raster_dir / Path(f.name).name
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(f, out, length=1024 * 1024)
        st.session_state.raster_dir = str(raster_dir)
        st.session_state.raster_files = [Path(f.name).name for f in raster_files]
    else:
        st.session_state.pop("raster_dir", None)
        st.session_state.pop("raster_files", None)

    st.markdown('<div class="section-header">📍 Presence Points Upload</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="info-box">
    <strong>Supported format:</strong> CSV (.csv)<br>
    <strong>Maximum file size:</strong> 2 GB<br>
    <strong>Required columns:</strong> Species name, X/Longitude, Y/Latitude<br>
    <strong>Optional columns:</strong> Any additional attributes
    </div>
    """,
        unsafe_allow_html=True,
    )

    presence_file = st.file_uploader(
        "Upload Presence Points CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="CSV file with presence locations (up to 2 GB)",
    )

    if presence_file:
        st.success(f"✅ Presence file uploaded: {presence_file.name}")

        try:
            presence_file.seek(0)
            df_preview = pd.read_csv(presence_file, nrows=5)
            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df_preview, width="stretch")

            st.markdown("**Column Mapping:**")
            cols = df_preview.columns.tolist()
            if not cols:
                st.error("Uploaded CSV has no columns.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    species_idx = _guess_column_index(cols, ["species", "taxon", "scientific_name", "name"], 0)
                    species_col = st.selectbox("Species Column", cols, index=species_idx)
                with col2:
                    x_idx = _guess_column_index(cols, ["longitude", "lon", "long", "x", "easting"], 1)
                    x_col = st.selectbox("X / Longitude Column", cols, index=x_idx)
                with col3:
                    y_idx = _guess_column_index(cols, ["latitude", "lat", "y", "northing"], 2)
                    y_col = st.selectbox("Y / Latitude Column", cols, index=y_idx)

                presence_path = Path(st.session_state.temp_dir) / "presence.csv"
                presence_file.seek(0)
                with open(presence_path, "wb") as out:
                    shutil.copyfileobj(presence_file, out, length=1024 * 1024)
                st.session_state.presence_path = str(presence_path)
                st.session_state.presence_mapping = {
                    "species_col": species_col,
                    "x_col": x_col,
                    "y_col": y_col,
                }
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.session_state.pop("presence_path", None)
        st.session_state.pop("presence_mapping", None)

    # Coordinate system info
    st.markdown('<div class="section-header">🗺️ Coordinate System</div>', unsafe_allow_html=True)
    if use_wgs84:
        st.markdown(
            """
        <div class="info-box">
        <strong>WGS84 (EPSG:4326) Selected</strong><br>
        Presence coordinates should be in decimal degrees (Longitude, Latitude).
        Rasters will be reprojected to WGS84 if needed.
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="info-box">
        <strong>UTM (Auto-detect) Selected</strong><br>
        UTM zone will be automatically detected from raster extent.
        Rasters and geographic presence coordinates will be transformed to meters.
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_processing_tab(
    use_wgs84,
    regularization,
    feature_classes,
    max_iterations,
    convergence_threshold,
    random_seed,
    test_percentage,
    background_points,
    apply_threshold,
    use_r,
    n_jobs,
    generate_pdf,
    generate_word,
    generate_raster,
    generate_response_curves,
    generate_jackknife,
):
    """Render the processing tab"""
    st.markdown('<div class="section-header">⚙️ Model Processing</div>', unsafe_allow_html=True)

    has_rasters = "raster_dir" in st.session_state
    has_presence = "presence_path" in st.session_state

    if not has_rasters:
        st.warning("⚠️ Please upload environmental rasters first (Data Upload tab)")
    if not has_presence:
        st.warning("⚠️ Please upload presence points first (Data Upload tab)")

    if has_rasters and has_presence:
        st.success("✅ All data uploaded. Ready to process.")

        with st.expander("📋 Processing Configuration Summary", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                - **Coordinate System**: {'WGS84 (EPSG:4326)' if use_wgs84 else 'UTM (Auto)'}
                - **Regularization**: {regularization}
                - **Feature Classes**: {', '.join(feature_classes) if feature_classes else 'Linear (default)'}
                - **Max Iterations**: {max_iterations}
                - **Convergence Threshold**: {convergence_threshold}
                """
                )
            with col2:
                st.markdown(
                    f"""
                - **Random Seed**: {random_seed}
                - **Test Data %**: {test_percentage}%
                - **Background Points**: {background_points}
                - **Apply Threshold**: {apply_threshold}
                - **Use R Backend**: {use_r}
                - **Parallel Jobs**: {n_jobs}
                """
                )

        if st.button("🚀 Run MaxEnt Analysis", type="primary", width="stretch"):
            run_maxent_analysis(
                use_wgs84,
                regularization,
                feature_classes,
                max_iterations,
                convergence_threshold,
                random_seed,
                test_percentage,
                background_points,
                apply_threshold,
                use_r,
                n_jobs,
                generate_pdf,
                generate_word,
                generate_raster,
                generate_response_curves,
                generate_jackknife,
            )
    else:
        st.info("👆 Upload data in the 'Data Upload' tab to enable processing")


def run_maxent_analysis(
    use_wgs84,
    regularization,
    feature_classes,
    max_iterations,
    convergence_threshold,
    random_seed,
    test_percentage,
    background_points,
    apply_threshold,
    use_r,
    n_jobs,
    generate_pdf,
    generate_word,
    generate_raster,
    generate_response_curves,
    generate_jackknife,
):
    """Execute the MaxEnt analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()

    def update_progress(step, total, message):
        progress_bar.progress(step / total)
        status_text.text(message)
        with log_container:
            st.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    try:
        # Step 1: Load and process rasters
        update_progress(1, 8, "Loading and validating rasters...")
        raster_processor = RasterProcessor()
        raster_stack, raster_meta = raster_processor.load_rasters(
            st.session_state.raster_dir,
            target_crs="EPSG:4326" if use_wgs84 else None,
        )
        st.session_state.raster_meta = raster_meta

        # Step 2: Load and process presence points
        update_progress(2, 8, "Loading presence points...")
        presence_processor = PresenceProcessor()
        presence_data = presence_processor.load_presence(
            st.session_state.presence_path,
            st.session_state.presence_mapping,
        )

        # Step 3: Coordinate transformation
        update_progress(3, 8, "Transforming coordinates...")
        transformer = CoordinateTransformer()
        if not use_wgs84:
            utm_crs = transformer.detect_utm_zone(raster_meta["bounds"], src_crs=raster_meta["crs"])
            presence_data = transformer.transform_to_utm(presence_data, utm_crs)
            raster_stack, raster_meta = transformer.transform_rasters(raster_stack, raster_meta, utm_crs)
            raster_processor.raster_stack = raster_stack
            raster_processor.raster_meta = raster_meta
            st.session_state.raster_meta = raster_meta
            st.session_state.target_crs = str(utm_crs)
        else:
            st.session_state.target_crs = "EPSG:4326"

        # Step 4: Extract environmental values at presence points
        update_progress(4, 8, "Extracting environmental values at presence locations...")
        presence_with_env = raster_processor.extract_values(raster_stack, presence_data)

        # Step 5: Generate background points
        update_progress(5, 8, "Generating background points...")
        background_data = raster_processor.generate_background(
            raster_stack,
            n_points=background_points,
            random_seed=random_seed,
        )
        background_with_env = raster_processor.extract_values(raster_stack, background_data)

        # Step 6: Train MaxEnt model
        update_progress(6, 8, "Training MaxEnt model...")
        model = MaxEntModel(
            regularization=regularization,
            feature_classes=feature_classes,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            random_seed=random_seed,
            use_r=use_r,
            n_jobs=n_jobs,
        )
        if use_r and not model.use_r:
            with log_container:
                st.warning("R backend is unavailable or missing required packages. Running Python backend.")

        train_results = model.train(
            presence_with_env,
            background_with_env,
            test_percentage=test_percentage / 100.0,
            raster_stack=raster_stack if model.use_r else None,
            raster_meta=raster_meta if model.use_r else None,
            do_response_curves=generate_response_curves,
            do_jackknife=generate_jackknife,
        )

        # Step 7: Predict suitability across study area
        update_progress(7, 8, "Predicting habitat suitability...")
        prediction_raster = model.predict(raster_stack)

        # Step 8: Generate outputs and reports
        update_progress(8, 8, "Generating outputs, parameter exports, and reports...")

        output_dir = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        response_curves = None
        if generate_response_curves:
            response_curves = model.get_response_curves(raster_stack)

        jackknife_results = None
        if generate_jackknife:
            jackknife_results = model.jackknife_test(presence_with_env, background_with_env, raster_stack)

        threshold = None
        if apply_threshold:
            pres_preds = train_results.get("presence_train_predictions", train_results["train_predictions"])
            threshold = model.calculate_threshold(pres_preds, percentile=10)

        # Compute omission rates and memory-safe raster/prediction statistics
        pres_train_preds = np.asarray(train_results.get("presence_train_predictions", []), dtype=np.float64)
        pres_test_preds = np.asarray(train_results.get("presence_test_predictions", []), dtype=np.float64)
        omission_stats = {
            "train_omission_pct": float(np.mean(pres_train_preds < threshold) * 100.0)
            if (threshold is not None and pres_train_preds.size > 0)
            else None,
            "test_omission_pct": float(np.mean(pres_test_preds < threshold) * 100.0)
            if (threshold is not None and pres_test_preds.size > 0)
            else None,
        }

        raster_stats = raster_processor.get_raster_statistics(raster_stack)
        pred_stats = compute_prediction_stats(prediction_raster, threshold=threshold)

        if generate_raster:
            raster_processor.save_raster(prediction_raster, output_dir / "suitability.tif", raster_meta)
            if apply_threshold and threshold is not None:
                raster_processor.save_binary_raster(
                    prediction_raster, threshold, output_dir / "suitability_binary.tif", raster_meta
                )

        env_cols = raster_meta["band_names"]
        valid_presence_count = int(presence_with_env.dropna(subset=env_cols).shape[0])

        run_params = {
            "crs_mode": "WGS84 (EPSG:4326)" if use_wgs84 else "UTM (Auto-detect)",
            "target_crs": st.session_state.get("target_crs", str(raster_meta.get("crs"))),
            "regularization": regularization,
            "feature_classes": feature_classes,
            "max_iterations": max_iterations,
            "convergence_threshold": convergence_threshold,
            "random_seed": random_seed,
            "test_percentage": test_percentage,
            "background_points": background_points,
            "apply_threshold": apply_threshold,
            "use_r": use_r,
            "n_jobs": n_jobs,
            "generate_pdf": generate_pdf,
            "generate_word": generate_word,
            "generate_raster": generate_raster,
            "generate_response_curves": generate_response_curves,
            "generate_jackknife": generate_jackknife,
        }

        report_generator = ReportGenerator()
        exported_files, summary_df, coef_df = report_generator.export_data_outputs(
            output_dir=output_dir,
            run_params=run_params,
            train_results=train_results,
            pred_stats=pred_stats,
            omission_stats=omission_stats,
            raster_stats=raster_stats,
            response_curves=response_curves,
            jackknife_results=jackknife_results,
            threshold=threshold,
            raster_meta=raster_meta,
            raster_files=st.session_state.raster_files,
            presence_with_env=presence_with_env,
            n_background=background_data.shape[0],
        )

        report_paths = {}

        if generate_pdf:
            pdf_path = report_generator.generate_pdf(
                output_dir / "MaxEnt_Report.pdf",
                train_results,
                prediction_raster,
                response_curves,
                jackknife_results,
                threshold,
                raster_meta,
                st.session_state.raster_files,
                valid_presence_count,
                run_params=run_params,
                raster_stats=raster_stats,
                pred_stats=pred_stats,
                omission_stats=omission_stats,
                n_background=background_data.shape[0],
            )
            report_paths["pdf"] = str(pdf_path)

        if generate_word:
            word_path = report_generator.generate_word(
                output_dir / "MaxEnt_Report.docx",
                train_results,
                prediction_raster,
                response_curves,
                jackknife_results,
                threshold,
                raster_meta,
                st.session_state.raster_files,
                valid_presence_count,
                run_params=run_params,
                raster_stats=raster_stats,
                pred_stats=pred_stats,
                omission_stats=omission_stats,
                n_background=background_data.shape[0],
            )
            report_paths["word"] = str(word_path)

        # Standalone AUC Graph Report (always generated)
        auc_pdf_path = report_generator.generate_auc_report_pdf(
            output_dir / "AUC_Graph_Report.pdf",
            train_results=train_results,
            omission_stats=omission_stats,
            threshold=threshold,
            run_params=run_params,
            n_presence=valid_presence_count,
            n_background=background_data.shape[0],
        )
        report_paths["auc_pdf"] = str(auc_pdf_path)

        # Standalone Jackknife Analysis Report
        if generate_jackknife or jackknife_results:
            jk_pdf_path = report_generator.generate_jackknife_report_pdf(
                output_dir / "Jackknife_Analysis_Report.pdf",
                jackknife_results=jackknife_results,
                train_results=train_results,
                run_params=run_params,
                raster_files=st.session_state.raster_files,
            )
            report_paths["jackknife_pdf"] = str(jk_pdf_path)

        bundle_zip = report_generator.create_bundle_zip(output_dir)
        exported_files["bundle_zip"] = bundle_zip

        st.session_state.results = {
            "train_results": train_results,
            "roc_data": compute_roc_data(train_results),
            "prediction_raster": prediction_raster,
            "pred_stats": pred_stats,
            "omission_stats": omission_stats,
            "raster_stats": raster_stats,
            "run_params": run_params,
            "summary_df": summary_df,
            "coef_df": coef_df,
            "response_curves": response_curves,
            "jackknife_results": jackknife_results,
            "threshold": threshold,
            "raster_meta": raster_meta,
            "output_dir": str(output_dir),
            "report_paths": report_paths,
            "exported_files": exported_files,
            "presence_count": valid_presence_count,
            "background_count": background_data.shape[0],
            "raster_files": st.session_state.raster_files,
        }
        st.session_state.processing_complete = True

        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        st.success(f"🎉 MaxEnt analysis completed successfully ({train_results.get('backend', 'Python')})!")
        st.balloons()

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        import traceback

        with log_container:
            st.code(traceback.format_exc())


def render_results_tab():
    """Render the results tab"""
    st.markdown('<div class="section-header">📊 Model Results & All Parameters</div>', unsafe_allow_html=True)

    if not st.session_state.processing_complete or st.session_state.results is None:
        st.info("Run the analysis in the 'Processing' tab to see results here.")
        return

    results = st.session_state.results
    train_results = results["train_results"]
    pred_stats = results.get("pred_stats") or compute_prediction_stats(
        results["prediction_raster"], threshold=results.get("threshold")
    )
    omission_stats = results.get("omission_stats") or {}

    st.caption(f"Model Backend: **{train_results.get('backend', 'Python')}**")

    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training AUC", f"{train_results.get('train_auc', 0):.4f}")
    with col2:
        st.metric("Test AUC", f"{train_results.get('test_auc', 0):.4f}")
    with col3:
        st.metric("Presence Points", results["presence_count"])
    with col4:
        st.metric("Background Points", results["background_count"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train Correlation", f"{train_results.get('train_correlation', 0):.4f}")
    with col2:
        st.metric("Test Correlation", f"{train_results.get('test_correlation', 0):.4f}")
    with col3:
        thresh_val = results.get("threshold")
        st.metric("Threshold (10th %ile)", f"{thresh_val:.4f}" if thresh_val is not None else "N/A")
    with col4:
        suit_pct = pred_stats.get("suitable_pct")
        st.metric("Suitable Area (%)", f"{suit_pct:.2f}%" if suit_pct is not None else "N/A")

    if omission_stats.get("train_omission_pct") is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Omission Rate", f"{omission_stats['train_omission_pct']:.2f}%")
        with col2:
            test_om = omission_stats.get("test_omission_pct")
            st.metric("Test Omission Rate", f"{test_om:.2f}%" if test_om is not None else "N/A")

    # ROC / AUC Curve Graph
    roc_data = results.get("roc_data") or compute_roc_data(train_results)
    if roc_data:
        st.markdown("### 📉 ROC / AUC Curve")
        roc_fig = go.Figure()
        if "train" in roc_data:
            tr = roc_data["train"]
            roc_fig.add_trace(
                go.Scatter(
                    x=tr["fpr"],
                    y=tr["tpr"],
                    mode="lines",
                    name=f"Training ROC (AUC = {tr['auc']:.4f})",
                    line=dict(color="#1B5E20", width=2.5),
                )
            )
        if "test" in roc_data:
            te = roc_data["test"]
            roc_fig.add_trace(
                go.Scatter(
                    x=te["fpr"],
                    y=te["tpr"],
                    mode="lines",
                    name=f"Test ROC (AUC = {te['auc']:.4f})",
                    line=dict(color="#1565C0", width=2.5, dash="dashdot"),
                )
            )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Prediction (AUC = 0.5000)",
                line=dict(color="#757575", width=1.5, dash="dash"),
            )
        )
        roc_fig.update_layout(
            title="Receiver Operating Characteristic (ROC) & AUC Curve",
            xaxis_title="1 - Specificity (False Positive Rate)",
            yaxis_title="Sensitivity (True Positive Rate)",
            template="plotly_white",
            height=420,
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(roc_fig, width="stretch")

    # Complete Parameters & Metrics Table
    with st.expander("📋 All Input Parameters & Summary Metrics", expanded=True):
        if results.get("summary_df") is not None:
            st.dataframe(results["summary_df"], width="stretch", hide_index=True)

    # Fitted Model Coefficients / R Parameters
    coef_df = results.get("coef_df")
    if coef_df is not None and not coef_df.empty:
        with st.expander("🧮 Fitted Model Parameters / Feature Weights", expanded=False):
            st.dataframe(coef_df, width="stretch", hide_index=True)

    # Environmental Layer Statistics
    raster_stats = results.get("raster_stats")
    if raster_stats:
        with st.expander("🌍 Environmental Layer Statistics", expanded=False):
            rs_df = pd.DataFrame([{"Layer": k, **v} for k, v in raster_stats.items()])
            st.dataframe(rs_df, width="stretch", hide_index=True)

    # Variable importance
    st.markdown("### 📈 Variable Importance")
    if train_results.get("variable_importance"):
        vi_df = pd.DataFrame(
            [{"Variable": k, "Importance (%)": v} for k, v in train_results["variable_importance"].items()]
        ).sort_values("Importance (%)", ascending=False)

        fig = px.bar(
            vi_df,
            x="Variable",
            y="Importance (%)",
            title="Variable Importance (Permutation Importance)",
            color="Importance (%)",
            color_continuous_scale="Greens",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(vi_df, width="stretch", hide_index=True)

    # Response curves
    if results.get("response_curves"):
        st.markdown("### 📉 Response Curves")
        response_curves = results["response_curves"]
        for var_name, curve_data in response_curves.items():
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=curve_data["x"],
                    y=curve_data["y"],
                    mode="lines",
                    name=var_name,
                    line=dict(color="#2E7D32", width=2),
                )
            )
            fig.update_layout(
                title=f"Response Curve: {var_name}",
                xaxis_title=var_name,
                yaxis_title="Predicted Suitability",
                template="plotly_white",
                height=300,
            )
            st.plotly_chart(fig, width="stretch", key=f"resp_curve_{var_name}")

    # Jackknife results
    jk = results.get("jackknife_results")
    if jk:
        st.markdown("### 🔍 Jackknife Analysis")
        jk_df = pd.DataFrame(jk)
        st.dataframe(jk_df, width="stretch", hide_index=True)

        fig = px.bar(
            jk_df,
            x="variable",
            y=["gain_without", "gain_with_only"],
            title="Jackknife Test: Variable Contribution",
            barmode="group",
        )
        st.plotly_chart(fig, width="stretch")
    elif results.get("run_params", {}).get("generate_jackknife"):
        st.markdown("### 🔍 Jackknife Analysis")
        st.info("Jackknife analysis requires 2 or more environmental raster layers.")

    # Prediction raster preview (statistics + downsampled map + histogram)
    st.markdown("### 🗺️ Prediction Statistics & Map Preview")
    pred = results["prediction_raster"]

    if pred_stats["valid_pixels"] == 0:
        st.warning("No valid (non-NoData) pixels found in prediction raster.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Suitability", f"{pred_stats['min']:.4f}")
        with col2:
            st.metric("Max Suitability", f"{pred_stats['max']:.4f}")
        with col3:
            st.metric("Mean Suitability", f"{pred_stats['mean']:.4f}")
        with col4:
            st.metric("Std Dev", f"{pred_stats['std']:.4f}")

        h, w = pred.shape
        step = max(1, max(h, w) // 600)
        preview = pred[::step, ::step]
        map_fig = px.imshow(
            preview,
            color_continuous_scale="YlGn",
            title=f"Habitat Suitability Map Preview (stride {step}x)" if step > 1 else "Habitat Suitability Map",
            labels={"color": "Suitability"},
        )
        map_fig.update_xaxes(showticklabels=False)
        map_fig.update_yaxes(showticklabels=False)
        st.plotly_chart(map_fig, width="stretch")

        hist_vals = pred_stats["hist_sample"]
        fig = px.histogram(
            hist_vals,
            nbins=50,
            title="Suitability Value Distribution",
            labels={"value": "Suitability", "count": "Frequency"},
        )
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, width="stretch")


def render_reports_tab():
    """Render the reports tab"""
    st.markdown('<div class="section-header">📄 Generated Reports & Parameter Exports</div>', unsafe_allow_html=True)

    if not st.session_state.processing_complete or st.session_state.results is None:
        st.info("Run the analysis to generate reports and parameter exports.")
        return

    results = st.session_state.results
    report_paths = results.get("report_paths", {})
    exported_files = results.get("exported_files", {})
    output_dir = results.get("output_dir", "")

    if output_dir and Path(output_dir).exists():
        st.success(f"All outputs generated in: `{output_dir}`")

        # One-click bundle download for all reports + CSV/JSON parameters
        bundle_path = exported_files.get("bundle_zip")
        if bundle_path and Path(bundle_path).exists():
            with open(bundle_path, "rb") as f:
                st.download_button(
                    label="📦 Download All Parameters & Reports Bundle (.zip)",
                    data=f.read(),
                    file_name="MaxEnt_Parameters_And_Reports.zip",
                    mime="application/zip",
                    type="primary",
                    width="stretch",
                )

        col1, col2 = st.columns(2)
        with col1:
            if "pdf" in report_paths and Path(report_paths["pdf"]).exists():
                st.markdown("### 📕 PDF Report")
                with open(report_paths["pdf"], "rb") as f:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=f.read(),
                        file_name="MaxEnt_Report.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )

        with col2:
            if "word" in report_paths and Path(report_paths["word"]).exists():
                st.markdown("### 📘 Word Report")
                with open(report_paths["word"], "rb") as f:
                    st.download_button(
                        label="📥 Download Word Report",
                        data=f.read(),
                        file_name="MaxEnt_Report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        width="stretch",
                    )

        # Dedicated AUC Graph Report & Jackknife Analysis Report
        st.markdown("### 📊 Specialized Analysis Reports (AUC Graph & Jackknife)")
        ac1, ac2 = st.columns(2)
        with ac1:
            if "auc_pdf" in report_paths and Path(report_paths["auc_pdf"]).exists():
                with open(report_paths["auc_pdf"], "rb") as f:
                    st.download_button(
                        label="📉 Download AUC Graph Report (.pdf)",
                        data=f.read(),
                        file_name="AUC_Graph_Report.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )
        with ac2:
            if "jackknife_pdf" in report_paths and Path(report_paths["jackknife_pdf"]).exists():
                with open(report_paths["jackknife_pdf"], "rb") as f:
                    st.download_button(
                        label="🔍 Download Jackknife Analysis Report (.pdf)",
                        data=f.read(),
                        file_name="Jackknife_Analysis_Report.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )

        # Structured CSV / JSON Parameter Exports
        st.markdown("### 📋 Parameter, Chart & Data Exports (CSV / JSON / PNG)")
        export_buttons = [
            ("summary_csv", "📥 Parameters & Metrics Summary (.csv)", "model_parameters_summary.csv", "text/csv"),
            ("parameters_json", "📥 Full Parameters & Results (.json)", "parameters.json", "application/json"),
            ("roc_curve_csv", "📥 ROC / AUC Curve Data (.csv)", "roc_curve_data.csv", "text/csv"),
            ("auc_graph_png", "📥 AUC / ROC Graph Image (.png)", "auc_graph.png", "image/png"),
            ("jackknife_csv", "📥 Jackknife Results (.csv)", "jackknife_results.csv", "text/csv"),
            ("jackknife_png", "📥 Jackknife Chart Image (.png)", "jackknife_analysis.png", "image/png"),
            ("coefficients_csv", "📥 Model Coefficients / R Params (.csv)", "model_coefficients.csv", "text/csv"),
            ("variable_importance_csv", "📥 Variable Importance (.csv)", "variable_importance.csv", "text/csv"),
            ("response_curves_csv", "📥 Response Curves (.csv)", "response_curves.csv", "text/csv"),
            ("raster_stats_csv", "📥 Environmental Layer Stats (.csv)", "raster_statistics.csv", "text/csv"),
            ("presence_predictions_csv", "📥 Presence Point Predictions (.csv)", "presence_predictions.csv", "text/csv"),
        ]
        available_exports = [item for item in export_buttons if item[0] in exported_files and Path(exported_files[item[0]]).exists()]
        for idx in range(0, len(available_exports), 2):
            c1, c2 = st.columns(2)
            for col_widget, (key, label, fname, mime) in zip([c1, c2], available_exports[idx : idx + 2]):
                with col_widget:
                    with open(exported_files[key], "rb") as f:
                        st.download_button(
                            label=label,
                            data=f.read(),
                            file_name=fname,
                            mime=mime,
                            width="stretch",
                            key=f"dl_{key}",
                        )

        # GeoTIFF Rasters
        raster_path = Path(output_dir) / "suitability.tif"
        binary_path = Path(output_dir) / "suitability_binary.tif"
        if raster_path.exists() or binary_path.exists():
            st.markdown("### 🗺️ Suitability Rasters (GeoTIFF)")
            rc1, rc2 = st.columns(2)
            with rc1:
                if raster_path.exists():
                    size_mb = raster_path.stat().st_size / (1024 * 1024)
                    with open(raster_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download Continuous Suitability GeoTIFF ({size_mb:.1f} MB)",
                            data=f.read(),
                            file_name="suitability.tif",
                            mime="image/tiff",
                            width="stretch",
                        )
            with rc2:
                if binary_path.exists():
                    bsize_mb = binary_path.stat().st_size / (1024 * 1024)
                    with open(binary_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download Binary Threshold GeoTIFF ({bsize_mb:.1f} MB)",
                            data=f.read(),
                            file_name="suitability_binary.tif",
                            mime="image/tiff",
                            width="stretch",
                        )
    else:
        st.warning("No outputs generated. Check output options in the sidebar.")

    st.markdown("### 📁 Output Directory Contents")
    if output_dir and Path(output_dir).exists():
        files = sorted(Path(output_dir).glob("*"))
        if files:
            file_info = []
            for f in files:
                file_info.append(
                    {
                        "File": f.name,
                        "Size (KB)": round(f.stat().st_size / 1024, 2),
                        "Modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            st.dataframe(pd.DataFrame(file_info), width="stretch", hide_index=True)
        else:
            st.info("Output directory is empty")


def main():
    st.markdown('<div class="main-header">🌿 Maximum Entropy Presence Prediction</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="info-box">
    <strong>Welcome to MaxEnt Presence Prediction Application</strong><br>
    This application performs species distribution modeling using Maximum Entropy (MaxEnt) approach.
    Upload environmental rasters and presence points (up to 2 GB per file) to generate habitat suitability maps and comprehensive reports.<br>
    <small>Concept &amp; Design by K.S.Chandra, 2026, kavikondalaschandra@gmail.com. (Inspired by MaxEnt Open Source - AMNH)</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown("#### Coordinate Reference System")
        crs_option = st.radio(
            "Select coordinate system for presence points:",
            ["WGS84 (EPSG:4326)", "UTM (Auto-detect zone)"],
            help="WGS84 uses lat/lon in degrees. UTM uses meters - zone will be auto-detected from data bounds.",
        )
        use_wgs84 = crs_option == "WGS84 (EPSG:4326)"

        st.markdown("---")

        st.markdown("#### MaxEnt Parameters")
        regularization = st.slider(
            "Regularization Multiplier",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Higher values = smoother, less complex models",
        )

        feature_classes = st.multiselect(
            "Feature Classes",
            ["Linear", "Quadratic", "Product", "Threshold", "Hinge"],
            default=["Linear", "Quadratic", "Hinge"],
            help="Types of features to use in the model",
        )

        max_iterations = st.number_input(
            "Maximum Iterations",
            min_value=100,
            max_value=10000,
            value=500,
            step=100,
        )

        convergence_threshold = st.number_input(
            "Convergence Threshold",
            min_value=1e-6,
            max_value=1e-2,
            value=1e-5,
            format="%.6f",
        )

        st.markdown("---")

        st.markdown("#### Output Options")
        generate_pdf = st.checkbox("Generate PDF Report", value=True)
        generate_word = st.checkbox("Generate Word Report", value=True)
        generate_raster = st.checkbox("Generate Suitability Raster (GeoTIFF)", value=True)
        generate_response_curves = st.checkbox("Generate Response Curves", value=True)
        generate_jackknife = st.checkbox("Generate Jackknife Analysis", value=True)

        st.markdown("---")

        with st.expander("🔧 Advanced Options"):
            random_seed = st.number_input("Random Seed", value=42, min_value=0)
            test_percentage = st.slider(
                "Test Data %", 0, 50, 25, help="Percentage of presence points for testing"
            )
            background_points = st.number_input(
                "Background Points", min_value=1000, max_value=100000, value=10000, step=1000
            )
            apply_threshold = st.checkbox(
                "Apply Threshold (10th percentile training presence)", value=True
            )

            use_r = st.checkbox(
                "Use R for MaxEnt (requires 'dismo' or 'maxnet')",
                value=False,
                help="Uses R's dismo::maxent (or maxnet fallback) with raster/terra/rJava/jsonlite/pROC.",
            )

            if st.session_state.r_env is None:
                st.session_state.r_env = check_r_environment()
            r_env = st.session_state.r_env

            if r_env.get("r_available"):
                if r_env.get("dismo_ready"):
                    st.caption(f"R ready: {r_env.get('r_version')} (dismo + maxent.jar)")
                elif r_env.get("maxnet_ready"):
                    st.caption(f"R ready: {r_env.get('r_version')} (maxnet)")
                else:
                    st.caption("R found, but required R packages are missing.")
                    if st.button("Install Required R Packages", key="install_r_pkgs"):
                        with st.spinner("Installing R packages..."):
                            res = install_r_packages()
                        st.session_state.r_env = check_r_environment()
                        if res["success"]:
                            st.success("R packages installed.")
                        else:
                            st.error(f"R package installation failed:\n{res['output']}")
            else:
                st.caption("R not detected on system; Python backend will be used.")

            n_jobs = st.number_input("Parallel Jobs", min_value=1, max_value=8, value=1)

    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "⚙️ Processing", "📊 Results", "📄 Reports"])

    with tab1:
        render_upload_tab(use_wgs84)

    with tab2:
        render_processing_tab(
            use_wgs84,
            regularization,
            feature_classes,
            max_iterations,
            convergence_threshold,
            random_seed,
            test_percentage,
            background_points,
            apply_threshold,
            use_r,
            n_jobs,
            generate_pdf,
            generate_word,
            generate_raster,
            generate_response_curves,
            generate_jackknife,
        )

    with tab3:
        render_results_tab()

    with tab4:
        render_reports_tab()

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.9rem;'>"
        "MaxEnt Presence Prediction App | Built with Streamlit, Python &amp; R<br>"
        "Concept &amp; Design by K.S.Chandra, 2026, kavikondalaschandra@gmail.com. (Inspired by MaxEnt Open Source - AMNH)"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
