"""Report Generator Module
Generates comprehensive PDF, Word, CSV, and JSON outputs for MaxEnt results
"""
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from fpdf import FPDF
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_roc_data(train_results):
    """Compute Training and Test ROC curves (FPR, TPR, thresholds) and Youden's J optimal point."""
    roc_info = {}
    for split, y_key, p_key in [
        ("train", "y_train", "train_predictions"),
        ("test", "y_test", "test_predictions"),
    ]:
        y_true = train_results.get(y_key)
        y_score = train_results.get(p_key)
        if y_true is None or y_score is None:
            continue
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        valid = np.isfinite(y_score)
        y_true, y_score = y_true[valid], y_score[valid]
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        roc_info[split] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(train_results.get(f"{split}_auc", 0.0)),
            "opt_fpr": float(fpr[best_idx]),
            "opt_tpr": float(tpr[best_idx]),
            "opt_threshold": float(thresholds[best_idx]) if np.isfinite(thresholds[best_idx]) else 0.5,
        }
    return roc_info


def compute_prediction_stats(prediction_raster, threshold=None, max_hist_samples=100_000):
    """Compute exact min/max/mean/std, suitable area, and bounded histogram samples in row chunks."""
    height, width = prediction_raster.shape
    vmin = np.inf
    vmax = -np.inf
    total_sum = 0.0
    total_sq = 0.0
    valid_count = 0
    suitable_count = 0
    hist_samples = []
    rng = np.random.default_rng(42)

    for r0 in range(0, height, 1024):
        r1 = min(r0 + 1024, height)
        chunk = prediction_raster[r0:r1]
        vals = chunk[np.isfinite(chunk)]
        if vals.size == 0:
            continue
        c_min = float(np.min(vals))
        c_max = float(np.max(vals))
        if c_min < vmin:
            vmin = c_min
        if c_max > vmax:
            vmax = c_max
        vals64 = vals.astype(np.float64, copy=False)
        total_sum += float(np.sum(vals64))
        total_sq += float(np.sum(vals64 * vals64))
        valid_count += int(vals.size)
        if threshold is not None:
            suitable_count += int(np.sum(vals >= threshold))

        cap = max(500, max_hist_samples // max(1, (height // 1024) + 1))
        if vals.size > cap:
            idx = rng.choice(vals.size, size=cap, replace=False)
            hist_samples.append(vals[idx])
        else:
            hist_samples.append(vals)

    if valid_count == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "valid_pixels": 0,
            "total_pixels": int(height * width),
            "suitable_pixels": 0,
            "suitable_pct": 0.0,
            "hist_sample": np.array([], dtype=np.float32),
        }

    mean = total_sum / valid_count
    var = max(0.0, (total_sq / valid_count) - (mean * mean))
    return {
        "min": float(vmin),
        "max": float(vmax),
        "mean": float(mean),
        "std": float(np.sqrt(var)),
        "valid_pixels": valid_count,
        "total_pixels": int(height * width),
        "suitable_pixels": suitable_count if threshold is not None else None,
        "suitable_pct": (suitable_count / valid_count * 100.0) if threshold is not None else None,
        "hist_sample": np.concatenate(hist_samples) if hist_samples else np.array([], dtype=np.float32),
    }


def _safe_pdf_text(text):
    """Ensure text is latin-1 compatible for standard FPDF core fonts."""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class ReportGenerator:
    def __init__(self):
        pass

    def _build_parameter_rows(self, run_params, train_results, pred_stats, threshold, omission_stats, raster_meta, n_presence, n_background):
        """Return ordered list of (Category, Parameter, Value) rows covering every setting and metric."""
        rows = [
            ("Backend & Runtime", "Model Backend", train_results.get("backend", "Python (scikit-learn)")),
            ("Backend & Runtime", "Use R Backend", str(run_params.get("use_r", False))),
            ("Backend & Runtime", "Parallel Jobs", str(run_params.get("n_jobs", 1))),
            ("Coordinate System", "CRS Mode", run_params.get("crs_mode", "WGS84 (EPSG:4326)")),
            ("Coordinate System", "Target CRS", str(run_params.get("target_crs", raster_meta.get("crs", "N/A")))),
            ("Model Parameters", "Regularization Multiplier", str(train_results.get("regularization", run_params.get("regularization", "N/A")))),
            ("Model Parameters", "Feature Classes", ", ".join(train_results.get("feature_classes", [])) or "Linear"),
            ("Model Parameters", "Max Iterations", str(train_results.get("max_iterations", run_params.get("max_iterations", "N/A")))),
            ("Model Parameters", "Iterations Completed", str(train_results.get("n_iter") if train_results.get("n_iter") is not None else "N/A")),
            ("Model Parameters", "Convergence Threshold", str(train_results.get("convergence_threshold", run_params.get("convergence_threshold", "N/A")))),
            ("Model Parameters", "Random Seed", str(train_results.get("random_seed", run_params.get("random_seed", "N/A")))),
            ("Sampling & Split", "Test Data %", f"{run_params.get('test_percentage', 25)}%"),
            ("Sampling & Split", "Background Points Requested", str(run_params.get("background_points", n_background))),
            ("Sampling & Split", "Background Points Sampled", str(n_background)),
            ("Sampling & Split", "Valid Presence Points", str(n_presence)),
            ("Sampling & Split", "Total Training Samples (Pres+Bg)", str(train_results.get("n_train_samples", "N/A"))),
            ("Sampling & Split", "Total Test Samples (Pres+Bg)", str(train_results.get("n_test_samples", "N/A"))),
            ("Output Options", "Apply 10th Percentile Threshold", str(run_params.get("apply_threshold", True))),
            ("Output Options", "Generate PDF Report", str(run_params.get("generate_pdf", True))),
            ("Output Options", "Generate Word Report", str(run_params.get("generate_word", True))),
            ("Output Options", "Generate Suitability GeoTIFF", str(run_params.get("generate_raster", True))),
            ("Output Options", "Generate Response Curves", str(run_params.get("generate_response_curves", True))),
            ("Output Options", "Generate Jackknife Analysis", str(run_params.get("generate_jackknife", True))),
            ("Performance Metrics", "Training AUC", f"{train_results.get('train_auc', 0):.4f}"),
            ("Performance Metrics", "Test AUC", f"{train_results.get('test_auc', 0):.4f}"),
            ("Performance Metrics", "Training Correlation", f"{train_results.get('train_correlation', 0):.4f}"),
            ("Performance Metrics", "Test Correlation", f"{train_results.get('test_correlation', 0):.4f}"),
            ("Threshold & Omission", "10th Percentile Threshold", f"{threshold:.4f}" if threshold is not None else "Not Applied"),
            ("Threshold & Omission", "Training Presence Omission Rate (%)", f"{omission_stats['train_omission_pct']:.2f}%" if omission_stats.get("train_omission_pct") is not None else "N/A"),
            ("Threshold & Omission", "Test Presence Omission Rate (%)", f"{omission_stats['test_omission_pct']:.2f}%" if omission_stats.get("test_omission_pct") is not None else "N/A"),
            ("Threshold & Omission", "Predicted Suitable Area (% of valid pixels)", f"{pred_stats['suitable_pct']:.2f}%" if pred_stats.get("suitable_pct") is not None else "N/A"),
            ("Threshold & Omission", "Predicted Suitable Pixels", str(pred_stats["suitable_pixels"]) if pred_stats.get("suitable_pixels") is not None else "N/A"),
            ("Suitability Statistics", "Min Suitability", f"{pred_stats['min']:.4f}"),
            ("Suitability Statistics", "Max Suitability", f"{pred_stats['max']:.4f}"),
            ("Suitability Statistics", "Mean Suitability", f"{pred_stats['mean']:.4f}"),
            ("Suitability Statistics", "Std Dev Suitability", f"{pred_stats['std']:.4f}"),
            ("Suitability Statistics", "Valid Raster Pixels", str(pred_stats["valid_pixels"])),
            ("Suitability Statistics", "Total Raster Pixels", str(pred_stats["total_pixels"])),
            ("Raster Metadata", "Width (pixels)", str(raster_meta.get("width", "N/A"))),
            ("Raster Metadata", "Height (pixels)", str(raster_meta.get("height", "N/A"))),
            ("Raster Metadata", "Number of Bands", str(raster_meta.get("n_bands", "N/A"))),
            ("Raster Metadata", "Bounds", str(raster_meta.get("bounds", "N/A"))),
        ]
        return rows

    def _build_coefficients_df(self, train_results):
        """Build DataFrame of fitted feature coefficients (Python) or R MaxEnt results."""
        if train_results.get("r_results"):
            r_res = train_results["r_results"]
            return pd.DataFrame(
                [{"Parameter": k, "Value": v} for k, v in r_res.items()]
            )
        coefs = train_results.get("coefficients")
        feat_names = train_results.get("transformed_feature_names") or train_results.get("feature_names") or []
        if coefs is not None and len(feat_names) == len(coefs):
            rows = [{"Parameter": "(Intercept)", "Value": float(train_results.get("intercept", 0.0))}]
            for name, val in zip(feat_names, coefs):
                rows.append({"Parameter": name, "Value": float(val)})
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["Parameter", "Value"])

    def _render_charts(
        self,
        chart_dir,
        prediction_raster,
        pred_stats,
        train_results,
        response_curves,
        jackknife_results,
        threshold,
    ):
        """Render publication-ready matplotlib PNG charts into chart_dir."""
        chart_paths = {}

        # 1. Suitability Map + Histogram
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        h, w = prediction_raster.shape
        step = max(1, max(h, w) // 800)
        preview = prediction_raster[::step, ::step]
        im = axes[0].imshow(preview, cmap="YlGn", vmin=0.0, vmax=max(1.0, pred_stats["max"] or 1.0))
        axes[0].set_title("Habitat Suitability Map (Preview)")
        axes[0].set_xlabel("Column (downsampled)" if step > 1 else "Column")
        axes[0].set_ylabel("Row (downsampled)" if step > 1 else "Row")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Suitability")

        hist_vals = pred_stats.get("hist_sample", np.array([]))
        if hist_vals.size > 0:
            axes[1].hist(hist_vals, bins=40, color="#4CAF50", edgecolor="#1B5E20", alpha=0.85)
            if threshold is not None:
                axes[1].axvline(threshold, color="#D32F2F", linestyle="--", linewidth=1.8, label=f"Threshold = {threshold:.4f}")
                axes[1].legend()
        axes[1].set_title("Suitability Value Distribution")
        axes[1].set_xlabel("Predicted Suitability")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(alpha=0.25)
        plt.tight_layout()
        p_map = chart_dir / "suitability_overview.png"
        fig.savefig(p_map, dpi=150)
        plt.close(fig)
        chart_paths["suitability"] = str(p_map)

        # 2. Variable Importance
        vi = train_results.get("variable_importance") or {}
        if vi:
            items = sorted(vi.items(), key=lambda x: x[1], reverse=True)
            names = [k for k, _ in items][::-1]
            vals = [v for _, v in items][::-1]
            fig, ax = plt.subplots(figsize=(8, max(3.0, 0.45 * len(names) + 1.2)))
            ax.barh(names, vals, color="#2E7D32")
            ax.set_xlabel("Importance (%)")
            ax.set_title("Variable Importance (Permutation Importance)")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            p_vi = chart_dir / "variable_importance.png"
            fig.savefig(p_vi, dpi=150)
            plt.close(fig)
            chart_paths["variable_importance"] = str(p_vi)

        # 3. Response Curves
        if response_curves:
            n_vars = len(response_curves)
            ncols = min(2, n_vars)
            nrows = int(np.ceil(n_vars / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.3 * nrows), squeeze=False)
            for idx, (var_name, curve_data) in enumerate(response_curves.items()):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                ax.plot(curve_data["x"], curve_data["y"], color="#2E7D32", linewidth=2)
                ax.set_title(f"Response: {var_name}")
                ax.set_xlabel(var_name)
                ax.set_ylabel("Suitability")
                ax.grid(alpha=0.3)
            for idx in range(n_vars, nrows * ncols):
                r, c = divmod(idx, ncols)
                axes[r][c].axis("off")
            plt.tight_layout()
            p_rc = chart_dir / "response_curves.png"
            fig.savefig(p_rc, dpi=150)
            plt.close(fig)
            chart_paths["response_curves"] = str(p_rc)

        # 4. Jackknife
        if jackknife_results:
            p_jk = chart_dir / "jackknife_analysis.png"
            self._save_jackknife_chart(jackknife_results, p_jk)
            chart_paths["jackknife"] = str(p_jk)

        # 5. ROC / AUC Graph
        roc_data = compute_roc_data(train_results)
        if roc_data:
            p_roc = chart_dir / "auc_graph.png"
            self._save_roc_chart(roc_data, p_roc)
            chart_paths["roc_auc"] = str(p_roc)

        return chart_paths

    def _save_roc_chart(self, roc_data, out_path):
        """Save high-resolution ROC / AUC curve plot to PNG."""
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        if "train" in roc_data:
            tr = roc_data["train"]
            ax.plot(tr["fpr"], tr["tpr"], color="#1B5E20", linewidth=2.2, label=f"Training ROC (AUC = {tr['auc']:.4f})")
            ax.scatter([tr["opt_fpr"]], [tr["opt_tpr"]], color="#1B5E20", s=45, zorder=5)
        if "test" in roc_data:
            te = roc_data["test"]
            ax.plot(te["fpr"], te["tpr"], color="#1565C0", linewidth=2.2, linestyle="-.", label=f"Test ROC (AUC = {te['auc']:.4f})")
            ax.scatter([te["opt_fpr"]], [te["opt_tpr"]], color="#1565C0", s=45, zorder=5)
        ax.plot([0, 1], [0, 1], color="#757575", linestyle="--", linewidth=1.3, label="Random Prediction (AUC = 0.5000)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("1 - Specificity (False Positive Rate)")
        ax.set_ylabel("Sensitivity (True Positive Rate)")
        ax.set_title("Receiver Operating Characteristic (ROC) & AUC Graph")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return str(out_path)

    def _save_jackknife_chart(self, jackknife_results, out_path):
        """Save high-resolution Jackknife bar chart to PNG."""
        jk_df = pd.DataFrame(jackknife_results)
        y_pos = np.arange(len(jk_df))
        height_bar = 0.35
        metric_label = jk_df["metric"].iloc[0] if "metric" in jk_df.columns else "Gain / AUC"
        fig, ax = plt.subplots(figsize=(8.5, max(3.4, 0.6 * len(jk_df) + 1.5)))
        ax.barh(y_pos - height_bar / 2, jk_df["gain_without"], height=height_bar, label="Without Variable", color="#90CAF9")
        ax.barh(y_pos + height_bar / 2, jk_df["gain_with_only"], height=height_bar, label="With Only Variable", color="#2E7D32")
        full_gain = float(jk_df["full_gain"].iloc[0])
        ax.axvline(full_gain, color="#D32F2F", linestyle="--", linewidth=1.6, label=f"Full Model ({full_gain:.4f})")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(jk_df["variable"])
        ax.set_xlabel(metric_label)
        ax.set_title("Jackknife Test of Variable Contribution")
        ax.legend(loc="best", frameon=True)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return str(out_path)

    def export_data_outputs(
        self,
        output_dir,
        run_params,
        train_results,
        pred_stats,
        omission_stats,
        raster_stats,
        response_curves,
        jackknife_results,
        threshold,
        raster_meta,
        raster_files,
        presence_with_env,
        n_background,
    ):
        """Export structured JSON and CSV parameter/output files to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported = {}

        env_cols = raster_meta.get("band_names", [])
        n_presence = int(presence_with_env.dropna(subset=env_cols).shape[0]) if env_cols else len(presence_with_env)

        # 1. Flat parameter & metric summary CSV
        param_rows = self._build_parameter_rows(
            run_params, train_results, pred_stats, threshold, omission_stats, raster_meta, n_presence, n_background
        )
        summary_df = pd.DataFrame(param_rows, columns=["Category", "Parameter", "Value"])
        summary_csv = output_dir / "model_parameters_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        exported["summary_csv"] = str(summary_csv)

        # 2. Model coefficients / R MaxEnt parameters CSV
        coef_df = self._build_coefficients_df(train_results)
        if not coef_df.empty:
            coef_csv = output_dir / "model_coefficients.csv"
            coef_df.to_csv(coef_csv, index=False)
            exported["coefficients_csv"] = str(coef_csv)

        # 3. Raster layer statistics CSV
        if raster_stats:
            rs_rows = [{"Layer": k, **v} for k, v in raster_stats.items()]
            rs_csv = output_dir / "raster_statistics.csv"
            pd.DataFrame(rs_rows).to_csv(rs_csv, index=False)
            exported["raster_stats_csv"] = str(rs_csv)

        # 4. Variable importance CSV
        vi = train_results.get("variable_importance") or {}
        if vi:
            vi_df = pd.DataFrame(
                [{"Variable": k, "Permutation_Importance_Pct": float(v)} for k, v in vi.items()]
            ).sort_values("Permutation_Importance_Pct", ascending=False)
            vi_csv = output_dir / "variable_importance.csv"
            vi_df.to_csv(vi_csv, index=False)
            exported["variable_importance_csv"] = str(vi_csv)

        # 5. Response curves CSV
        if response_curves:
            rc_rows = []
            for var_name, cdata in response_curves.items():
                for x_val, y_val in zip(cdata["x"], cdata["y"]):
                    rc_rows.append({"Variable": var_name, "Value": float(x_val), "Predicted_Suitability": float(y_val)})
            rc_csv = output_dir / "response_curves.csv"
            pd.DataFrame(rc_rows).to_csv(rc_csv, index=False)
            exported["response_curves_csv"] = str(rc_csv)

        # 6. Jackknife results CSV & PNG
        if jackknife_results:
            jk_csv = output_dir / "jackknife_results.csv"
            pd.DataFrame(jackknife_results).to_csv(jk_csv, index=False)
            exported["jackknife_csv"] = str(jk_csv)
            jk_png = output_dir / "jackknife_analysis.png"
            self._save_jackknife_chart(jackknife_results, jk_png)
            exported["jackknife_png"] = str(jk_png)

        # 6b. ROC / AUC Curve CSV & PNG
        roc_data = compute_roc_data(train_results)
        if roc_data:
            roc_rows = []
            for split, rdict in roc_data.items():
                for fpr_v, tpr_v, thr_v in zip(rdict["fpr"], rdict["tpr"], rdict["thresholds"]):
                    roc_rows.append(
                        {
                            "Split": split,
                            "FPR_1_minus_Specificity": float(fpr_v),
                            "TPR_Sensitivity": float(tpr_v),
                            "Threshold": float(thr_v) if np.isfinite(thr_v) else None,
                            "AUC": rdict["auc"],
                        }
                    )
            roc_csv = output_dir / "roc_curve_data.csv"
            pd.DataFrame(roc_rows).to_csv(roc_csv, index=False)
            exported["roc_curve_csv"] = str(roc_csv)
            auc_png = output_dir / "auc_graph.png"
            self._save_roc_chart(roc_data, auc_png)
            exported["auc_graph_png"] = str(auc_png)

        # 7. Presence predictions CSV
        pres_clean = presence_with_env.dropna(subset=env_cols).copy() if env_cols else presence_with_env.copy()
        pres_preds = train_results.get("presence_all_predictions")
        if pres_preds is not None and len(pres_preds) == len(pres_clean):
            pres_clean["predicted_suitability"] = pres_preds
            if threshold is not None:
                pres_clean["suitable_binary"] = (pres_preds >= threshold).astype(int)
        pres_csv = output_dir / "presence_predictions.csv"
        pres_clean.to_csv(pres_csv, index=False)
        exported["presence_predictions_csv"] = str(pres_csv)

        # 8. Complete structured parameters.json
        payload = {
            "generated_at": datetime.now().isoformat(),
            "run_parameters": run_params,
            "raster_files": raster_files,
            "raster_metadata": {
                "crs": str(raster_meta.get("crs")),
                "width": raster_meta.get("width"),
                "height": raster_meta.get("height"),
                "bounds": list(raster_meta["bounds"]) if raster_meta.get("bounds") else None,
                "band_names": raster_meta.get("band_names", []),
                "n_bands": raster_meta.get("n_bands"),
            },
            "raster_statistics": raster_stats,
            "model_results": {
                "backend": train_results.get("backend"),
                "train_auc": train_results.get("train_auc"),
                "test_auc": train_results.get("test_auc"),
                "train_correlation": train_results.get("train_correlation"),
                "test_correlation": train_results.get("test_correlation"),
                "n_iter": train_results.get("n_iter"),
                "n_train_samples": train_results.get("n_train_samples"),
                "n_test_samples": train_results.get("n_test_samples"),
                "variable_importance": train_results.get("variable_importance"),
                "coefficients": coef_df.to_dict(orient="records") if not coef_df.empty else [],
            },
            "threshold_and_omission": {
                "threshold": threshold,
                **omission_stats,
            },
            "prediction_statistics": {k: v for k, v in pred_stats.items() if k != "hist_sample"},
            "jackknife_results": jackknife_results or [],
        }
        params_json = output_dir / "parameters.json"
        params_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        exported["parameters_json"] = str(params_json)

        return exported, summary_df, coef_df

    def create_bundle_zip(self, output_dir):
        """Bundle all CSV, JSON, PNG, PDF, and DOCX outputs in output_dir (excluding multi-hundred-MB GeoTIFFs)."""
        output_dir = Path(output_dir)
        zip_path = output_dir / "MaxEnt_Parameters_And_Reports.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(output_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in {".csv", ".json", ".png", ".pdf", ".docx"}:
                    zf.write(f, arcname=f.name)
        return str(zip_path)

    def generate_pdf(
        self,
        output_path,
        train_results,
        prediction_raster,
        response_curves,
        jackknife_results,
        threshold,
        raster_meta,
        raster_files,
        n_presence,
        run_params=None,
        raster_stats=None,
        pred_stats=None,
        omission_stats=None,
        n_background=0,
    ):
        run_params = run_params or {}
        raster_stats = raster_stats or {}
        omission_stats = omission_stats or {}
        pred_stats = pred_stats or compute_prediction_stats(prediction_raster, threshold=threshold)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 22)
        pdf.ln(24)
        pdf.cell(0, 14, "MaxEnt Presence Prediction Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(12)
        for line in [
            f"Backend: {train_results.get('backend', 'Python')}",
            f"Coordinate System: {run_params.get('crs_mode', 'WGS84 (EPSG:4326)')}",
            f"Training AUC: {train_results.get('train_auc', 0):.4f}   |   Test AUC: {train_results.get('test_auc', 0):.4f}",
            f"Presence Points: {n_presence}   |   Background Points: {n_background}",
            f"Environmental Layers: {len(raster_files)}",
        ]:
            pdf.cell(0, 8, _safe_pdf_text(line), align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.cell(
            0,
            6,
            _safe_pdf_text("Concept & Design by K.S.Chandra, 2026, kavikondalaschandra@gmail.com. (Inspired by MaxEnt Open Source - AMNH)"),
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )

        # Complete Parameters & Metrics Table
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 10, "All Input Parameters & Model Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        param_rows = self._build_parameter_rows(
            run_params, train_results, pred_stats, threshold, omission_stats, raster_meta, n_presence, n_background
        )
        pdf.set_font("Helvetica", "", 9)
        for cat, key, val in param_rows:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, _safe_pdf_text(f"[{cat}] {key}: {val}"), new_x="LMARGIN", new_y="NEXT")

        # Model Coefficients / R Parameters
        coef_df = self._build_coefficients_df(train_results)
        if not coef_df.empty:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 15)
            pdf.cell(0, 10, "Fitted Model Parameters / Coefficients", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 9)
            for _, row in coef_df.iterrows():
                pdf.set_x(pdf.l_margin)
                val = row["Value"]
                val_str = f"{val:.6g}" if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
                pdf.multi_cell(0, 5.5, _safe_pdf_text(f"{row['Parameter']}: {val_str}"), new_x="LMARGIN", new_y="NEXT")

        # Environmental Layer Statistics
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 10, "Environmental Layers & Statistics", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 9.5)
        for i, fname in enumerate(raster_files, 1):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, _safe_pdf_text(f"{i}. {fname}"), new_x="LMARGIN", new_y="NEXT")
        if raster_stats:
            pdf.ln(3)
            for layer, st_dict in raster_stats.items():
                line = (
                    f"{layer}: min={st_dict.get('min', float('nan')):.4f}, "
                    f"max={st_dict.get('max', float('nan')):.4f}, "
                    f"mean={st_dict.get('mean', float('nan')):.4f}, "
                    f"std={st_dict.get('std', float('nan')):.4f}, "
                    f"median={st_dict.get('median', float('nan')):.4f}"
                )
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5.5, _safe_pdf_text(line), new_x="LMARGIN", new_y="NEXT")

        # Render charts and embed them
        with tempfile.TemporaryDirectory(prefix="maxent_charts_") as cdir:
            charts = self._render_charts(
                Path(cdir),
                prediction_raster,
                pred_stats,
                train_results,
                response_curves,
                jackknife_results,
                threshold,
            )

            if "suitability" in charts:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 15)
                pdf.cell(0, 10, "Habitat Suitability Map & Distribution", new_x="LMARGIN", new_y="NEXT")
                pdf.image(charts["suitability"], x=10, w=190)

            if "roc_auc" in charts:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 15)
                pdf.cell(0, 10, "ROC & AUC Performance Curve", new_x="LMARGIN", new_y="NEXT")
                pdf.image(charts["roc_auc"], x=20, w=170)

            if train_results.get("variable_importance"):
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 15)
                pdf.cell(0, 10, "Variable Importance (Permutation Importance)", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9.5)
                for var, imp in sorted(train_results["variable_importance"].items(), key=lambda x: x[1], reverse=True):
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 5.5, _safe_pdf_text(f"{var}: {imp:.2f}%"), new_x="LMARGIN", new_y="NEXT")
                if "variable_importance" in charts:
                    pdf.ln(3)
                    pdf.image(charts["variable_importance"], x=15, w=180)

            if "response_curves" in charts:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 15)
                pdf.cell(0, 10, "Response Curves", new_x="LMARGIN", new_y="NEXT")
                pdf.image(charts["response_curves"], x=10, w=190)

            if jackknife_results:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 15)
                pdf.cell(0, 10, "Jackknife Analysis", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9.5)
                for row in jackknife_results:
                    line = (
                        f"{row.get('variable', '')}: "
                        f"Without = {row.get('gain_without', 0):.4f}, "
                        f"With Only = {row.get('gain_with_only', 0):.4f}, "
                        f"Full = {row.get('full_gain', 0):.4f}"
                    )
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 5.5, _safe_pdf_text(line), new_x="LMARGIN", new_y="NEXT")
                if "jackknife" in charts:
                    pdf.ln(3)
                    pdf.image(charts["jackknife"], x=15, w=180)
            elif run_params.get("generate_jackknife"):
                pdf.ln(4)
                pdf.set_font("Helvetica", "I", 9.5)
                pdf.multi_cell(0, 5.5, "Jackknife Analysis: Requires 2 or more environmental layers.", new_x="LMARGIN", new_y="NEXT")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(output_path))
        return str(output_path)

    def generate_word(
        self,
        output_path,
        train_results,
        prediction_raster,
        response_curves,
        jackknife_results,
        threshold,
        raster_meta,
        raster_files,
        n_presence,
        run_params=None,
        raster_stats=None,
        pred_stats=None,
        omission_stats=None,
        n_background=0,
    ):
        run_params = run_params or {}
        raster_stats = raster_stats or {}
        omission_stats = omission_stats or {}
        pred_stats = pred_stats or compute_prediction_stats(prediction_raster, threshold=threshold)

        doc = Document()

        title = doc.add_heading("MaxEnt Presence Prediction Report", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        date_para = doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_run = date_para.add_run(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        date_run.font.size = Pt(11)

        credit_para = doc.add_paragraph()
        credit_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        credit_run = credit_para.add_run(
            "Concept & Design by K.S.Chandra, 2026, kavikondalaschandra@gmail.com. (Inspired by MaxEnt Open Source - AMNH)"
        )
        credit_run.font.size = Pt(9.5)
        credit_run.italic = True

        # Complete Parameters & Metrics Table
        doc.add_heading("All Input Parameters & Model Metrics", level=1)
        param_rows = self._build_parameter_rows(
            run_params, train_results, pred_stats, threshold, omission_stats, raster_meta, n_presence, n_background
        )
        p_table = doc.add_table(rows=len(param_rows) + 1, cols=3, style="Table Grid")
        p_table.cell(0, 0).text = "Category"
        p_table.cell(0, 1).text = "Parameter"
        p_table.cell(0, 2).text = "Value"
        for i, (cat, key, val) in enumerate(param_rows, 1):
            p_table.cell(i, 0).text = str(cat)
            p_table.cell(i, 1).text = str(key)
            p_table.cell(i, 2).text = str(val)

        # Model Coefficients / R MaxEnt Parameters
        coef_df = self._build_coefficients_df(train_results)
        if not coef_df.empty:
            doc.add_heading("Fitted Model Parameters / Coefficients", level=1)
            c_table = doc.add_table(rows=len(coef_df) + 1, cols=2, style="Table Grid")
            c_table.cell(0, 0).text = "Parameter"
            c_table.cell(0, 1).text = "Value"
            for i, (_, row) in enumerate(coef_df.iterrows(), 1):
                val = row["Value"]
                c_table.cell(i, 0).text = str(row["Parameter"])
                c_table.cell(i, 1).text = f"{val:.6g}" if isinstance(val, (int, float, np.floating, np.integer)) else str(val)

        # Environmental Layer Statistics
        doc.add_heading("Environmental Layers & Statistics", level=1)
        for i, fname in enumerate(raster_files, 1):
            doc.add_paragraph(f"{i}. {fname}", style="List Number")
        if raster_stats:
            rs_table = doc.add_table(rows=len(raster_stats) + 1, cols=6, style="Table Grid")
            for idx, hdr in enumerate(["Layer", "Min", "Max", "Mean", "Std Dev", "Median"]):
                rs_table.cell(0, idx).text = hdr
            for i, (layer, st_dict) in enumerate(raster_stats.items(), 1):
                rs_table.cell(i, 0).text = str(layer)
                rs_table.cell(i, 1).text = f"{st_dict.get('min', float('nan')):.4f}"
                rs_table.cell(i, 2).text = f"{st_dict.get('max', float('nan')):.4f}"
                rs_table.cell(i, 3).text = f"{st_dict.get('mean', float('nan')):.4f}"
                rs_table.cell(i, 4).text = f"{st_dict.get('std', float('nan')):.4f}"
                rs_table.cell(i, 5).text = f"{st_dict.get('median', float('nan')):.4f}"

        # Charts & Variable Importance / Jackknife
        with tempfile.TemporaryDirectory(prefix="maxent_charts_") as cdir:
            charts = self._render_charts(
                Path(cdir),
                prediction_raster,
                pred_stats,
                train_results,
                response_curves,
                jackknife_results,
                threshold,
            )

            if "suitability" in charts:
                doc.add_heading("Habitat Suitability Map & Distribution", level=1)
                doc.add_picture(charts["suitability"], width=Inches(6.4))

            if "roc_auc" in charts:
                doc.add_heading("ROC & AUC Performance Curve", level=1)
                doc.add_picture(charts["roc_auc"], width=Inches(5.8))

            if train_results.get("variable_importance"):
                doc.add_heading("Variable Importance", level=1)
                vi = train_results["variable_importance"]
                vi_table = doc.add_table(rows=len(vi) + 1, cols=2, style="Table Grid")
                vi_table.cell(0, 0).text = "Variable"
                vi_table.cell(0, 1).text = "Importance (%)"
                for i, (var, imp) in enumerate(sorted(vi.items(), key=lambda x: x[1], reverse=True), 1):
                    vi_table.cell(i, 0).text = var
                    vi_table.cell(i, 1).text = f"{imp:.2f}"
                if "variable_importance" in charts:
                    doc.add_paragraph()
                    doc.add_picture(charts["variable_importance"], width=Inches(6.0))

            if "response_curves" in charts:
                doc.add_heading("Response Curves", level=1)
                doc.add_picture(charts["response_curves"], width=Inches(6.4))

            if jackknife_results:
                doc.add_heading("Jackknife Analysis", level=1)
                jk_table = doc.add_table(rows=len(jackknife_results) + 1, cols=4, style="Table Grid")
                jk_table.cell(0, 0).text = "Variable"
                jk_table.cell(0, 1).text = "Without Variable"
                jk_table.cell(0, 2).text = "With Only Variable"
                jk_table.cell(0, 3).text = "Full Model"
                for i, row in enumerate(jackknife_results, 1):
                    jk_table.cell(i, 0).text = str(row.get("variable", ""))
                    jk_table.cell(i, 1).text = f"{row.get('gain_without', 0):.4f}"
                    jk_table.cell(i, 2).text = f"{row.get('gain_with_only', 0):.4f}"
                    jk_table.cell(i, 3).text = f"{row.get('full_gain', 0):.4f}"
                if "jackknife" in charts:
                    doc.add_paragraph()
                    doc.add_picture(charts["jackknife"], width=Inches(6.0))
            elif run_params.get("generate_jackknife"):
                doc.add_heading("Jackknife Analysis", level=1)
                doc.add_paragraph("Jackknife Analysis requires 2 or more environmental layers.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return str(output_path)

    def generate_auc_report_pdf(
        self,
        output_path,
        train_results,
        omission_stats=None,
        threshold=None,
        run_params=None,
        n_presence=0,
        n_background=0,
    ):
        """Generate standalone AUC / ROC Graph PDF report."""
        omission_stats = omission_stats or {}
        run_params = run_params or {}
        roc_data = compute_roc_data(train_results)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "MaxEnt ROC & AUC Graph Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0,
            6,
            _safe_pdf_text(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Backend: {train_results.get('backend', 'Python')}"
            ),
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.cell(
            0,
            5,
            _safe_pdf_text("Concept & Design by K.S.Chandra, 2026, kavikondalaschandra@gmail.com. (Inspired by MaxEnt Open Source - AMNH)"),
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(4)

        with tempfile.TemporaryDirectory(prefix="maxent_auc_") as cdir:
            if roc_data:
                chart_path = Path(cdir) / "auc_graph.png"
                self._save_roc_chart(roc_data, chart_path)
                pdf.image(str(chart_path), x=20, w=170)
                pdf.ln(4)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "ROC / AUC Summary Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        lines = [
            f"Training AUC: {train_results.get('train_auc', 0):.4f}",
            f"Test AUC: {train_results.get('test_auc', 0):.4f}",
            f"Training Correlation: {train_results.get('train_correlation', 0):.4f}",
            f"Test Correlation: {train_results.get('test_correlation', 0):.4f}",
            f"Valid Presence Points: {n_presence}  |  Background Points: {n_background}",
            f"Test Data Split: {run_params.get('test_percentage', 25)}%",
            f"Regularization Multiplier: {train_results.get('regularization', run_params.get('regularization', 'N/A'))}",
            f"Feature Classes: {', '.join(train_results.get('feature_classes', [])) or 'Linear'}",
        ]
        if threshold is not None:
            lines.append(f"10th Percentile Training Presence Threshold: {threshold:.4f}")
        if omission_stats.get("train_omission_pct") is not None:
            lines.append(f"Training Omission Rate: {omission_stats['train_omission_pct']:.2f}%")
        if omission_stats.get("test_omission_pct") is not None:
            lines.append(f"Test Omission Rate: {omission_stats['test_omission_pct']:.2f}%")

        for split_label, key in [("Training", "train"), ("Test", "test")]:
            if key in roc_data:
                r = roc_data[key]
                lines.append(
                    f"{split_label} Optimal Cutoff (Youden J): threshold={r['opt_threshold']:.4f}, "
                    f"Sensitivity={r['opt_tpr']:.4f}, Specificity={1.0 - r['opt_fpr']:.4f}"
                )

        for ln in lines:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 6, _safe_pdf_text(ln), new_x="LMARGIN", new_y="NEXT")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(output_path))
        return str(output_path)

    def generate_jackknife_report_pdf(
        self,
        output_path,
        jackknife_results,
        train_results,
        run_params=None,
        raster_files=None,
    ):
        """Generate standalone Jackknife Analysis PDF report."""
        run_params = run_params or {}
        raster_files = raster_files or []

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "MaxEnt Jackknife Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0,
            6,
            _safe_pdf_text(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Backend: {train_results.get('backend', 'Python')}"
            ),
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.cell(
            0,
            5,
            _safe_pdf_text("Concept & Design by K.S.Chandra, 2026, kavikondalaschandra@gmail.com. (Inspired by MaxEnt Open Source - AMNH)"),
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(4)

        if jackknife_results:
            with tempfile.TemporaryDirectory(prefix="maxent_jk_") as cdir:
                jk_path = Path(cdir) / "jackknife_analysis.png"
                self._save_jackknife_chart(jackknife_results, jk_path)
                pdf.image(str(jk_path), x=15, w=180)
                pdf.ln(4)

            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "Jackknife Test Results", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9.5)
            vi = train_results.get("variable_importance") or {}
            for row in jackknife_results:
                var = row.get("variable", "")
                wo = float(row.get("gain_without", 0))
                only = float(row.get("gain_with_only", 0))
                full = float(row.get("full_gain", 0))
                loss = full - wo
                metric_nm = row.get("metric", "Gain / AUC")
                imp_pct = vi.get(var)
                imp_str = f", Perm Importance={imp_pct:.2f}%" if imp_pct is not None else ""
                line = (
                    f"{var} [{metric_nm}]: Without={wo:.4f}, With Only={only:.4f}, "
                    f"Full={full:.4f}, Unique Loss={loss:.4f}{imp_str}"
                )
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 6, _safe_pdf_text(line), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 10.5)
            pdf.multi_cell(0, 6, "No Jackknife results were generated for this run.", new_x="LMARGIN", new_y="NEXT")

        if raster_files:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 7, "Environmental Layers Evaluated", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9.5)
            for idx, fname in enumerate(raster_files, 1):
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5.5, _safe_pdf_text(f"{idx}. {fname}"), new_x="LMARGIN", new_y="NEXT")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(output_path))
        return str(output_path)
