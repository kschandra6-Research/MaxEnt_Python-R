"""
MaxEnt Core Model Implementation
Supports both Python (scikit-learn) and R (dismo / maxnet) backends
"""
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from utils.r_bridge import PROJECT_R_LIB, check_r_environment, run_r_maxent
except ImportError:
    try:
        from r_bridge import PROJECT_R_LIB, check_r_environment, run_r_maxent
    except ImportError:
        PROJECT_R_LIB = None
        check_r_environment = None
        run_r_maxent = None


def _safe_auc(y_true, y_pred):
    """Compute ROC AUC safely even if only one class or constant predictions exist."""
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def _safe_corr(y_true, y_pred):
    """Compute Pearson correlation safely without NaN when variance is zero."""
    if len(np.unique(y_true)) < 2 or np.std(y_pred) == 0:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


class MaxEntModel:
    """Maximum Entropy model for species distribution modeling"""

    def __init__(
        self,
        regularization=1.0,
        feature_classes=None,
        max_iterations=500,
        convergence_threshold=1e-5,
        random_seed=42,
        use_r=False,
        n_jobs=1,
    ):
        self.regularization = float(regularization)
        self.feature_classes = list(feature_classes) if feature_classes else ["Linear", "Quadratic", "Hinge"]
        self.max_iterations = int(max_iterations)
        self.convergence_threshold = float(convergence_threshold)
        self.random_seed = int(random_seed)
        self.use_r = bool(use_r)
        self.n_jobs = int(n_jobs)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self._threshold_knots = {}
        self._hinge_knots = {}
        self._r_env = None
        self._r_prediction_raster = None
        self._r_response_curves = None
        self._r_jackknife_results = None

        if self.use_r:
            self._init_r()

    def _init_r(self):
        """Check R environment availability."""
        if check_r_environment is None:
            self.use_r = False
            return
        self._r_env = check_r_environment()
        if not (
            self._r_env.get("r_available")
            and (self._r_env.get("dismo_ready") or self._r_env.get("maxnet_ready"))
        ):
            self.use_r = False

    def _fit_feature_params(self, X_raw):
        """Fit threshold and hinge quartile knots on training data."""
        n_features = X_raw.shape[1]
        self._threshold_knots = {}
        self._hinge_knots = {}
        quantiles = (0.25, 0.5, 0.75)

        if "Threshold" in self.feature_classes:
            for i in range(n_features):
                self._threshold_knots[i] = [float(np.quantile(X_raw[:, i], q)) for q in quantiles]

        if "Hinge" in self.feature_classes:
            for i in range(n_features):
                self._hinge_knots[i] = [float(np.quantile(X_raw[:, i], q)) for q in quantiles]

    def _transform_features(self, X_raw):
        """Transform raw environmental matrix into configured MaxEnt feature classes."""
        n_samples, n_features = X_raw.shape
        base_names = (
            list(self.feature_names)
            if self.feature_names and len(self.feature_names) == n_features
            else [f"var_{i}" for i in range(n_features)]
        )
        feature_matrix = []
        feature_names = []

        classes = self.feature_classes if self.feature_classes else ["Linear"]
        if "Linear" in classes or not any(
            c in classes for c in ("Linear", "Quadratic", "Product", "Threshold", "Hinge")
        ):
            feature_matrix.append(X_raw)
            feature_names.extend(base_names)

        if "Quadratic" in classes:
            feature_matrix.append(X_raw ** 2)
            feature_names.extend([f"{base_names[i]}^2" for i in range(n_features)])

        if "Product" in classes and n_features > 1:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    prod = (X_raw[:, i] * X_raw[:, j]).reshape(-1, 1)
                    feature_matrix.append(prod)
                    feature_names.append(f"{base_names[i]}*{base_names[j]}")

        if "Threshold" in classes:
            quantiles = (0.25, 0.5, 0.75)
            for i in range(n_features):
                knots = self._threshold_knots.get(i) or [float(np.quantile(X_raw[:, i], q)) for q in quantiles]
                for q, thresh in zip(quantiles, knots):
                    feat = (X_raw[:, i] > thresh).astype(float).reshape(-1, 1)
                    feature_matrix.append(feat)
                    feature_names.append(f"{base_names[i]}>q{int(q * 100)}({thresh:.4g})")

        if "Hinge" in classes:
            quantiles = (0.25, 0.5, 0.75)
            for i in range(n_features):
                knots = self._hinge_knots.get(i) or [float(np.quantile(X_raw[:, i], q)) for q in quantiles]
                for q, hinge_pt in zip(quantiles, knots):
                    feat = np.maximum(X_raw[:, i] - hinge_pt, 0.0).reshape(-1, 1)
                    feature_matrix.append(feat)
                    feature_names.append(f"max({base_names[i]}-q{int(q * 100)}({hinge_pt:.4g}),0)")

        return np.hstack(feature_matrix), feature_names

    def _create_features(self, X):
        """Backward-compatible wrapper that fits knots if unset and transforms."""
        if not self._threshold_knots and not self._hinge_knots:
            self._fit_feature_params(X)
        return self._transform_features(X)

    def train(
        self,
        presence_data,
        background_data,
        test_percentage=0.25,
        raster_stack=None,
        raster_meta=None,
        do_response_curves=True,
        do_jackknife=True,
    ):
        """
        Train the MaxEnt model using R (dismo/maxnet) or Python (scikit-learn).
        """
        env_cols = [
            c
            for c in presence_data.columns
            if c not in ["species", "x", "y", "longitude", "latitude"]
        ]
        if not env_cols:
            raise ValueError("No environmental variable columns found in presence data.")
        self.feature_names = env_cols

        pres_clean = presence_data.dropna(subset=env_cols)
        back_clean = background_data.dropna(subset=env_cols)

        if pres_clean.empty:
            raise ValueError(
                "No presence points fall within valid (non-NoData) raster pixels. "
                "Check coordinate system selection (WGS84 vs UTM) and raster extent."
            )
        if back_clean.empty:
            raise ValueError("No valid background points available for training.")

        X_presence = pres_clean[env_cols].values.astype(np.float64)
        X_background = back_clean[env_cols].values.astype(np.float64)

        X_raw = np.vstack([X_presence, X_background])
        y = np.hstack([np.ones(len(X_presence)), np.zeros(len(X_background))])

        if test_percentage and float(test_percentage) > 0.0:
            test_frac = min(max(float(test_percentage), 0.01), 0.9)
            try:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_raw, y, test_size=test_frac, random_state=self.random_seed, stratify=y
                )
            except ValueError:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_raw, y, test_size=test_frac, random_state=self.random_seed
                )
        else:
            X_train_raw, X_test_raw = X_raw, X_raw
            y_train, y_test = y, y

        if self.use_r and run_r_maxent is not None:
            try:
                return self._train_r(
                    X_train_raw,
                    y_train,
                    X_test_raw,
                    y_test,
                    X_presence,
                    env_cols,
                    has_test=bool(test_percentage and float(test_percentage) > 0.0),
                    raster_stack=raster_stack,
                    raster_meta=raster_meta,
                    do_response_curves=do_response_curves,
                    do_jackknife=do_jackknife,
                )
            except Exception as exc:
                print(f"R training failed: {exc}. Falling back to Python backend.")
                self.use_r = False

        # Fit feature knots & scaler strictly on training split
        self._fit_feature_params(X_train_raw)
        X_train_feat, feat_names = self._transform_features(X_train_raw)
        X_test_feat, _ = self._transform_features(X_test_raw)

        X_train_scaled = self.scaler.fit_transform(X_train_feat)
        X_test_scaled = self.scaler.transform(X_test_feat)

        return self._train_python(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            X_test_raw,
            X_presence,
            feat_names,
        )

    def _train_python(
        self, X_train_scaled, X_test_scaled, y_train, y_test, X_test_raw, X_presence, feature_names
    ):
        """Train using Python (scikit-learn) backend."""
        C = 1.0 / self.regularization if self.regularization > 0 else 1e6

        self.model = LogisticRegression(
            C=C,
            max_iter=self.max_iterations,
            tol=self.convergence_threshold,
            random_state=self.random_seed,
            solver="lbfgs",
            class_weight="balanced",
        )
        self.model.fit(X_train_scaled, y_train)

        train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        test_pred = self.model.predict_proba(X_test_scaled)[:, 1]
        pres_all_pred = self.predict_points(X_presence)

        train_auc = _safe_auc(y_train, train_pred)
        test_auc = _safe_auc(y_test, test_pred)

        var_importance = self._calculate_permutation_importance(X_test_raw, y_test)
        train_corr = _safe_corr(y_train, train_pred)
        test_corr = _safe_corr(y_test, test_pred)

        return {
            "backend": "Python (scikit-learn)",
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_correlation": train_corr,
            "test_correlation": test_corr,
            "train_predictions": train_pred,
            "test_predictions": test_pred,
            "y_train": y_train.astype(int),
            "y_test": y_test.astype(int),
            "presence_train_predictions": train_pred[y_train == 1],
            "presence_test_predictions": test_pred[y_test == 1],
            "presence_all_predictions": pres_all_pred,
            "variable_importance": var_importance,
            "coefficients": self.model.coef_[0] if hasattr(self.model, "coef_") else None,
            "intercept": float(self.model.intercept_[0]) if hasattr(self.model, "intercept_") else None,
            "transformed_feature_names": feature_names,
            "feature_names": list(self.feature_names),
            "r_results": None,
            "n_train_samples": int(len(y_train)),
            "n_test_samples": int(len(y_test)),
            "n_iter": int(self.model.n_iter_[0]) if hasattr(self.model, "n_iter_") else None,
            "regularization": self.regularization,
            "feature_classes": self.feature_classes,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "random_seed": self.random_seed,
        }

    def predict_points(self, X_raw):
        """Predict suitability probabilities for a 2D array of raw environmental values."""
        if X_raw is None or len(X_raw) == 0 or self.model is None:
            return np.array([], dtype=np.float64)
        feat, _ = self._transform_features(np.asarray(X_raw, dtype=np.float64))
        scaled = self.scaler.transform(feat)
        return self.model.predict_proba(scaled)[:, 1]

    def _train_r(
        self,
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        X_presence,
        env_cols,
        has_test=True,
        raster_stack=None,
        raster_meta=None,
        do_response_curves=True,
        do_jackknife=True,
    ):
        """Train and optionally predict using R (dismo::maxent / maxnet) via Rscript."""
        self._r_prediction_raster = None
        self._r_response_curves = None
        self._r_jackknife_results = None

        with tempfile.TemporaryDirectory(prefix="maxent_r_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            train_df = pd.DataFrame(X_train_raw, columns=env_cols)
            train_df["pa"] = y_train.astype(int)
            train_csv = tmp_dir / "train.csv"
            train_df.to_csv(train_csv, index=False)

            pres_df = pd.DataFrame(X_presence, columns=env_cols)
            presence_csv = tmp_dir / "presence_all.csv"
            pres_df.to_csv(presence_csv, index=False)

            test_csv = ""
            if has_test:
                test_df = pd.DataFrame(X_test_raw, columns=env_cols)
                test_df["pa"] = y_test.astype(int)
                test_csv_path = tmp_dir / "test.csv"
                test_df.to_csv(test_csv_path, index=False)
                test_csv = str(test_csv_path)

            raster_stack_tif = ""
            prediction_tif = ""
            if raster_stack is not None and raster_meta is not None:
                stack_path = tmp_dir / "raster_stack.tif"
                pred_path = tmp_dir / "r_prediction.tif"
                data = raster_stack["data"]
                profile = {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "width": raster_meta["width"],
                    "height": raster_meta["height"],
                    "count": data.shape[0],
                    "crs": raster_meta["crs"],
                    "transform": raster_meta["transform"],
                    "nodata": -9999.0,
                    "compress": "lzw",
                }
                with rasterio.open(stack_path, "w", **profile) as dst:
                    for i in range(data.shape[0]):
                        band = data[i].astype(np.float32, copy=True)
                        band[np.isnan(band)] = -9999.0
                        dst.write(band, i + 1)
                raster_stack_tif = str(stack_path)
                prediction_tif = str(pred_path)

            result_json = tmp_dir / "r_result.json"
            cfg_path = tmp_dir / "config.json"
            cfg = {
                "project_r_lib": str(PROJECT_R_LIB) if PROJECT_R_LIB else "",
                "work_dir": str(tmp_dir),
                "train_csv": str(train_csv),
                "test_csv": test_csv,
                "presence_csv": str(presence_csv),
                "env_cols": env_cols,
                "feature_classes": self.feature_classes,
                "regularization": self.regularization,
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold,
                "random_seed": self.random_seed,
                "do_response_curves": bool(do_response_curves),
                "do_jackknife": bool(do_jackknife),
                "raster_stack_tif": raster_stack_tif,
                "prediction_tif": prediction_tif,
                "result_json": str(result_json),
            }
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

            run_r_maxent(cfg_path)

            if not result_json.is_file():
                raise RuntimeError("R script did not produce result JSON.")

            r_out = json.loads(result_json.read_text(encoding="utf-8"))

            if prediction_tif and Path(prediction_tif).is_file():
                with rasterio.open(prediction_tif) as src:
                    pred_arr = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        pred_arr[pred_arr == np.float32(src.nodata)] = np.nan
                    if isinstance(raster_stack, dict) and "mask" in raster_stack:
                        pred_arr[raster_stack["mask"]] = np.nan
                self._r_prediction_raster = pred_arr

            if r_out.get("response_curves"):
                self._r_response_curves = {
                    k: {"x": np.asarray(v["x"]), "y": np.asarray(v["y"])}
                    for k, v in r_out["response_curves"].items()
                }

            if r_out.get("jackknife_results"):
                self._r_jackknife_results = r_out["jackknife_results"]

            train_pred = np.asarray(r_out["train_predictions"], dtype=np.float64)
            test_pred = np.asarray(r_out["test_predictions"], dtype=np.float64)
            pres_pred = np.asarray(r_out["presence_train_predictions"], dtype=np.float64)
            pres_test_pred = np.asarray(r_out.get("presence_test_predictions", []), dtype=np.float64)
            pres_all_pred = np.asarray(r_out.get("presence_all_predictions", []), dtype=np.float64)

            return {
                "backend": r_out.get("backend", "R (dismo)"),
                "train_auc": float(r_out["train_auc"]),
                "test_auc": float(r_out["test_auc"]),
                "train_correlation": float(r_out["train_correlation"]),
                "test_correlation": float(r_out["test_correlation"]),
                "train_predictions": train_pred,
                "test_predictions": test_pred,
                "y_train": np.asarray(r_out.get("y_train", y_train), dtype=int),
                "y_test": np.asarray(r_out.get("y_test", y_test), dtype=int),
                "presence_train_predictions": pres_pred,
                "presence_test_predictions": pres_test_pred,
                "presence_all_predictions": pres_all_pred,
                "variable_importance": r_out.get("variable_importance", {}),
                "coefficients": None,
                "intercept": None,
                "transformed_feature_names": env_cols,
                "feature_names": env_cols,
                "r_results": r_out.get("r_results", {}),
                "n_train_samples": int(len(y_train)),
                "n_test_samples": int(len(y_test)),
                "n_iter": int(r_out["r_results"]["Iterations"])
                if isinstance(r_out.get("r_results"), dict) and "Iterations" in r_out["r_results"]
                else None,
                "regularization": self.regularization,
                "feature_classes": self.feature_classes,
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold,
                "random_seed": self.random_seed,
            }

    def _calculate_permutation_importance(self, X_test_raw, y_test):
        """Calculate permutation importance by permuting each raw environmental variable."""
        X_base_feat, _ = self._transform_features(X_test_raw)
        X_base_scaled = self.scaler.transform(X_base_feat)
        baseline_auc = _safe_auc(y_test, self.model.predict_proba(X_base_scaled)[:, 1])

        rng = np.random.default_rng(self.random_seed)
        importance = {}

        for i, orig_name in enumerate(self.feature_names):
            if baseline_auc <= 0:
                importance[orig_name] = 0.0
                continue
            X_perm_raw = X_test_raw.copy()
            X_perm_raw[:, i] = rng.permutation(X_perm_raw[:, i])
            X_perm_feat, _ = self._transform_features(X_perm_raw)
            X_perm_scaled = self.scaler.transform(X_perm_feat)
            permuted_auc = _safe_auc(y_test, self.model.predict_proba(X_perm_scaled)[:, 1])
            importance[orig_name] = max(0.0, (baseline_auc - permuted_auc) / baseline_auc * 100.0)

        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total * 100.0 for k, v in importance.items()}
        elif len(self.feature_names) > 0:
            equal = 100.0 / len(self.feature_names)
            importance = {k: equal for k in self.feature_names}

        return importance

    def predict(self, raster_stack, chunk_size=500_000):
        """
        Predict suitability across raster stack in memory-safe chunks.

        Args:
            raster_stack: 3D array (bands, height, width) or dict with 'data' and 'mask'
            chunk_size: Number of valid pixels per prediction batch

        Returns:
            2D prediction array
        """
        if self._r_prediction_raster is not None:
            return self._r_prediction_raster

        if isinstance(raster_stack, dict):
            data = raster_stack["data"]
            mask = raster_stack.get("mask", None)
        else:
            data = raster_stack
            mask = None

        n_bands, height, width = data.shape
        pixels = data.reshape(n_bands, -1).T  # (n_pixels, n_bands)

        if mask is not None:
            valid_mask = (~mask.flatten()) & (~np.any(np.isnan(pixels), axis=1))
        else:
            valid_mask = ~np.any(np.isnan(pixels), axis=1)

        pred_flat = np.full(pixels.shape[0], np.nan, dtype=np.float32)
        valid_indices = np.flatnonzero(valid_mask)

        if len(valid_indices) > 0:
            for start in range(0, len(valid_indices), chunk_size):
                batch_idx = valid_indices[start : start + chunk_size]
                batch_pixels = pixels[batch_idx].astype(np.float64)
                batch_feat, _ = self._transform_features(batch_pixels)
                batch_scaled = self.scaler.transform(batch_feat)
                batch_pred = self.model.predict_proba(batch_scaled)[:, 1]
                pred_flat[batch_idx] = batch_pred.astype(np.float32)

        return pred_flat.reshape(height, width)

    def get_response_curves(self, raster_stack, n_points=100):
        """Generate response curves for each environmental variable."""
        if self._r_response_curves is not None:
            return self._r_response_curves

        if isinstance(raster_stack, dict):
            data = raster_stack["data"]
            mask = raster_stack.get("mask", None)
        else:
            data = raster_stack
            mask = None

        n_bands = data.shape[0]
        pixels = data.reshape(n_bands, -1).T

        if mask is not None:
            valid_rows = (~mask.flatten()) & (~np.any(np.isnan(pixels), axis=1))
        else:
            valid_rows = ~np.any(np.isnan(pixels), axis=1)

        valid_pixels = pixels[valid_rows]
        if len(valid_pixels) == 0:
            return {}

        mean_values = np.nanmean(valid_pixels, axis=0)
        curves = {}

        for i, var_name in enumerate(self.feature_names):
            var_values = valid_pixels[:, i]
            x_min = float(np.nanmin(var_values))
            x_max = float(np.nanmax(var_values))
            x_seq = np.linspace(x_min, x_max, n_points)

            synthetic = np.tile(mean_values, (n_points, 1)).astype(np.float64)
            synthetic[:, i] = x_seq

            synth_feat, _ = self._transform_features(synthetic)
            synth_scaled = self.scaler.transform(synth_feat)
            y_pred = self.model.predict_proba(synth_scaled)[:, 1]

            curves[var_name] = {"x": x_seq, "y": y_pred}

        return curves

    def jackknife_test(self, presence_data, background_data, raster_stack):
        """Perform jackknife test: train with each variable omitted and with only each variable."""
        if self._r_jackknife_results is not None:
            return self._r_jackknife_results

        env_cols = [
            c
            for c in presence_data.columns
            if c not in ["species", "x", "y", "longitude", "latitude"]
        ]
        if len(env_cols) == 0:
            return []

        results = []

        full_model = MaxEntModel(
            regularization=self.regularization,
            feature_classes=self.feature_classes,
            max_iterations=self.max_iterations,
            convergence_threshold=self.convergence_threshold,
            random_seed=self.random_seed,
            use_r=False,
            n_jobs=self.n_jobs,
        )
        full_results = full_model.train(presence_data, background_data, test_percentage=0)
        full_gain = full_results.get("train_auc", 0)

        if len(env_cols) == 1:
            return [
                {
                    "variable": env_cols[0],
                    "gain_without": 0.5,
                    "gain_with_only": full_gain,
                    "full_gain": full_gain,
                    "metric": "Training AUC (single-layer baseline)",
                }
            ]

        pres_base_cols = ["species", "x", "y"] if "species" in presence_data.columns else ["x", "y"]
        pres_base_cols = [c for c in pres_base_cols if c in presence_data.columns]
        back_base_cols = [c for c in ["x", "y"] if c in background_data.columns]

        for var in env_cols:
            other_vars = [v for v in env_cols if v != var]

            pres_without = presence_data[pres_base_cols + other_vars]
            back_without = background_data[back_base_cols + other_vars]

            model_without = MaxEntModel(
                regularization=self.regularization,
                feature_classes=self.feature_classes,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                random_seed=self.random_seed,
                use_r=False,
                n_jobs=self.n_jobs,
            )
            try:
                res_without = model_without.train(pres_without, back_without, test_percentage=0)
                gain_without = res_without.get("train_auc", 0)
            except Exception:
                gain_without = 0.0

            pres_only = presence_data[pres_base_cols + [var]]
            back_only = background_data[back_base_cols + [var]]

            model_only = MaxEntModel(
                regularization=self.regularization,
                feature_classes=self.feature_classes,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                random_seed=self.random_seed,
                use_r=False,
                n_jobs=self.n_jobs,
            )
            try:
                res_only = model_only.train(pres_only, back_only, test_percentage=0)
                gain_with_only = res_only.get("train_auc", 0)
            except Exception:
                gain_with_only = 0.0

            results.append(
                {
                    "variable": var,
                    "gain_without": gain_without,
                    "gain_with_only": gain_with_only,
                    "full_gain": full_gain,
                    "metric": "Training AUC",
                }
            )

        return results

    def calculate_threshold(self, train_predictions, percentile=10):
        """Calculate threshold at given percentile of training presence predictions."""
        if len(train_predictions) == 0:
            return 0.0
        return float(np.percentile(train_predictions, percentile))
