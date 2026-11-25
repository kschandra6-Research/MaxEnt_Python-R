"""Raster Processing Module
Handles loading, validation, reprojection, and processing of environmental rasters
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

warnings.filterwarnings("ignore")


class RasterProcessor:
    """Process environmental rasters for MaxEnt modeling"""

    def __init__(self):
        self.raster_stack = None
        self.raster_meta = None
        self.nodata_value = None

    def load_rasters(self, raster_dir, target_crs=None):
        """
        Load, align/reproject, and stack multiple rasters.

        Args:
            raster_dir: Directory containing raster files
            target_crs: Target CRS for reprojection (optional)

        Returns:
            tuple: (raster_stack_dict, metadata_dict)
        """
        raster_dir = Path(raster_dir)
        raster_files = sorted(
            list(raster_dir.glob("*.tif"))
            + list(raster_dir.glob("*.tiff"))
            + list(raster_dir.glob("*.asc"))
        )

        if not raster_files:
            raise ValueError("No raster files found in directory")

        parsed_target_crs = CRS.from_user_input(target_crs) if target_crs else None

        with rasterio.open(raster_files[0]) as src0:
            src0_crs = src0.crs or parsed_target_crs or CRS.from_epsg(4326)
            if parsed_target_crs and src0_crs != parsed_target_crs:
                ref_transform, ref_width, ref_height = calculate_default_transform(
                    src0_crs, parsed_target_crs, src0.width, src0.height, *src0.bounds
                )
                ref_crs = parsed_target_crs
                ref_bounds = BoundingBox(*array_bounds(ref_height, ref_width, ref_transform))
            else:
                ref_crs = src0_crs
                ref_transform = src0.transform
                ref_width = src0.width
                ref_height = src0.height
                ref_bounds = src0.bounds
            ref_nodata = src0.nodata

        bands_data = []
        band_names = []

        for rf in raster_files:
            with rasterio.open(rf) as src:
                src_crs = src.crs or ref_crs
                raw = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    if np.isnan(src.nodata):
                        pass
                    else:
                        raw[raw == np.float32(src.nodata)] = np.nan

                needs_warp = (
                    src_crs != ref_crs
                    or src.width != ref_width
                    or src.height != ref_height
                    or src.transform != ref_transform
                )

                if needs_warp:
                    dest = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
                    reproject(
                        source=raw,
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src_crs,
                        src_nodata=np.nan,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                    data = dest
                else:
                    data = raw

                bands_data.append(data)
                band_names.append(rf.stem)

        raster_stack = np.stack(bands_data, axis=0)
        mask = np.any(np.isnan(raster_stack), axis=0)

        metadata = {
            "crs": ref_crs,
            "transform": ref_transform,
            "width": ref_width,
            "height": ref_height,
            "bounds": ref_bounds,
            "nodata": np.nan,
            "src_nodata": ref_nodata,
            "dtype": "float32",
            "band_names": band_names,
            "n_bands": len(band_names),
        }

        self.raster_stack = {"data": raster_stack, "mask": mask}
        self.raster_meta = metadata
        self.nodata_value = np.nan

        return self.raster_stack, metadata

    def extract_values(self, raster_stack, points_df):
        """
        Extract raster values at point locations.

        Args:
            raster_stack: Dict with data and mask
            points_df: DataFrame with x, y columns

        Returns:
            DataFrame with environmental values added
        """
        data = raster_stack["data"]
        mask = raster_stack.get("mask")
        transform = self.raster_meta["transform"]
        band_names = self.raster_meta["band_names"]

        rows, cols = rasterio.transform.rowcol(
            transform, points_df["x"].values, points_df["y"].values
        )
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)

        height, width = data.shape[1], data.shape[2]
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if mask is not None:
            in_bounds_valid = np.zeros(len(points_df), dtype=bool)
            in_bounds_valid[valid] = ~mask[rows[valid], cols[valid]]
            valid = in_bounds_valid

        result = points_df.copy()
        for i, band_name in enumerate(band_names):
            values = np.full(len(points_df), np.nan, dtype=np.float64)
            values[valid] = data[i, rows[valid], cols[valid]]
            result[band_name] = values

        return result

    def generate_background(self, raster_stack, n_points=10000, random_seed=42):
        """
        Generate random background points within valid raster pixels.

        Args:
            raster_stack: Dict with data and mask
            n_points: Number of background points
            random_seed: Random seed for reproducibility

        Returns:
            DataFrame with background point coordinates
        """
        rng = np.random.default_rng(random_seed)
        mask = raster_stack["mask"]
        transform = self.raster_meta["transform"]

        valid_rows, valid_cols = np.where(~mask)

        if len(valid_rows) == 0:
            raise ValueError("No valid pixels in raster for background generation")

        n_samples = min(int(n_points), len(valid_rows))
        sample_idx = rng.choice(len(valid_rows), n_samples, replace=False)

        sample_rows = valid_rows[sample_idx]
        sample_cols = valid_cols[sample_idx]

        xs, ys = rasterio.transform.xy(transform, sample_rows, sample_cols)

        return pd.DataFrame({"x": xs, "y": ys})

    def save_raster(self, prediction_array, output_path, metadata):
        """
        Save 2D prediction array as GeoTIFF.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": metadata["width"],
            "height": metadata["height"],
            "count": 1,
            "crs": metadata["crs"],
            "transform": metadata["transform"],
            "nodata": -9999.0,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            out_data = prediction_array.astype(np.float32, copy=True)
            out_data[np.isnan(out_data)] = -9999.0
            dst.write(out_data, 1)

        return str(output_path)

    def save_binary_raster(self, prediction_array, threshold, output_path, metadata):
        """Save thresholded binary suitability GeoTIFF (1=suitable, 0=unsuitable, 255=NoData)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": metadata["width"],
            "height": metadata["height"],
            "count": 1,
            "crs": metadata["crs"],
            "transform": metadata["transform"],
            "nodata": 255,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            for row_start in range(0, metadata["height"], 1024):
                row_end = min(row_start + 1024, metadata["height"])
                chunk = prediction_array[row_start:row_end, :]
                out_chunk = np.full(chunk.shape, 255, dtype=np.uint8)
                valid = ~np.isnan(chunk)
                out_chunk[valid] = (chunk[valid] >= threshold).astype(np.uint8)
                dst.write(out_chunk, 1, window=rasterio.windows.Window(0, row_start, metadata["width"], row_end - row_start))

        return str(output_path)

    def save_stack(self, raster_stack, output_path, metadata):
        """Save multi-band raster stack as GeoTIFF for external tools (e.g. R)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = raster_stack["data"]

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": metadata["width"],
            "height": metadata["height"],
            "count": data.shape[0],
            "crs": metadata["crs"],
            "transform": metadata["transform"],
            "nodata": -9999.0,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(data.shape[0]):
                band = data[i].astype(np.float32, copy=True)
                band[np.isnan(band)] = -9999.0
                dst.write(band, i + 1)
                if i < len(metadata.get("band_names", [])):
                    dst.set_band_description(i + 1, metadata["band_names"][i])

        return str(output_path)

    def get_raster_statistics(self, raster_stack, max_median_samples=2_000_000):
        """Get basic statistics of raster stack in memory-safe row chunks."""
        data = raster_stack["data"]
        mask = raster_stack["mask"]
        band_names = self.raster_meta["band_names"]
        height = data.shape[1]
        rng = np.random.default_rng(42)

        stats = {}
        for i, name in enumerate(band_names):
            band_data = data[i]
            vmin = np.inf
            vmax = -np.inf
            total_sum = 0.0
            total_sq = 0.0
            count = 0
            median_samples = []

            for r0 in range(0, height, 1024):
                r1 = min(r0 + 1024, height)
                chunk = band_data[r0:r1]
                chunk_mask = mask[r0:r1]
                vals = chunk[~chunk_mask]
                vals = vals[np.isfinite(vals)]
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
                count += int(vals.size)

                per_chunk_cap = max(1000, max_median_samples // max(1, (height // 1024) + 1))
                if vals.size > per_chunk_cap:
                    idx = rng.choice(vals.size, size=per_chunk_cap, replace=False)
                    median_samples.append(vals[idx])
                else:
                    median_samples.append(vals)

            if count == 0:
                stats[name] = {k: float("nan") for k in ("min", "max", "mean", "std", "median")}
            else:
                mean = total_sum / count
                var = max(0.0, (total_sq / count) - (mean * mean))
                med_arr = np.concatenate(median_samples) if median_samples else np.array([mean])
                stats[name] = {
                    "min": float(vmin),
                    "max": float(vmax),
                    "mean": float(mean),
                    "std": float(np.sqrt(var)),
                    "median": float(np.median(med_arr)),
                }
        return stats
