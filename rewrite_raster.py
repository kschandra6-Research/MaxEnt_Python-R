import pathlib
p = pathlib.Path("C:/Users/DuDu-PC/Downloads/MaxEnt_Python+R/app/python/raster_processor.py")
content = '''Raster Processing Module
Handles loading, validation, and processing of environmental rasters
"""
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.enums import Resampling as ResamplingEnum
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class RasterProcessor:
    """Process environmental rasters for MaxEnt modeling"""
    
    def __init__(self):
        self.raster_stack = None
        self.raster_meta = None
        self.nodata_value = None
    
    def load_rasters(self, raster_dir, target_crs=None):
        """
        Load and stack multiple rasters
        
        Args:
            raster_dir: Directory containing raster files
            target_crs: Target CRS for reprojection (e.g., "EPSG:4326")
            
        Returns:
            tuple: (raster_stack_dict, metadata_dict)
        """
        raster_dir = Path(raster_dir)
        raster_files = list(raster_dir.glob("*.tif")) + list(raster_dir.glob("*.tiff")) + list(raster_dir.glob("*.asc"))
        
        if not raster_files:
            raise ValueError("No raster files found in directory")
        
        with rasterio.open(raster_files[0]) as src:
            ref_crs = src.crs
            ref_transform = src.transform
            ref_width = src.width
            ref_height = src.height
            ref_bounds = src.bounds
            ref_nodata = src.nodata
            ref_dtype = src.dtypes[0]
        
        for rf in raster_files[1:]:
            with rasterio.open(rf) as src:
                if src.width != ref_width or src.height != ref_height:
                    raise ValueError(f"Raster {rf.name} has different dimensions")
                if src.crs != ref_crs:
                    print(f"Warning: {rf.name} has different CRS, will reproject")
                if src.bounds != ref_bounds:
                    print(f"Warning: {rf.name} has different bounds")
        
        bands_data = []
        band_names = []
        
        for rf in raster_files:
            with rasterio.open(rf) as src:
                if target_crs and src.crs != target_crs:
                    data = self._reproject_raster(src, target_crs)
                    ref_crs = target_crs
                else:
                    data = src.read(1)
                
                bands_data.append(data)
                band_names.append(rf.stem)
        
        raster_stack = np.stack(bands_data, axis=0)
        
        if ref_nodata is not None:
            mask = np.any(raster_stack == ref_nodata, axis=0)
            mask = mask | np.any(np.isnan(raster_stack), axis=0)
        else:
            mask = np.any(np.isnan(raster_stack), axis=0)
        
        raster_stack = raster_stack.astype(float)
        if ref_nodata is not None:
            raster_stack[raster_stack == ref_nodata] = np.nan
        
        metadata = {
            "crs": ref_crs,
            "transform": ref_transform,
            "width": ref_width,
            "height": ref_height,
            "bounds": ref_bounds,
            "nodata": ref_nodata,
            "dtype": ref_dtype,
            "band_names": band_names,
            "n_bands": len(band_names)
        }
        
        self.raster_stack = {"data": raster_stack, "mask": mask}
        self.raster_meta = metadata
        self.nodata_value = ref_nodata
        
        return self.raster_stack, metadata
    
    def _reproject_raster(self, src, target_crs):
        """Reproject a single raster to target CRS"""
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        destination = np.empty((height, width), dtype=src.dtypes[0])
        
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        
        return destination
    
    def extract_values(self, raster_stack, points_df):
        """
        Extract raster values at point locations
        
        Args:
            raster_stack: Dict with "data" (bands, height, width) and "mask"
            points_df: DataFrame with "x", "y" columns
            
        Returns:
            DataFrame with environmental values added
        """
        data = raster_stack["data"]
        transform = self.raster_meta["transform"]
        band_names = self.raster_meta["band_names"]
        
        cols, rows = rasterio.transform.rowcol(transform, points_df["x"].values, points_df["y"].values)
        
        height, width = data.shape[1], data.shape[2]
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        
        result = points_df.copy()
        for i, band_name in enumerate(band_names):
            values = np.full(len(points_df), np.nan)
            values[valid] = data[i, rows[valid], cols[valid]]
            result[band_name] = values
        
        return result
    
    def generate_background(self, raster_stack, n_points=10000, random_seed=42):
        """
        Generate random background points within raster extent
        
        Args:
            raster_stack: Dict with "data" and "mask"
            n_points: Number of background points
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with background point coordinates
        """
        np.random.seed(random_seed)
        data = raster_stack["data"]
        mask = raster_stack["mask"]
        transform = self.raster_meta["transform"]
        height, width = data.shape[1], data.shape[2]
        
        valid_rows, valid_cols = np.where(~mask)
        
        if len(valid_rows) == 0:
            raise ValueError("No valid pixels in raster for background generation")
        
        n_samples = min(n_points, len(valid_rows))
        sample_idx = np.random.choice(len(valid_rows), n_samples, replace=False)
        
        sample_rows = valid_rows[sample_idx]
        sample_cols = valid_cols[sample_idx]
        
        xs, ys = rasterio.transform.xy(transform, sample_rows, sample_cols)
        
        return pd.DataFrame({"x": xs, "y": ys})
    
    def save_raster(self, prediction_array, output_path, metadata):
        """
        Save prediction array as GeoTIFF
        
        Args:
            prediction_array: 2D array of predictions
            output_path: Output file path
            metadata: Raster metadata dictionary
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
            "nodata": -9999,
            "compress": "lzw"
        }
        
        with rasterio.open(output_path, "w", **profile) as dst:
            out_data = prediction_array.astype(np.float32)
            out_data[np.isnan(out_data)] = -9999
            dst.write(out_data, 1)
        
        return str(output_path)
    
    def get_raster_statistics(self, raster_stack):
        """Get basic statistics of raster stack"""
        data = raster_stack["data"]
        mask = raster_stack["mask"]
        band_names = self.raster_meta["band_names"]
        
        stats = {}
        for i, name in enumerate(band_names):
            band_data = data[i]
            valid_data = band_data[~mask]
            stats[name] = {
                "min": float(np.nanmin(valid_data)),
                "max": float(np.nanmax(valid_data)),
                "mean": float(np.nanmean(valid_data)),
                "std": float(np.nanstd(valid_data)),
                "median": float(np.nanmedian(valid_data))
            }
        return stats
'''
p.write_text(content)
print("Raster processor rewritten")
