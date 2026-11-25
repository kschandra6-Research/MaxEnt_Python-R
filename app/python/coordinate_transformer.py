"""Coordinate Transformer Module
Handles coordinate system transformations (WGS84 <-> UTM)
"""
import numpy as np
from pyproj import CRS, Transformer
from rasterio.coords import BoundingBox
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds


class CoordinateTransformer:
    def __init__(self):
        pass

    def detect_utm_zone(self, bounds, src_crs=None):
        """Auto-detect UTM zone from bounding box (handles geographic or projected CRS)."""
        crs_obj = CRS.from_user_input(src_crs) if src_crs else CRS.from_epsg(4326)

        if crs_obj.is_projected:
            # Already UTM? Keep it.
            name_upper = (crs_obj.name or "").upper()
            if "UTM" in name_upper:
                return crs_obj
            wgs_left, wgs_bottom, wgs_right, wgs_top = transform_bounds(
                crs_obj, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
            )
        else:
            wgs_left, wgs_bottom, wgs_right, wgs_top = (
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )

        center_lon = (wgs_left + wgs_right) / 2.0
        center_lat = (wgs_bottom + wgs_top) / 2.0

        zone_number = int((center_lon + 180.0) / 6.0) + 1
        zone_number = max(1, min(60, zone_number))
        epsg = (32600 if center_lat >= 0 else 32700) + zone_number
        return CRS.from_epsg(epsg)

    def transform_to_utm(self, points_df, utm_crs, src_crs=None):
        """Transform points from WGS84 (or src_crs) to UTM if coordinates are in lon/lat."""
        result = points_df.copy()
        xs = result["x"].values
        ys = result["y"].values

        # Only transform from WGS84 if values look like geographic degrees
        looks_like_lonlat = np.all((np.abs(xs) <= 180.0) & (np.abs(ys) <= 90.0))
        if src_crs is not None:
            src_obj = CRS.from_user_input(src_crs)
            if src_obj == CRS.from_user_input(utm_crs) and not looks_like_lonlat:
                return result
            from_crs = src_obj if not looks_like_lonlat else CRS.from_epsg(4326)
        else:
            if not looks_like_lonlat:
                # Coordinates appear already projected (in meters)
                return result
            from_crs = CRS.from_epsg(4326)

        transformer = Transformer.from_crs(from_crs, utm_crs, always_xy=True)
        x_utm, y_utm = transformer.transform(xs, ys)
        result["x"] = x_utm
        result["y"] = y_utm
        return result

    def transform_rasters(self, raster_stack, raster_meta, utm_crs):
        """Reproject raster stack and metadata to target UTM CRS."""
        src_crs = CRS.from_user_input(raster_meta["crs"])
        dst_crs = CRS.from_user_input(utm_crs)

        if src_crs == dst_crs:
            return raster_stack, raster_meta

        data = raster_stack["data"]
        n_bands, src_height, src_width = data.shape
        src_transform = raster_meta["transform"]
        src_bounds = raster_meta["bounds"]

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_width, src_height, *src_bounds
        )
        dst_bounds = BoundingBox(*array_bounds(dst_height, dst_width, dst_transform))

        dst_bands = []
        for i in range(n_bands):
            dest = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
            reproject(
                source=data[i].astype(np.float32),
                destination=dest,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            dst_bands.append(dest)

        new_data = np.stack(dst_bands, axis=0)
        new_mask = np.any(np.isnan(new_data), axis=0)

        new_meta = dict(raster_meta)
        new_meta.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "bounds": dst_bounds,
            }
        )

        return {"data": new_data, "mask": new_mask}, new_meta
