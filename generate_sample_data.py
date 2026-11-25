#!/usr/bin/env python
"""
Generate sample environmental rasters and presence points for testing
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import pandas as pd
from pathlib import Path
import tempfile
import shutil

def generate_sample_rasters(output_dir, n_layers=5, width=200, height=200):
    """Generate sample environmental raster layers"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bounds = (-120.0, 35.0, -119.0, 36.0)
    transform = from_bounds(*bounds, width, height)
    
    layer_names = [
        "bio1_annual_mean_temp",
        "bio12_annual_precipitation",
        "bio4_temp_seasonality",
        "bio15_precipitation_seasonality",
        "elevation",
        "slope",
        "aspect",
        "landcover"
    ]
    
    np.random.seed(42)
    
    for i in range(min(n_layers, len(layer_names))):
        name = layer_names[i]
        
        if "temp" in name.lower():
            data = np.random.normal(15, 5, (height, width)) + np.linspace(-2, 2, height)[:, None]
            data = np.clip(data, -10, 35)
        elif "precip" in name.lower():
            data = np.random.exponential(500, (height, width)) + np.linspace(0, 500, height)[:, None]
            data = np.clip(data, 0, 3000)
        elif "elevation" in name.lower():
            data = np.random.normal(500, 200, (height, width)) + np.linspace(-100, 100, height)[:, None]
            data = np.clip(data, 0, 2000)
        elif "slope" in name.lower():
            data = np.random.exponential(5, (height, width))
            data = np.clip(data, 0, 45)
        elif "aspect" in name.lower():
            data = np.random.uniform(0, 360, (height, width))
        elif "landcover" in name.lower():
            data = np.random.randint(1, 10, (height, width)).astype(float)
        else:
            data = np.random.normal(0, 1, (height, width))
        
        from scipy.ndimage import gaussian_filter
        data = gaussian_filter(data, sigma=3)
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': width,
            'height': height,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform,
            'nodata': -9999,
            'compress': 'lzw'
        }
        
        output_path = output_dir / f"{name}.tif"
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype(np.float32), 1)
        
        print(f"Generated: {output_path}")
    
    return [layer_names[i] for i in range(min(n_layers, len(layer_names)))]

def generate_sample_presence(output_dir, n_points=100, species_name="TestSpecies"):
    """Generate sample presence points within raster bounds"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bounds = (-120.0, 35.0, -119.0, 36.0)
    
    np.random.seed(123)
    
    centers = [
        (-119.5, 35.5),
        (-119.3, 35.7),
        (-119.7, 35.3)
    ]
    
    points = []
    for _ in range(n_points):
        center = np.random.choice(len(centers))
        cx, cy = centers[center]
        x = cx + np.random.normal(0, 0.05)
        y = cy + np.random.normal(0, 0.05)
        x = np.clip(x, bounds[0], bounds[2])
        y = np.clip(y, bounds[1], bounds[3])
        points.append({
            'species': species_name,
            'longitude': x,
            'latitude': y
        })
    
    df = pd.DataFrame(points)
    output_path = output_dir / f"{species_name}_presence.csv"
    df.to_csv(output_path, index=False)
    print("Generated presence points: {} ({} points)".format(output_path, len(df)))
    
    return output_path

def main():
    temp_dir = Path(tempfile.gettempdir()) / "maxent_sample_data"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    rasters_dir = temp_dir / "rasters"
    presence_dir = temp_dir / "presence"
    
    print("Generating sample environmental rasters...")
    generate_sample_rasters(rasters_dir, n_layers=6, width=300, height=300)
    
    print("Generating sample presence points.....")
    generate_sample_presence(presence_dir, n_points=150, species_name="SampleSpecies")
    
    print("Sample data generated in: {}".format(temp_dir))
    print("Rasters: {}".format(rasters_dir))
    print("Presence: {}".format(presence_dir))
    print("Use these files to test the MaxEnt app!")

if __name__ == "__main__":
    main()
