# MaxEnt Presence Prediction Application

A Streamlit-based application for Maximum Entropy (MaxEnt) species distribution modeling with Python and R integration.

## Features

- **Environmental Raster Upload**: Support for GeoTIFF (.tif, .tiff) and ESRI ASCII (.asc) formats, up to 2GB per file
- **Presence Points Upload**: CSV format with species name, longitude/X, latitude/Y columns, up to 2GB
- **Coordinate Systems**: WGS84 (EPSG:4326) and UTM (auto-detect zone) support
- **MaxEnt Modeling**: Python (scikit-learn) backend with optional R (dismo) integration
- **Model Parameters**: Configurable regularization, feature classes, iterations, convergence threshold
- **Output Options**: 
  - PDF Report
  - Word Report (.docx)
  - Suitability Raster (GeoTIFF)
  - Response Curves
  - Jackknife Analysis
  - Variable Importance
- **Results Visualization**: AUC metrics, variable importance plots, response curves, jackknife plots, suitability histograms

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# For R integration (optional)
# Install R and run: Rscript app/R/install_packages.R
# Required R packages: dismo, raster, terra, sp, sf, rJava, maxnet, jsonlite, pROC
```

## Usage

### Run the Application

```bash
python run_app.py
```

Or directly with Streamlit:

```bash
streamlit run app/streamlit_app.py
```

### Workflow

1. **Data Upload Tab**: 
   - Upload environmental rasters (multiple .tif/.asc files)
   - Upload presence points CSV
   - Map columns (species, longitude/X, latitude/Y)
   - Select coordinate system (WGS84 or UTM)

2. **Processing Tab**:
   - Configure MaxEnt parameters
   - Set output options
   - Click "Run MaxEnt Analysis"

3. **Results Tab**:
   - View model performance metrics (AUC, correlation)
   - Variable importance chart
   - Response curves
   - Jackknife analysis
   - Prediction statistics

4. **Reports Tab**:
   - Download PDF report
   - Download Word report
   - Download suitability raster (GeoTIFF)

## Sample Data Generation

To generate sample data for testing:

```bash
python generate_sample_data.py
```

This creates sample environmental layers and presence points in your temp directory.

## Requirements

- Python 3.8+
- R (optional, for dismo backend)
- GDAL/PROJ (via rasterio)

## Output Files

Reports are saved to `output/YYYYMMDD_HHMMSS/`:
- `MaxEnt_Report.pdf` - Comprehensive PDF report
- `MaxEnt_Report.docx` - Microsoft Word report
- `suitability.tif` - Habitat suitability GeoTIFF

## Configuration

Model parameters can be adjusted in the sidebar:
- **Regularization Multiplier**: Controls model complexity (0.1-5.0)
- **Feature Classes**: Linear, Quadratic, Product, Threshold, Hinge
- **Max Iterations**: Maximum optimization iterations (100-10000)
- **Convergence Threshold**: Stopping criterion (1e-6 to 1e-2)
- **Test Percentage**: Hold-out test data percentage (0-50%)
- **Background Points**: Number of background/pseudo-absence points (1000-100000)
- **Threshold**: 10th percentile training presence threshold

## Credits

Concept & Design by K.S.Chandra, 2026, kavikondalaschandra@gmail.com. (Inspired by MaxEnt Open Source - AMNH)

## License

MIT License
