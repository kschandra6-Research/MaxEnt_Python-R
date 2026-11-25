import pathlib
base = pathlib.Path(r"C:/Users/DuDu-PC/Downloads/MaxEnt_Python+R/app/python")

# presence_processor.py
content1 = '''Presence Points Processing Module
import pandas as pd
import numpy as np
from pathlib import Path

class PresenceProcessor:
    def __init__(self):
        pass
    
    def load_presence(self, csv_path, mapping):
        df = pd.read_csv(csv_path)
        species_col = mapping["species_col"]
        x_col = mapping["x_col"]
        y_col = mapping["y_col"]
        
        result = pd.DataFrame()
        result["species"] = df[species_col]
        result["x"] = pd.to_numeric(df[x_col], errors="coerce")
        result["y"] = pd.to_numeric(df[y_col], errors="coerce")
        
        result = result.dropna(subset=["x", "y"])
        return result.reset_index(drop=True)'''

(base / "presence_processor.py").write_text(content1)
print("presence_processor.py done")

# coordinate_transformer.py
content2 = '''Coordinate Transformer Module
Handles coordinate system transformations (WGS84 <-> UTM)
"""
import numpy as np
import pandas as pd
import pyproj
from pyproj import CRS, Transformer
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

class CoordinateTransformer:
    def __init__(self):
        pass
    
    def detect_utm_zone(self, bounds):
        """Auto-detect UTM zone from bounding box"""
        # bounds: (left, bottom, right, top) in WGS84
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
        
        # Calculate UTM zone
        zone_number = int((center_lon + 180) / 6) + 1
        zone_letter = "N" if center_lat >= 0 else "S"
        
        utm_crs = CRS.from_epsg(32600 + zone_number if zone_letter == "N" else 32700 + zone_number)
        return utm_crs
    
    def transform_to_utm(self, points_df, utm_crs):
        """Transform points from WGS84 to UTM"""
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        x_utm, y_utm = transformer.transform(points_df["x"].values, points_df["y"].values)
        
        result = points_df.copy()
        result["x"] = x_utm
        result["y"] = y_utm
        return result
    
    def transform_rasters(self, raster_stack, utm_crs):
        """Transform raster stack to UTM (placeholder - would need full reprojection)"""
        # For simplicity, return as-is
        # Full implementation would reproject each band
        return raster_stack'''

(base / "coordinate_transformer.py").write_text(content2)
print("coordinate_transformer.py done")

# report_generator.py
content3 = '''Report Generator Module
Generates PDF and Word reports for MaxEnt results
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import tempfile

class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_pdf(self, output_path, train_results, prediction_raster, response_curves, 
                     jackknife_results, threshold, raster_meta, raster_files, n_presence):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.ln(30)
        pdf.cell(0, 15, "MaxEnt Presence Prediction Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        pdf.set_font("Helvetica", "", 14)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(20)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Training AUC: {train_results.get('train_auc', 0):.4f}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Test AUC: {train_results.get('test_auc', 0):.4f}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Presence Points: {n_presence}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Environmental Layers: {len(raster_files)}", align="C", new_x="LMARGIN", new_y="NEXT")
        
        # Model Configuration
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Model Configuration", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        config_items = [
            f"Regularization Multiplier: {train_results.get('regularization', 'N/A')}",
            f"Feature Classes: {', '.join(train_results.get('feature_classes', []))}",
            f"Max Iterations: {train_results.get('max_iterations', 'N/A')}",
            f"Convergence Threshold: {train_results.get('convergence_threshold', 'N/A')}",
            f"Random Seed: {train_results.get('random_seed', 'N/A')}",
        ]
        for item in config_items:
            pdf.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")
        
        # Model Performance
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Model Performance", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        perf_items = [
            f"Training AUC: {train_results.get('train_auc', 0):.4f}",
            f"Test AUC: {train_results.get('test_auc', 0):.4f}",
            f"Training Correlation: {train_results.get('train_correlation', 0):.4f}",
            f"Test Correlation: {train_results.get('test_correlation', 0):.4f}",
        ]
        if threshold:
            perf_items.append(f"Threshold (10th percentile): {threshold:.4f}")
        for item in perf_items:
            pdf.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")
        
        # Variable Importance
        if "variable_importance" in train_results:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Variable Importance (Permutation Importance)", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)
            pdf.set_font("Helvetica", "", 10)
            vi = train_results["variable_importance"]
            for var, imp in sorted(vi.items(), key=lambda x: x[1], reverse=True):
                pdf.cell(0, 6, f"{var}: {imp:.2f}%", new_x="LMARGIN", new_y="NEXT")
        
        # Prediction Statistics
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Prediction Statistics", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        pred_stats = [
            f"Min Suitability: {np.nanmin(prediction_raster):.4f}",
            f"Max Suitability: {np.nanmax(prediction_raster):.4f}",
            f"Mean Suitability: {np.nanmean(prediction_raster):.4f}",
            f"Std Dev: {np.nanstd(prediction_raster):.4f}",
        ]
        for item in pred_stats:
            pdf.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")
        
        # Environmental Layers
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Environmental Layers", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        for i, fname in enumerate(raster_files, 1):
            pdf.cell(0, 6, f"{i}. {fname}", new_x="LMARGIN", new_y="NEXT")
        
        # Raster Metadata
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Raster Metadata", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        meta_items = [
            f"CRS: {raster_meta.get('crs', 'N/A')}",
            f"Width: {raster_meta.get('width', 'N/A')} pixels",
            f"Height: {raster_meta.get('height', 'N/A')} pixels",
            f"Bounds: {raster_meta.get('bounds', 'N/A')}",
            f"Number of Bands: {raster_meta.get('n_bands', 'N/A')}",
        ]
        for item in meta_items:
            pdf.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(output_path))
        return str(output_path)
    
    def generate_word(self, output_path, train_results, prediction_raster, response_curves,
                      jackknife_results, threshold, raster_meta, raster_files, n_presence):
        doc = Document()
        
        # Title
        title = doc.add_heading("MaxEnt Presence Prediction Report", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Date
        date_para = doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_run = date_para.add_run(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        date_run.font.size = Pt(12)
        
        doc.add_paragraph()
        
        # Summary Table
        doc.add_heading("Model Summary", level=1)
        table = doc.add_table(rows=5, cols=2, style="Table Grid")
        table.cell(0, 0).text = "Metric"
        table.cell(0, 1).text = "Value"
        summary_data = [
            ("Training AUC", f"{train_results.get('train_auc', 0):.4f}"),
            ("Test AUC", f"{train_results.get('test_auc', 0):.4f}"),
            ("Presence Points", str(n_presence)),
            ("Environmental Layers", str(len(raster_files))),
        ]
        for i, (metric, value) in enumerate(summary_data, 1):
            table.cell(i, 0).text = metric
            table.cell(i, 1).text = value
        
        # Model Configuration
        doc.add_heading("Model Configuration", level=1)
        config_items = [
            f"Regularization Multiplier: {train_results.get('regularization', 'N/A')}",
            f"Feature Classes: {', '.join(train_results.get('feature_classes', []))}",
            f"Max Iterations: {train_results.get('max_iterations', 'N/A')}",
            f"Convergence Threshold: {train_results.get('convergence_threshold', 'N/A')}",
            f"Random Seed: {train_results.get('random_seed', 'N/A')}",
        ]
        for item in config_items:
            doc.add_paragraph(item, style="List Bullet")
        
        # Model Performance
        doc.add_heading("Model Performance", level=1)
        perf_items = [
            f"Training AUC: {train_results.get('train_auc', 0):.4f}",
            f"Test AUC: {train_results.get('test_auc', 0):.4f}",
            f"Training Correlation: {train_results.get('train_correlation', 0):.4f}",
            f"Test Correlation: {train_results.get('test_correlation', 0):.4f}",
        ]
        if threshold:
            perf_items.append(f"Threshold (10th percentile): {threshold:.4f}")
        for item in perf_items:
            doc.add_paragraph(item, style="List Bullet")
        
        # Variable Importance
        if "variable_importance" in train_results:
            doc.add_heading("Variable Importance", level=1)
            vi = train_results["variable_importance"]
            table = doc.add_table(rows=len(vi)+1, cols=2, style="Table Grid")
            table.cell(0, 0).text = "Variable"
            table.cell(0, 1).text = "Importance (%)"
            for i, (var, imp) in enumerate(sorted(vi.items(), key=lambda x: x[1], reverse=True), 1):
                table.cell(i, 0).text = var
                table.cell(i, 1).text = f"{imp:.2f}"
        
        # Prediction Statistics
        doc.add_heading("Prediction Statistics", level=1)
        pred_stats = [
            f"Min Suitability: {np.nanmin(prediction_raster):.4f}",
            f"Max Suitability: {np.nanmax(prediction_raster):.4f}",
            f"Mean Suitability: {np.nanmean(prediction_raster):.4f}",
            f"Std Dev: {np.nanstd(prediction_raster):.4f}",
        ]
        for item in pred_stats:
            doc.add_paragraph(item, style="List Bullet")
        
        # Environmental Layers
        doc.add_heading("Environmental Layers", level=1)
        for i, fname in enumerate(raster_files, 1):
            doc.add_paragraph(f"{i}. {fname}", style="List Number")
        
        # Raster Metadata
        doc.add_heading("Raster Metadata", level=1)
        meta_items = [
            f"CRS: {raster_meta.get('crs', 'N/A')}",
            f"Width: {raster_meta.get('width', 'N/A')} pixels",
            f"Height: {raster_meta.get('height', 'N/A')} pixels",
            f"Bounds: {raster_meta.get('bounds', 'N/A')}",
            f"Number of Bands: {raster_meta.get('n_bands', 'N/A')}",
        ]
        for item in meta_items:
            doc.add_paragraph(item, style="List Bullet")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return str(output_path)'''

(base / "report_generator.py").write_text(content3)
print("report_generator.py done")
print("All modules written!")
