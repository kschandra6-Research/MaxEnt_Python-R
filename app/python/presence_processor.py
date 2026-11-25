"""Presence Points Processing Module"""
import pandas as pd


class PresenceProcessor:
    def __init__(self):
        pass

    def load_presence(self, csv_path, mapping):
        df = pd.read_csv(csv_path)
        species_col = mapping["species_col"]
        x_col = mapping["x_col"]
        y_col = mapping["y_col"]

        for col in (species_col, x_col, y_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in presence CSV.")

        result = pd.DataFrame()
        result["species"] = df[species_col].astype(str)
        result["x"] = pd.to_numeric(df[x_col], errors="coerce")
        result["y"] = pd.to_numeric(df[y_col], errors="coerce")

        result = result.dropna(subset=["x", "y"]).reset_index(drop=True)
        if result.empty:
            raise ValueError("No valid numeric coordinates found in presence CSV.")
        return result
