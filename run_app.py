#!/usr/bin/env python
"""
Launcher script for MaxEnt Presence Prediction Streamlit App
"""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent
    app_path = root_dir / "app" / "streamlit_app.py"

    if not app_path.exists():
        print(f"Error: App not found at {app_path}")
        sys.exit(1)

    print("Starting MaxEnt Presence Prediction App...")
    print(f"App location: {app_path}")
    print("Max file upload size: 2048 MB (2 GB)")
    print("Open your browser to the URL shown below")
    print("-" * 50)

    # Run streamlit from project root so .streamlit/config.toml is picked up,
    # and pass explicit 2GB upload/message limits as fallback.
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.maxUploadSize=2048",
                "--server.maxMessageSize=2048",
            ],
            cwd=str(root_dir),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApp stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
