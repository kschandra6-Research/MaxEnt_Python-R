"""
R Bridge Utility
Discovers Rscript, verifies required R packages, installs missing R plugins,
and executes R MaxEnt scripts via subprocess.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_R_DIR = PROJECT_ROOT / "app" / "R"
PROJECT_R_LIB = PROJECT_ROOT / "R" / "library"


def find_rscript() -> Optional[str]:
    """Locate Rscript executable across PATH and common Windows/Unix install paths."""
    env_rscript = os.environ.get("RSCRIPT_PATH")
    if env_rscript and Path(env_rscript).is_file():
        return env_rscript

    which_r = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if which_r:
        return which_r

    # Common Windows R installation locations (pick newest version)
    candidates = []
    for base in [
        Path("C:/Program Files/R"),
        Path("C:/Program Files (x86)/R"),
        Path.home() / "AppData" / "Local" / "Programs" / "R",
    ]:
        if base.is_dir():
            for r_dir in sorted(base.glob("R-*"), reverse=True):
                exe = r_dir / "bin" / "Rscript.exe"
                if exe.is_file():
                    candidates.append(str(exe))
                exe_x64 = r_dir / "bin" / "x64" / "Rscript.exe"
                if exe_x64.is_file():
                    candidates.append(str(exe_x64))

    if candidates:
        return candidates[0]

    # Common Unix locations
    for unix_path in ["/usr/bin/Rscript", "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript"]:
        if Path(unix_path).is_file():
            return unix_path

    return None


def check_r_environment() -> Dict[str, Any]:
    """Check R installation, required packages, and maxent.jar status."""
    rscript = find_rscript()
    if not rscript:
        return {
            "r_available": False,
            "rscript_path": None,
            "dismo_ready": False,
            "maxnet_ready": False,
            "error": "Rscript executable not found. Install R (4.x+) or add Rscript to PATH.",
        }

    check_script = APP_R_DIR / "check_r_env.R"
    if not check_script.is_file():
        return {
            "r_available": True,
            "rscript_path": rscript,
            "dismo_ready": False,
            "maxnet_ready": False,
            "error": f"Missing R check script: {check_script}",
        }

    try:
        proc = subprocess.run(
            [rscript, "--vanilla", str(check_script), str(PROJECT_R_LIB)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = (proc.stdout or "").strip()
        # Extract last JSON line in case R startup printed warnings
        json_line = ""
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                json_line = line
                break

        if not json_line:
            return {
                "r_available": True,
                "rscript_path": rscript,
                "dismo_ready": False,
                "maxnet_ready": False,
                "error": proc.stderr.strip() or stdout or "Failed to parse R environment check output.",
            }

        data = json.loads(json_line)
        data["r_available"] = True
        data["rscript_path"] = rscript
        return data
    except Exception as exc:
        return {
            "r_available": True,
            "rscript_path": rscript,
            "dismo_ready": False,
            "maxnet_ready": False,
            "error": str(exc),
        }


def install_r_packages(timeout: int = 600) -> Dict[str, Any]:
    """Run app/R/install_packages.R to install missing CRAN R packages into project library."""
    rscript = find_rscript()
    if not rscript:
        return {"success": False, "output": "Rscript executable not found."}

    PROJECT_R_LIB.mkdir(parents=True, exist_ok=True)
    install_script = APP_R_DIR / "install_packages.R"
    proc = subprocess.run(
        [rscript, "--vanilla", str(install_script), str(PROJECT_R_LIB)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = "\n".join(part for part in [proc.stdout, proc.stderr] if part).strip()
    return {"success": proc.returncode == 0, "output": output}


def run_r_maxent(config_path: Path, timeout: int = 3600) -> Dict[str, Any]:
    """Execute app/R/run_maxent.R with a JSON configuration file."""
    rscript = find_rscript()
    if not rscript:
        raise RuntimeError("Rscript executable not found.")

    run_script = APP_R_DIR / "run_maxent.R"
    if not run_script.is_file():
        raise FileNotFoundError(f"R MaxEnt runner script not found: {run_script}")

    proc = subprocess.run(
        [rscript, "--vanilla", str(run_script), str(config_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or f"R exited with code {proc.returncode}").strip()
        raise RuntimeError(f"R MaxEnt execution failed: {err_msg}")

    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
