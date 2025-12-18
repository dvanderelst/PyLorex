# Repository Guidelines

## Project Structure & Module Organization
- `PyLorex/library/` houses the core camera, calibration, and TCP server modules (e.g., `Lorex.py`, `Homography.py`, `Simple_tcp.py`).
- `PyLorex/` includes executable scripts such as `script_start_server.py`, `script_run_homography.py`, and `script_test_lorex.py`.
- `PyLorex/Calibration/` stores calibration inputs/outputs; `Calibration/Results/` contains generated intrinsics and pose bundles.
- `PyLorex/Markers/` contains ArUco marker and calibration assets (SVGs and label sheets).
- `PyLorex/Simulations/` contains simulation scripts for rectilinear and fish-eye data.
- Root `run_server.py` and `PyLorex/run_server.py` are thin entry points for the TCP telemetry server.

## Build, Test, and Development Commands
- `python PyLorex/script_start_server.py` runs the TCP telemetry server with the lab’s default camera list.
- `python -m PyLorex.library.simple_tcp --camera tiger --camera panther` starts the server with ad-hoc camera names.
- `python PyLorex/script_run_homography.py` estimates homography from a calibration image.
- `python PyLorex/script_test_lorex.py` sanity-checks capture and undistortion using saved intrinsics.

## Coding Style & Naming Conventions
- Python uses 4-space indentation; follow the existing, pragmatic style rather than strict PEP 8.
- Prefer snake_case for functions and variables; keep module filenames descriptive (e.g., `script_quick_check_aruco.py`).
- Inline early-return checks are common; maintain this pattern for validation and error handling.

## Testing Guidelines
- There is no formal test suite. Validate changes by running the relevant script(s) for your area.
- Keep outputs in `PyLorex/Calibration/Results/` when generating new calibration data.

## Commit & Pull Request Guidelines
- Git history shows short, direct commit messages (e.g., “updated calibration”). Keep messages concise and imperative.
- Include in PR descriptions: the script(s) you ran, any generated calibration files, and sample output/log snippets when relevant.

## Configuration & Assets
- Update camera names, host, and port directly in `PyLorex/script_start_server.py` for day-to-day runs.
- Keep marker assets in `PyLorex/Markers/` and avoid renaming unless you update references in scripts.
