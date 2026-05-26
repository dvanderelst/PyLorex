
import json, os, numpy as np
from LorexLib import Utils

def load_pose_bundle(camera_name):
    """Load H (raw), PnP pose (R,t), K_scaled, dist, and mm settings from saved files."""
    p = Utils.get_calibration_paths(camera_name)
    # Prefer NPZ (exact arrays); fall back to JSON for readability
    try:
        data = np.load(p["pose_npz"], allow_pickle=False)
        bundle = {
            "K":        data["K_scaled"],
            "dist":     data["dist"],
            "H_raw":    data["H_raw"],
            "R":        data["R_pnp"],
            "t":        data["t_pnp"],
            "obj_mm":   data["obj_mm"],
            "corners":  data["corners"],
        }
        # Load H_undistorted if available (newer calibrations)
        if "H_undistorted" in data:
            bundle["H_undistorted"] = data["H_undistorted"]
        # Load frame_size and alpha from JSON since NPZ doesn't have them
        try:
            with open(p["pose_json"], "r") as f:
                J = json.load(f)
            if "image_size_WH" in J:
                bundle["frame_size"] = tuple(J["image_size_WH"])
            if "alpha" in J:
                bundle["alpha"] = float(J["alpha"])
            if "H_undistorted" in J and "H_undistorted" not in bundle:
                bundle["H_undistorted"] = np.asarray(J["H_undistorted"], dtype=float)
        except Exception:
            pass
    except Exception:
        with open(p["pose_json"], "r") as f:
            J = json.load(f)
        bundle = {
            "K":     np.asarray(J["K_scaled"], dtype=float),
            "dist":  np.asarray(J["dist_coeffs"], dtype=float),
            "H_raw": np.asarray(J["H_raw"], dtype=float),
            "R":     np.asarray(J["R_pnp"], dtype=float),
            "t":     np.asarray(J["t_pnp"], dtype=float),
        }
        # Load H_undistorted if available (newer calibrations)
        if "H_undistorted" in J:
            bundle["H_undistorted"] = np.asarray(J["H_undistorted"], dtype=float)
        # Load frame_size and alpha if available
        if "image_size_WH" in J:
            bundle["frame_size"] = tuple(J["image_size_WH"])
        if "alpha" in J:
            bundle["alpha"] = float(J["alpha"])
    # convenient derived
    bundle["K_inv"] = np.linalg.inv(bundle["K"])

    # Optional: measured camera centre in board frame (plumb-line + tape
    # height). Used by the calibration tooling (script_set_camera_center.py
    # derives shark2tiger_delta from these, and script_check_camera_-
    # agreement.py prints PnP vs measured side-by-side as a diagnostic).
    # NOT used by Lorex.get_aruco at tracking time — the marker pipeline
    # keeps PnP's (R, t) paired (substituting measured C while keeping PnP
    # R produces a ~300 mm inconsistency at the principal point; see the
    # comment in Lorex.get_aruco for the explanation).
    c_path = p.get("c_measured_json")
    if c_path and os.path.exists(c_path):
        try:
            with open(c_path, "r") as f:
                Cj = json.load(f)
            bundle["C_measured"] = np.array(
                [float(Cj["Cx_mm"]), float(Cj["Cy_mm"]), float(Cj["Cz_mm"])],
                dtype=np.float64,
            )
        except Exception as e:
            print(f"[bundle] Could not load measured C from {c_path}: {e}")
    return bundle
