
import json, numpy as np
from library import Utils

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
    # convenient derived
    bundle["K_inv"] = np.linalg.inv(bundle["K"])
    return bundle
