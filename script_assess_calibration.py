import re, sys, math, statistics as stats
from pathlib import Path
from datetime import datetime
from html import escape

from library import Utils  # your utilities


camera_name = "shark"   # <-- set your camera name here

# ---------- Heuristic thresholds (with brief explanations) ----------
# MIN_IMAGES: We want enough diverse views for a stable calibration; <12 often underconstrained.
MIN_IMAGES = 12
# RMS_GOOD: Global RMS (px). Subpixel reprojection (<0.5 px) is typically “good” for non-fisheye lenses.
RMS_GOOD = 0.5
# P90_ERR_GOOD: 90th percentile per-image error; keeps most views under control (catchs a few bad frames).
P90_ERR_GOOD = 0.5
# MAX_ERR_OK: Hard ceiling for any single view’s error; >1 px often indicates a bad detection/outlier.
MAX_ERR_OK = 1.0
# ASPECT_WARN: Warn if fy/fx deviates >10% from 1.0 (square pixels). Could be true non-square pixels or bias.
ASPECT_WARN = 1.10
# PRINC_PT_WARN_FRAC: Warn if principal point is far from center (>8% of width/height). Can be real, but worth noting.
PRINC_PT_WARN_FRAC = 0.08
# -------------------------------------------------------------------

def parse_opencv_filestorage(text: str):
    """Tiny parser for common OpenCV FileStorage keys."""
    def read_scalar(key, cast=float):
        m = re.search(rf"^{key}:\s*([0-9.\-+Ee]+)\s*$", text, re.MULTILINE)
        return cast(m.group(1)) if m else None

    def read_int(key):
        m = re.search(rf"^{key}:\s*(\d+)\s*$", text, re.MULTILINE)
        return int(m.group(1)) if m else None

    def read_mat(name):
        pat = rf"{name}:\s*!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*\w\s*data:\s*\[([^\]]+)\]"
        m = re.search(pat, text, re.MULTILINE)
        if not m: return None
        rows, cols = int(m.group(1)), int(m.group(2))
        data = [float(x.strip()) for x in m.group(3).split(",")]
        return rows, cols, data

    iw = read_int("image_width")
    ih = read_int("image_height")
    rms = read_scalar("rms_reprojection_error")

    cam = read_mat("camera_matrix")
    dist = read_mat("distortion_coefficients")

    per_img = []
    for m in re.finditer(r'file:\s*"([^"]+)"\s*error:\s*([0-9.\-+Ee]+)', text):
        per_img.append((m.group(1), float(m.group(2))))

    return {
        "image_width": iw, "image_height": ih, "rms": rms,
        "camera_matrix": cam, "dist": dist, "per_image": per_img
    }

def analyze(yml_path: Path):
    text = yml_path.read_text()
    d = parse_opencv_filestorage(text)

    iw, ih = d["image_width"], d["image_height"]
    cam = d["camera_matrix"]
    if not iw or not ih or not cam:
        raise RuntimeError("Missing image size or camera_matrix in file.")

    rows, cols, Kd = cam
    if (rows, cols) != (3, 3):
        raise RuntimeError("camera_matrix is not 3x3.")

    fx, skew, cx, _, fy, cy, _, _, _ = Kd
    rms = d["rms"]
    errs = [e for _, e in d["per_image"]] if d["per_image"] else []

    cx0, cy0 = iw/2.0, ih/2.0
    dx_px, dy_px = cx - cx0, cy - cy0
    dx_frac, dy_frac = abs(dx_px)/iw, abs(dy_px)/ih
    aspect = fy/fx if fx != 0 else float("inf")
    hfov = 2*math.degrees(math.atan(iw/(2*fx))) if fx>0 else float("nan")
    vfov = 2*math.degrees(math.atan(ih/(2*fy))) if fy>0 else float("nan")

    n = len(errs)
    mean_e = stats.mean(errs) if n else float("nan")
    med_e  = stats.median(errs) if n else float("nan")
    p90_e  = (sorted(errs)[int(0.9*(n-1))] if n else float("nan"))
    max_e  = max(errs) if n else float("nan")
    min_e  = min(errs) if n else float("nan")

    ok_images = (n >= MIN_IMAGES)
    ok_rms    = (rms is not None and rms < RMS_GOOD)
    ok_p90    = (p90_e is not None and p90_e < P90_ERR_GOOD)
    ok_max    = (max_e is not None and max_e < MAX_ERR_OK)

    warn_aspect = (aspect > ASPECT_WARN or aspect < 1/ASPECT_WARN)
    warn_princp = (dx_frac > PRINC_PT_WARN_FRAC or dy_frac > PRINC_PT_WARN_FRAC)
    warn_skew   = (abs(skew) > 1e-6)

    passed = ok_images and ok_rms and ok_p90 and ok_max

    dist_str = None
    if d['dist']:
        _, _, distd = d['dist']
        distd = (distd + [0]*5)[:5]
        dist_str = dict(k1=distd[0], k2=distd[1], p1=distd[2], p2=distd[3], k3=distd[4])

    return {
        "img_size": (iw, ih),
        "fx": fx, "fy": fy, "skew": skew, "cx": cx, "cy": cy,
        "aspect": aspect, "hfov": hfov, "vfov": vfov,
        "dx_px": dx_px, "dy_px": dy_px, "dx_frac": dx_frac, "dy_frac": dy_frac,
        "rms": rms, "n": n,
        "mean_e": mean_e, "med_e": med_e, "p90_e": p90_e, "max_e": max_e, "min_e": min_e,
        "ok_images": ok_images, "ok_rms": ok_rms, "ok_p90": ok_p90, "ok_max": ok_max,
        "warn_aspect": warn_aspect, "warn_princp": warn_princp, "warn_skew": warn_skew,
        "passed": passed,
        "per_image": d["per_image"],
        "dist": dist_str
    }

def print_console(yml_path: Path, R):
    iw, ih = R["img_size"]
    print(f"== Calibration report for: {yml_path.name} ==")
    print(f"Image size            : {iw} x {ih}")
    print(f"fx, fy                : {R['fx']:.3f}, {R['fy']:.3f}  (fy/fx = {R['aspect']:.3f})")
    print(f"cx, cy                : {R['cx']:.3f}, {R['cy']:.3f}  (offsets: {R['dx_px']:.2f}px, {R['dy_px']:.2f}px)")
    print(f"Estimated FOV (deg)   : H {R['hfov']:.2f}°, V {R['vfov']:.2f}°")
    if R["dist"]:
        d = R["dist"]
        print(f"Distortion (k1,k2,p1,p2,k3): {d['k1']:.3f}, {d['k2']:.3f}, {d['p1']:.4f}, {d['p2']:.4f}, {d['k3']:.3f}")

    print("\n-- Reprojection error (px) --")
    print(f"Global RMS            : {R['rms']:.3f}")
    if R["n"]:
        print(f"Images used           : {R['n']}")
        print(f"Per-image: mean {R['mean_e']:.3f}, median {R['med_e']:.3f}, p90 {R['p90_e']:.3f}, max {R['max_e']:.3f}, min {R['min_e']:.3f}")
    else:
        print("Per-image errors      : (none listed)")

    print("\n-- Heuristic checks --")
    print(f"Enough images (≥{MIN_IMAGES}) : {'OK' if R['ok_images'] else 'LOW'}")
    print(f"RMS < {RMS_GOOD:.2f} px            : {'OK' if R['ok_rms'] else 'HIGH'}")
    print(f"p90 < {P90_ERR_GOOD:.2f} px         : {'OK' if R['ok_p90'] else 'HIGH'}")
    print(f"max < {MAX_ERR_OK:.1f} px           : {'OK' if R['ok_max'] else 'HIGH'}")

    print("\n== RESULT ==")
    print("PASS ✅" if R["passed"] else "WARN ⚠️  (see HTML notes)")

def write_html(html_path: Path, cam_name: str, yml_path: Path, R):
    html_path.parent.mkdir(parents=True, exist_ok=True)
    iw, ih = R["img_size"]

    def badge(ok): return f'<span class="b {"ok" if ok else "bad"}">{"OK" if ok else "CHECK"}</span>'
    def yesno(x): return "Yes" if x else "No"

    # thresholds explainer
    settings_rows = [
        ("MIN_IMAGES", f"{MIN_IMAGES}",
         "Minimum number of images used. Fewer images tend to under-constrain intrinsics."),
        ("RMS_GOOD", f"{RMS_GOOD:.2f} px",
         "Global RMS reprojection error; sub-pixel (<0.5 px) is typically good."),
        ("P90_ERR_GOOD", f"{P90_ERR_GOOD:.2f} px",
         "90th percentile of per-image errors; catches most bad views."),
        ("MAX_ERR_OK", f"{MAX_ERR_OK:.1f} px",
         "Maximum allowed error for any single image."),
        ("ASPECT_WARN", f"{ASPECT_WARN:.2f}",
         "Warn if fy/fx deviates more than this from 1.0 (square pixels)."),
        ("PRINC_PT_WARN_FRAC", f"{PRINC_PT_WARN_FRAC:.2f}",
         "Warn if principal point is > this fraction of width/height away from center.")
    ]

    perimg_table = ""
    if R["per_image"]:
        rows = []
        for f, e in R["per_image"]:
            rows.append(f"<tr><td>{escape(Path(f).name)}</td><td class='num'>{e:.3f}</td></tr>")
        perimg_table = f"""
        <h3>Per-image reprojection errors</h3>
        <table class="grid">
          <thead><tr><th>Image</th><th>Error (px)</th></tr></thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>"""

    dist_block = ""
    if R["dist"]:
        d = R["dist"]
        dist_block = f"""
        <tr><th>Distortion</th>
            <td>k1={d['k1']:.3f}, k2={d['k2']:.3f}, p1={d['p1']:.4f}, p2={d['p2']:.4f}, k3={d['k3']:.3f}</td></tr>"""

    warn_list = []
    if R["warn_aspect"]:
        warn_list.append(f"fy/fx = {R['aspect']:.3f} deviates > {(ASPECT_WARN-1)*100:.0f}% from 1.0 (square pixels). "
                         f"Could be true pixel aspect or biased views; consider more diverse angles or checking the stream.")
    if R["warn_princp"]:
        warn_list.append(f"Principal point offset: {R['dx_frac']*100:.1f}% of width, {R['dy_frac']*100:.1f}% of height. "
                         f"This can be real; just note it for downstream geometry.")
    if R["warn_skew"]:
        warn_list.append(f"Non-zero skew term detected ({R['skew']:.6f}); usually near 0.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Calibration Report – {escape(cam_name)}</title>
<style>
body {{ font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; color: #222; }}
h1,h2,h3 {{ margin: 0.2em 0 0.4em; }}
small {{ color:#666; }}
.code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.card {{ border:1px solid #eee; border-radius:12px; padding:16px; margin:14px 0; box-shadow:0 1px 2px rgba(0,0,0,0.04); }}
.grid {{ width:100%; border-collapse: collapse; }}
.grid th, .grid td {{ border:1px solid #eee; padding:8px 10px; text-align:left; }}
.grid .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
.b {{ display:inline-block; padding:.1em .5em; border-radius:999px; font-size:12px; font-weight:600; }}
.ok {{ background:#e6ffed; color:#067d2b; border:1px solid #bbf2c6; }}
.bad {{ background:#fff7e6; color:#8a5a00; border:1px solid #ffe1a3; }}
.pass {{ color:#067d2b; }} .warn {{ color:#8a5a00; }}
.kv th {{ width:250px; vertical-align:top; background:#fafafa; }}
.kv td {{ font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
  <h1>Calibration Report</h1>
  <div><b>Camera:</b> <span class="code">{escape(cam_name)}</span> &nbsp;|&nbsp;
       <b>Intrinsics file:</b> <span class="code">{escape(str(yml_path))}</span> &nbsp;|&nbsp;
       <small>Generated: {now}</small></div>

  <div class="card">
    <h2>Result: {"<span class='pass'>PASS ✅</span>" if R["passed"] else "<span class='warn'>WARN ⚠️</span>"}</h2>
    <table class="grid kv">
      <tr><th>Image size</th><td>{iw} × {ih}</td></tr>
      <tr><th>fx, fy (fy/fx)</th><td>{R['fx']:.3f}, {R['fy']:.3f} &nbsp; ( {R['aspect']:.3f} )</td></tr>
      <tr><th>cx, cy (offset px)</th><td>{R['cx']:.3f}, {R['cy']:.3f} &nbsp; ( {R['dx_px']:.2f}, {R['dy_px']:.2f} )</td></tr>
      <tr><th>Estimated FOV</th><td>H {R['hfov']:.2f}°, V {R['vfov']:.2f}°</td></tr>
      {dist_block}
    </table>
  </div>

  <div class="card">
    <h3>Reprojection error (px)</h3>
    <table class="grid kv">
      <tr><th>Global RMS</th><td class="num">{R['rms']:.3f} {badge(R['ok_rms'])}</td></tr>
      <tr><th>Images used</th><td class="num">{R['n']} {badge(R['ok_images'])}</td></tr>
      <tr><th>Mean / Median</th><td class="num">{R['mean_e']:.3f} / {R['med_e']:.3f}</td></tr>
      <tr><th>p90 / Max / Min</th><td class="num">{R['p90_e']:.3f} / {R['max_e']:.3f} / {R['min_e']:.3f} {badge(R['ok_p90'] and R['ok_max'])}</td></tr>
    </table>
  </div>

  {perimg_table}

  <div class="card">
    <h3>Warnings & Notes</h3>
    {"<ul>" + "".join(f"<li>{escape(w)}</li>" for w in warn_list) + "</ul>" if warn_list else "<p>None</p>"}
  </div>

  <div class="card">
    <h3>Heuristic settings (why we check these)</h3>
    <table class="grid">
      <thead><tr><th>Setting</th><th>Value</th><th>Explanation</th></tr></thead>
      <tbody>
        {"".join(f"<tr><td class='code'>{k}</td><td class='num'>{v}</td><td>{escape(expl)}</td></tr>" for k,v,expl in settings_rows)}
      </tbody>
    </table>
  </div>

</body></html>
"""
    html_path.write_text(html, encoding="utf-8")

if __name__ == "__main__":
    paths = Utils.get_calibration_paths(camera_name)
    yml_path = Path(paths["intrinsics_yml"])
    html_path = Path(paths["intrinsics_report"])

    if not yml_path.exists():
        print(f"[ERROR] YAML not found at {yml_path}")
        sys.exit(1)

    R = analyze(yml_path)
    print_console(yml_path, R)
    write_html(html_path, camera_name, yml_path, R)
    print(f"\nHTML report written to: {html_path}")
