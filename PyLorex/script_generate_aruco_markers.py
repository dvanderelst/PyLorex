"""Generate ArUco marker SVGs for laser cutting.

For each marker ID, writes a single-marker SVG. Optionally also writes a
master SVG with all markers laid out in a grid (handy so you only set
cut parameters once).

Three stroke colors are exposed so the cutter can distinguish:
  - the outer sheet-cut rectangle
  - the marker's outer boundary
  - the boundaries of the white bits
Fills stay black/white so the marker still rasterises correctly.
"""
from collections import defaultdict
from math import ceil
from pathlib import Path

import cv2 as cv

# --- parameters ---------------------------------------------------------
DICT_NAME = "DICT_4X4_1000"          # ArUco dictionary, e.g. DICT_5X5_250
IDS = list(range(75, 100))             # which marker IDs to generate
MARKER_MM = 90                     # marker side incl. 1-bit black border
BORDER_MM = 15                      # white quiet-zone border around marker
STROKE_MM = 0.1                      # stroke width for all coloured edges

OUTER_EDGE_COLOR = "#ff0000"         # sheet cut
MARKER_EDGE_COLOR = "#0000ff"        # marker outer boundary
BITS_EDGE_COLOR = "#00aa00"          # white-bit boundaries

WRITE_MASTER = True                  # also emit master files with marker grids
MASTER_COLS = 4                     # grid columns per master file
MASTER_ROWS = 2                      # grid rows per master file
MASTER_GAP_MM = 2.0                  # gap between markers in master file
MASTER_PREFIX = "master"             # output files: master_01.svg, master_02.svg, ...

OUT_DIR = Path(__file__).parent / "Markers" / "generated"
# ------------------------------------------------------------------------


def marker_bits(dictionary, marker_id, border_bits=1):
    grid = dictionary.markerSize + 2 * border_bits
    return cv.aruco.generateImageMarker(dictionary, marker_id, grid, borderBits=border_bits)


def trace_white_outlines(bits):
    """Trace boundary polygons of white-cell regions in cell units."""
    grid = bits.shape[0]
    is_white = (bits == 255)

    edges = []
    for row in range(grid):
        for col in range(grid):
            if not is_white[row, col]:
                continue
            if row == 0 or not is_white[row - 1, col]:
                edges.append(((col, row), (col + 1, row)))
            if col == grid - 1 or not is_white[row, col + 1]:
                edges.append(((col + 1, row), (col + 1, row + 1)))
            if row == grid - 1 or not is_white[row + 1, col]:
                edges.append(((col + 1, row + 1), (col, row + 1)))
            if col == 0 or not is_white[row, col - 1]:
                edges.append(((col, row + 1), (col, row)))

    edges_from = defaultdict(list)
    for i, e in enumerate(edges):
        edges_from[e[0]].append(i)

    used = [False] * len(edges)
    polygons = []
    for start_idx in range(len(edges)):
        if used[start_idx]:
            continue
        used[start_idx] = True
        poly = [edges[start_idx][0], edges[start_idx][1]]
        current = edges[start_idx][1]

        while current != poly[0]:
            candidates = [i for i in edges_from[current] if not used[i]]
            if not candidates:
                break
            # Diagonal-corner case: pick sharpest right turn to stay on
            # the same region (positive cross in SVG y-down coords).
            in_dx = poly[-1][0] - poly[-2][0]
            in_dy = poly[-1][1] - poly[-2][1]

            def turn_score(i):
                e = edges[i]
                return in_dx * (e[1][1] - e[0][1]) - in_dy * (e[1][0] - e[0][0])

            best = max(candidates, key=turn_score)
            used[best] = True
            poly.append(edges[best][1])
            current = edges[best][1]

        polygons.append(poly)
    return polygons


def simplify_polygon(poly):
    if len(poly) < 4:
        return poly
    pts = poly[:-1]
    n = len(pts)
    out = []
    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        curr = pts[i]
        nxt = pts[(i + 1) % n]
        dx1, dy1 = curr[0] - prev_pt[0], curr[1] - prev_pt[1]
        dx2, dy2 = nxt[0] - curr[0], nxt[1] - curr[1]
        if dx1 * dy2 - dy1 * dx2 != 0:
            out.append(curr)
    out.append(out[0])
    return out


def polygons_to_path_d(polygons, cell, offset_x, offset_y):
    parts = []
    for poly in polygons:
        poly = simplify_polygon(poly)
        if len(poly) < 4:
            continue
        x0 = offset_x + poly[0][0] * cell
        y0 = offset_y + poly[0][1] * cell
        parts.append(f"M{x0},{y0}")
        for pt in poly[1:-1]:
            x = offset_x + pt[0] * cell
            y = offset_y + pt[1] * cell
            parts.append(f"L{x},{y}")
        parts.append("Z")
    return " ".join(parts)


def marker_elements(bits, marker_mm, border_mm, x0=0.0, y0=0.0):
    """Return SVG element strings for a single marker, optionally offset."""
    grid = bits.shape[0]
    cell = marker_mm / grid
    total = marker_mm + 2 * border_mm

    polygons = trace_white_outlines(bits)
    white_d = polygons_to_path_d(polygons, cell, x0 + border_mm, y0 + border_mm)

    elems = [
        # outer cut rectangle
        f'<rect x="{x0}" y="{y0}" width="{total}" height="{total}" '
        f'fill="none" stroke="{OUTER_EDGE_COLOR}" stroke-width="{STROKE_MM}"/>',
        # marker body: black fill (engrave) + coloured outer-edge stroke
        f'<rect x="{x0 + border_mm}" y="{y0 + border_mm}" '
        f'width="{marker_mm}" height="{marker_mm}" '
        f'fill="#000000" stroke="{MARKER_EDGE_COLOR}" stroke-width="{STROKE_MM}"/>',
    ]
    if white_d:
        elems.append(
            f'<path d="{white_d}" fill="#ffffff" fill-rule="evenodd" '
            f'stroke="{BITS_EDGE_COLOR}" stroke-width="{STROKE_MM}"/>'
        )
    return elems


def wrap_svg(elements, width_mm, height_mm):
    body = "\n    ".join(elements)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {width_mm} {height_mm}">\n'
        '  <g style="shape-rendering:crispEdges">\n'
        f'    {body}\n'
        '  </g>\n'
        '</svg>\n'
    )


def main():
    if not hasattr(cv.aruco, DICT_NAME):
        raise SystemExit(f"Unknown ArUco dictionary: {DICT_NAME}")
    dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, DICT_NAME))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    short = DICT_NAME.replace("DICT_", "").lower()

    total = MARKER_MM + 2 * BORDER_MM
    bits_by_id = {}
    for marker_id in IDS:
        bits = marker_bits(dictionary, marker_id)
        bits_by_id[marker_id] = bits
        svg = wrap_svg(marker_elements(bits, MARKER_MM, BORDER_MM), total, total)
        path = OUT_DIR / f"{short}-{marker_id}.svg"
        path.write_text(svg)
        print(f"wrote {path}")

    if WRITE_MASTER and IDS:
        cols = max(1, MASTER_COLS)
        rows = max(1, MASTER_ROWS)
        per_master = cols * rows
        step = total + MASTER_GAP_MM
        master_w = cols * total + (cols - 1) * MASTER_GAP_MM
        master_h = rows * total + (rows - 1) * MASTER_GAP_MM
        n_masters = ceil(len(IDS) / per_master)
        digits = max(2, len(str(n_masters)))

        for page in range(n_masters):
            chunk = IDS[page * per_master:(page + 1) * per_master]
            elements = []
            for idx, marker_id in enumerate(chunk):
                r, c = divmod(idx, cols)
                elements.extend(
                    marker_elements(bits_by_id[marker_id], MARKER_MM, BORDER_MM,
                                    x0=c * step, y0=r * step)
                )
            master = wrap_svg(elements, master_w, master_h)
            master_path = OUT_DIR / f"{MASTER_PREFIX}_{page + 1:0{digits}d}.svg"
            master_path.write_text(master)
            print(f"wrote {master_path} ({len(chunk)} markers, "
                  f"ids {chunk[0]}-{chunk[-1]})")


if __name__ == "__main__":
    main()
