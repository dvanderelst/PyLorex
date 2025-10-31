import numpy as np
import math

def nx_from_mp_fov(mp, hfov_deg, vfov_deg):
    """
    Estimate horizontal and vertical pixel counts (Nx, Ny) for a rectilinear camera
    given total megapixels and horizontal/vertical FOVs in degrees.

    Args:
        mp (float): Total megapixels (e.g., 8 for 8MP)
        hfov_deg (float): Horizontal field of view (degrees)
        vfov_deg (float): Vertical field of view (degrees)

    Returns:
        tuple: (Nx, Ny)
            Nx: horizontal pixel count
            Ny: vertical pixel count
    """
    total_pixels = mp * 1e6
    aspect_ratio = math.tan(math.radians(hfov_deg / 2)) / math.tan(math.radians(vfov_deg / 2))
    nx = math.sqrt(total_pixels * aspect_ratio)
    ny = nx / aspect_ratio
    return round(nx), round(ny)


def pixels_across_rectilinear_nadir(
    d_mm: float,           # object diameter (mm)
    h_mm: float,           # camera height above floor (mm)
    fx_px: float = None,   # focal length in pixels (if known)
    nx: int = None,        # image width in pixels
    fov_deg: float = None # FOV in degrees (if using HFOV)
) -> float:
    """
    Returns pixels across the object's diameter for a rectilinear camera pointing straight down.
    Independent of lateral displacement, assuming negligible distortion.
    Provide either fx_px OR (Nx and HFOV_deg).
    """
    if fx_px is None:
        if nx is None or fov_deg is None: raise ValueError("Provide either fx_px, or Nx and HFOV_deg.")
        hfov = np.radians(fov_deg)
        fx_px = (nx / 2.0) / np.tan(hfov / 2.0)
    return (d_mm / h_mm) * fx_px

camera_height_mm = 3000 # mm
diameter_object = 60 # mm
horizontal_fov = 100 # degrees
vertical_fov = 54 # degrees
mp = 8 # megapixels

hpixels, vpixels = nx_from_mp_fov(mp, horizontal_fov, vertical_fov)

hpixels_object = pixels_across_rectilinear_nadir(
    diameter_object,
    camera_height_mm,
    nx=hpixels,
    fov_deg=horizontal_fov
)

vpixels_object = pixels_across_rectilinear_nadir(
    diameter_object,
    camera_height_mm,
    nx=vpixels,
    fov_deg=vertical_fov
)

print('horizontal pixels across object:', hpixels_object)
print('vertical pixels across object:', vpixels_object)

