import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# -----------------------------
# Geometry: angle from vertical & apparent angular extent
# -----------------------------
def angular_extension(diameter, h, d):
    """
    diameter, h, d in mm
    Returns:
      alpha_deg: viewing angle from vertical (degrees)
      theta_deg: angular extent (degrees)
    """
    alpha = np.degrees(np.arctan2(d, h))
    # apparent extent along the plane through the optical axis and object
    rads  = np.arctan2(d + diameter/2.0, h) - np.arctan2(d - diameter/2.0, h)
    theta = np.degrees(rads)
    return alpha, theta

# -----------------------------
# Fisheye projection models: r(theta) with theta in degrees, r in mm
# -----------------------------
def equidistant(theta_deg, f):      # r = f * theta (theta in radians)
    th = np.radians(theta_deg); return f * th

def stereographic(theta_deg, f):    # r = 2f * tan(theta/2)
    th = np.radians(theta_deg); return 2.0 * f * np.tan(th/2.0)

def equisolid(theta_deg, f):        # r = 2f * sin(theta/2)
    th = np.radians(theta_deg); return 2.0 * f * np.sin(th/2.0)

def orthographic(theta_deg, f):     # r = f * sin(theta)
    th = np.radians(theta_deg); return f * np.sin(th)

def rectilinear(theta_deg, f):      # r = f * tan(theta)
    th = np.radians(theta_deg); return f * np.tan(th)

# -----------------------------
# Parameters
# -----------------------------
focal_length_mm = 1.2  # mm

# Output image is 2992x2992 and the fisheye circle fills that square.
used_circle_px = 2992
sensor_width_mm, sensor_height_mm = 6.17, 4.55
used_circle_mm = min(sensor_width_mm, sensor_height_mm)   # inscribed circle diameter on sensor
pixel_pitch_mm = used_circle_mm / used_circle_px          # mm per pixel on the *output* fisheye circle

# Scene / object
object_diameter_mm = 50
camera_height_mm = 1500
lateral_displacements_mm = np.linspace(0, 2800, 200)

# Choose your projection model here:
proj = equisolid  # change to equidistant/stereographic/orthographic/rectilinear as needed

# -----------------------------
# Precompute pixels-per-degree as a function of theta (0..~90°)
# -----------------------------
theta_grid = np.linspace(0.0, 90.0, 901)  # degrees, visible hemisphere only
r_mm = proj(theta_grid, focal_length_mm)

# derivative wrt degrees -> mm/degree at each theta
dr_dtheta_mm_per_deg = np.gradient(r_mm, theta_grid)

# convert to pixels/degree via pixel pitch (mm/px)
pixels_per_degree = dr_dtheta_mm_per_deg / pixel_pitch_mm

# make an interpolator (outside 0..90, clamp)
ppd_interp = interp1d(theta_grid, pixels_per_degree, kind='linear',
                      bounds_error=False, fill_value=(pixels_per_degree[0], pixels_per_degree[-1]))

# -----------------------------
# Compute pixels across the object vs lateral displacement
# -----------------------------
alpha_deg, theta_deg = angular_extension(object_diameter_mm, camera_height_mm, lateral_displacements_mm)

# clamp alpha to [0, 89.9] to avoid pathological edge behavior
alpha_deg = np.clip(alpha_deg, 0.0, 89.9)

# pixels across object ≈ (pixels/degree at alpha) * (angular extent in degrees)
ppd_at_alpha = ppd_interp(alpha_deg)                # pixels per degree at the object’s viewing angle
pixels_on_object = ppd_at_alpha * theta_deg         # total pixels across diameter (radial direction)

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(theta_grid, r_mm)
plt.title(f'{proj.__name__} projection (f={focal_length_mm} mm)')
plt.xlabel('Angle from vertical (deg)')
plt.ylabel('Radial distance on sensor (mm)')
plt.grid()

plt.subplot(1,2,2)
plt.plot(theta_grid, dr_dtheta_mm_per_deg)
plt.title('dr/dθ (mm per degree)')
plt.xlabel('Angle from vertical (deg)')
plt.ylabel('mm/degree')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(theta_grid, pixels_per_degree)
plt.title('Pixels per degree vs angle')
plt.xlabel('Angle from vertical (deg)')
plt.ylabel('Pixels per degree')
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(lateral_displacements_mm, alpha_deg)
plt.title('Viewing angle vs lateral displacement')
plt.xlabel('Lateral displacement (mm)')
plt.ylabel('Angle from vertical (deg)')
plt.grid()

plt.subplot(1,2,2)
plt.plot(lateral_displacements_mm, theta_deg)
plt.title('Angular extent vs lateral displacement')
plt.xlabel('Lateral displacement (mm)')
plt.ylabel('Extent (deg)')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(alpha_deg, pixels_on_object)
plt.title('Pixels across object vs viewing angle')
plt.xlabel('Angle from vertical (deg)')
plt.ylabel('Pixels across diameter')
plt.grid()
plt.show()
