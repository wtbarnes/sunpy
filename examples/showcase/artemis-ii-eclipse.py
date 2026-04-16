"""
========================
Artemis-II Solar Eclipse
========================

This example demonstrates how to process a solar eclipse image taken by Artemis-II crew using digital camera aboard the spacecraft during their lunar flyby on April 7, 2026.
Due to the relative position of the Artemis-II spacecraft and the moon this eclipse lasted nearly 54 minutes of totality far exceeding what is possible on Earth
Starting from raw JPEG images with EXIF metadata the observation time is extracted.
Then the known positions of the Moon, Sun, and planets are retrieved from JPL Horizons via `sunpy.coordinates.get_horizons_coord` to build an initial Helioprojective WCS using `sunpy.map.header_helper.make_fitswcs_header`.
The camera roll angle is refined by comparing the predicted and detected pixel positions of Saturn, Mars, and Mercury, identified automatically using a medial filter and `skimage.feature.peak_local_max`.
Finally, the residual radial barrel distortion is modeling using a single SIP coefficient k1 derived from the planet positions.
The resulting calibrated sunpy.map.Map allows solar corona features to be located.


"""
from pathlib import Path

import exifread
import hvpy
import matplotlib
import numpy as np
import requests
from hvpy.datasource import DataSource
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import medfilt2d
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import canny, peak_local_max
from skimage.transform import hough_circle, hough_circle_peaks

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.wcs import WCS

import sunpy.map
from sunpy.coordinates import Helioprojective, SphericalScreen, get_horizons_coord
from sunpy.map import Map
from sunpy.map.header_helper import make_fitswcs_header
from sunpy.util.config import get_and_create_download_dir

# Accurate planetary ephemeris
solar_system_ephemeris.set('de440s')

##############################################################################
# Getting and convert the image
# =============================
#
# Download and read in the raw image data taken by crew on Artemis-II

url = "https://images-assets.nasa.gov/image/art002e009301/art002e009301~orig.jpg"
filename = url.split("/")[-1]
with requests.get(url, stream=True) as res:
    res.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            f.write(chunk)

artemis_image_rbg = np.flipud(matplotlib.image.imread(filename))
artemis_image = rgb2gray(artemis_image_rbg)

downsampled = True
# Downsample to reasonable size for processing and visualization
# can be set to False if running locally for full resolution
if downsampled:
    artemis_image = transform.rescale(artemis_image, 1/6, anti_aliasing=True)

fig, ax = plt.subplots()
ax.imshow(artemis_image_rbg, origin="lower")
ax.set_axis_off()
del artemis_image_rbg


##############################################################################
# Extract meta data (optional)
#
# Requires extra package (exifread)

with Path(filename).open("rb") as f:
    tags = exifread.process_file(f)

obsdate, obstime= tags['EXIF DateTimeDigitized'].values.split(" ")
obsdate = obsdate.replace(":", "-")
obstime = Time(f"{obsdate}T{obstime}")
print(obstime)

hours, _ = [int(part) for part in tags['EXIF OffsetTime'].values.split(":")]
offset = hours*u.hour

# obstime = obstime + offset # It seems like the timezone or offset is set incorrectly

##############################################################################
# Get Coordinates
# ===============
#
# Get coordinates at observation time and 48 hour interval

NAIF_IDS = {
    "artemis_ii": -1024,
    "moon": 301,
    "sun": 10,
    "mercury": 199,
    "venus": 299,
    "earth": 399,
    "mars": 499,
    "jupiter": 599,
    "saturn": 699,
    "uranus": 799,
    "neptune": 899
}

times = np.linspace(obstime - (24*u.hour), obstime + (24*u.hour), 100)

coords =  {name: get_horizons_coord(str(id), obstime) for name, id in NAIF_IDS.items()}

##############################################################################
# Find and fit Moon's limb and center
# ===================================
#
# Edge detection and Hough filtering to obtain the moons limb and center.
#
# First pass on a downscaled version to get an estimate and use this estimate
# to extract and ROI for full resolution pass

print("starting low res pass")
scale = 0.5 if downsampled else 0.1
down_scaled = transform.rescale(artemis_image, scale, anti_aliasing=True)

 # Edge detection
edges = canny(down_scaled, sigma=2)

 # Radius range in scaled image (diameter ~1/3 of image height)
h, w = down_scaled.shape
radii = np.arange(0.25*h, 0.4*h, 10)

 # Hough
hough_res = hough_circle(edges, radii)
accums, cx, cy, rad = hough_circle_peaks(hough_res, radii, total_num_peaks=1)
del hough_res

 # Scale back to original resolution
moon_x = int(cx[0] / scale)
moon_y = int(cy[0] / scale)
moon_r = rad[0] / scale
roi_ext = int(1.05*moon_r)

slice_y = slice(moon_y-roi_ext, moon_y+roi_ext)
slice_x = slice(moon_x-roi_ext, moon_x+roi_ext)
print(f"Low res pass moon_x: {moon_x}, moon_y: {moon_y}, moon_r: {moon_r}")

##############################################################################
# Full resolution pass within ROI

roi = artemis_image[slice_y, slice_x]

edges = canny(roi, sigma=2)

hough_radii = np.linspace(edges.shape[0] / 2.5, edges.shape[0] / 2, 30)
hough_res = hough_circle(edges, hough_radii).astype(np.float32) #  reduce peak memory usage

accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

print(f"High res pass moon_x: {cx[0] + slice_x.start}, moon_y: {cy[0]+slice_y.start}, moon_r: {radii}")

##############################################################################
# Plot edge detection Hough filtering results

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(9, 3))
ax[0].imshow(artemis_image[slice_y, slice_x])
ax[0].set_title("Original")
ax[1].imshow(edges)
ax[1].set_title("Canny")
circ = Circle(
    np.hstack([cx, cy]), radius=radii[0], facecolor="none", edgecolor="red",
    linewidth=2, linestyle="dashed", label="Hough fit")
ax[2].imshow(artemis_image[slice_y, slice_x])
ax[2].add_patch(circ)
ax[2].set_title("Original with fit")
fig.legend()


##############################################################################
# Create meta data
# ================
#
# Build up meta data required to make a map

im_cx = (cx[0] + slice_x.start) * u.pix
im_cy = (cy[0] + slice_y.start) * u.pix
im_radius = radii[0] * u.pix

moon = SkyCoord(coords['moon'], observer=coords['artemis_ii'])
R_moon = 0.2725076 *  u.R_earth  # IAU mean radius
dist_moon = SkyCoord(coords['artemis_ii']).separation_3d(moon)
moon_obs = np.arcsin(R_moon / dist_moon).to("arcsec")
print(moon_obs)

plate_scale = moon_obs / im_radius
print(plate_scale)


##############################################################################
# Make a Map
# ==========
#
# Make map using the metadata obtained so far and `sunpy.map.header_helper.make_fitswcs_header`

frame = Helioprojective(observer=coords['artemis_ii'], obstime=obstime)
moon_hpc = coords['moon'].transform_to(frame)

header = make_fitswcs_header(
    artemis_image,
    moon_hpc,
    reference_pixel=u.Quantity([im_cx, im_cy]),
    scale=u.Quantity([plate_scale, plate_scale])
)

artemis_map = Map(artemis_image, header)

##############################################################################
# Reusable plot helper

def plot_artemis_map(amap, moon_coord, planets, reset_lim=True, legend=True, figsize=(9,4), **kwargs):
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": amap}, figsize=(9, 4), **kwargs)
    amap.plot(axes=ax)
    amap.draw_grid(axes=ax)
    amap.draw_limb(axes=ax, label='Sun')
    ax.coords[0].set_format_unit(u.deg)
    ax.coords[1].set_format_unit(u.deg)

    ax.plot_coord(moon_coord, 'b+', label="Lunar Center")
    theta = np.linspace(0, 360, 100) * u.deg
    lunar_limb = np.vstack([moon_hpc.Tx + np.sin(theta) * moon_obs, moon_hpc.Ty + np.cos(theta) * moon_obs])
    with SphericalScreen(amap.observer_coordinate):
        ax.plot_coord(SkyCoord(*lunar_limb, frame=amap.coordinate_frame), label="Lunar Limb")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for name, coord in planets.items():
        ax.plot_coord(coord, 'o', markerfacecolor='none', label=name.title())
    if reset_lim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    return fig, ax

##############################################################################
# Plot map and positions of planets to see what should be visible

planets = {name: coord for name, coord in coords.items() if name not in ["sun", "moon", "artemis_ii"]}

fig, ax = plot_artemis_map(artemis_map, moon_hpc, planets, reset_lim=False)
fig.tight_layout()

##############################################################################
# So Mercury, Mars, Saturn and Neptune are all in the image but Neptune's not
# visible as to distant/faint.

planets = {name: coords[name] for name in ["mercury", "mars", "saturn"]}

fig, ax = plot_artemis_map(artemis_map, moon_hpc, planets, reset_lim=False)
fig.tight_layout()

##############################################################################
# Find roll angle
# ===============
#
# Can see a pretty clear roll so use positions of the planets to estimate the
# camera orientation or roll. Median filter the image to remove most stars and
# cosmic rays and then use `skimage.feature.peak_local_max` to find the
# remaining peaks which should be the planets if full resolution.

if downsampled:
    planets_pixels = peak_local_max(artemis_image, threshold_abs=0.9, num_peaks=3, min_distance=30)
else:
    artemis_median_img = medfilt2d(artemis_image, kernel_size=5)
    planets_pixels = peak_local_max(artemis_median_img, threshold_abs=0.9, num_peaks=3, min_distance=30)
    del artemis_median_img

planets_pix_x = planets_pixels[:,1]
planets_pix_y = planets_pixels[:,0]

planet_coords = artemis_map.pixel_to_world(planets_pix_x * u.pix, planets_pix_y * u.pix)

##############################################################################
# Verify we've found the planets.

fig, ax = plot_artemis_map(artemis_map, moon_hpc, planets)
with SphericalScreen(coords["artemis_ii"]):
    ax.plot_coord(planet_coords, 's', markerfacecolor='none')

##############################################################################
# Need to know which pixel position correspond to which planets from the map
# above can see that in terms of distance from the Moon they are Saturn, Mars
# and Mercury in order. So we will sort the by the separation angle between
# the moon's center and the planet

# Saturn, Mars, Mercury
with SphericalScreen(coords["artemis_ii"]):
    sep = moon_hpc.transform_to(planet_coords.frame).separation(planet_coords)
planet_index = np.argsort(sep)
actual_planets_pixel = CartesianRepresentation(planets_pix_x[planet_index], planets_pix_y[planet_index], [0] * 3) * u.pix

moon_pixel = CartesianRepresentation(*artemis_map.wcs.world_to_pixel(moon_hpc), 0)*u.pix

# to match the order of the actual_planets_pixel
planets_temp = SkyCoord([planets[name] for name in ["saturn", "mars", "mercury"]])
planets_pixel = CartesianRepresentation(*artemis_map.wcs.world_to_pixel(planets_temp), 0) * u.pix

vec_expected = planets_pixel - moon_pixel
vec_actual = actual_planets_pixel - moon_pixel
roll_angles = -np.arccos(vec_expected.dot(vec_actual) / (vec_expected.norm() * vec_actual.norm()))

weights = sep[planet_index].to(u.deg).value
roll_angles_weighted = np.average(roll_angles, weights=weights)
print(roll_angles.to('deg'))
print(roll_angles.mean().to('deg'))
print(f"Weighted roll: {np.degrees(roll_angles_weighted):.4f} deg")

##############################################################################
# Use derived roll and make new header and map

header_roll = make_fitswcs_header(
    artemis_image,
    moon_hpc,
    reference_pixel=u.Quantity([im_cx, im_cy]),
    scale=u.Quantity([plate_scale, plate_scale]),
    rotation_angle=-roll_angles_weighted.to('deg')
)

artemis_map_roll = Map(artemis_image, header_roll)

##############################################################################
# Plot map and positions of Saturn, Mar and Mercury to see if the WCS is good.
# Seems to be some residual distortion that get worse towards the edges.

fig, ax = plot_artemis_map(artemis_map_roll, moon_hpc, planets)

##############################################################################
# Correct optical distortion
# ==========================
#
# Assume this is some kind of lens distortion (barrel or pincushion) derive
# the correction from the observer vs actual planet positions.

cx, cy = artemis_map_roll.wcs.wcs.crpix
r_actual, r_predicted = [], []
for i, name in enumerate(["saturn", "mars", "mercury"]):
    hpc = planets[name].transform_to(artemis_map_roll.coordinate_frame)
    px, py = artemis_map_roll.wcs.world_to_pixel(hpc)
    ax, ay = planets_pix_x[planet_index][i], planets_pix_y[planet_index][i]
    r_predicted.append(np.sqrt((px - cx)**2 + (py - cy)**2))
    r_actual.append(np.sqrt((ax - cx)**2 + (ay - cy)**2))

r_predicted = np.array(r_predicted)
r_actual    = np.array(r_actual)

k1_estimates = (r_actual/r_predicted - 1) / r_predicted**2
weights = r_predicted  # weight by distance — Mercury most reliable
k1 = np.average(k1_estimates, weights=weights)
print(f"k1 per planet: {k1_estimates}")
print(f"Weighted k1:   {k1:.6e} pix^-2")

# Use only Mars and Mercury — Saturn too close to center and fit error
k1_estimates_reliable = k1_estimates[1:]
r_reliable = r_predicted[1:]
k1 = np.average(k1_estimates_reliable, weights=r_reliable)
print(f"k1 (Mars+Mercury only): {k1:.6e} pix^-2")

##############################################################################
# Create a SIP header, WCS and check if the SIP improves positions

header_sip = artemis_map_roll.fits_header.copy()
header_sip['CTYPE1'] = 'HPLN-TAN-SIP'
header_sip['CTYPE2'] = 'HPLT-TAN-SIP'
header_sip['A_ORDER'] = 3
header_sip['B_ORDER'] = 3
header_sip['A_3_0'] = -k1
header_sip['A_1_2'] = -k1
header_sip['B_0_3'] = -k1
header_sip['B_2_1'] = -k1
header_sip['A_DMAX'] = 1.0
header_sip['B_DMAX'] = 1.0
wcs_sip = WCS(header_sip)

for i, name in enumerate(["saturn", "mars", "mercury"]):
     hpc = planets[name].transform_to(artemis_map_roll.coordinate_frame)
     px_nosip = wcs_sip.wcs_world2pix([[hpc.Tx.to(u.deg).value, hpc.Ty.to(u.deg).value]], 0)[0]
     px_sip   = wcs_sip.all_world2pix([[hpc.Tx.to(u.deg).value, hpc.Ty.to(u.deg).value]], 0)[0]
     ax, ay   = planets_pix_x[planet_index][i], planets_pix_y[planet_index][i]
     print(f"{name}: residual without SIP=({ax-px_nosip[0]:.1f}, {ay-px_nosip[1]:.1f})  "
           f"with SIP=({ax-px_sip[0]:.1f}, {ay-px_sip[1]:.1f})")

#######################################################################
# We then construct a new Map with this corrected header.

artemis_map_final = Map((artemis_image, header_sip))
fig, ax = plot_artemis_map(artemis_map_final, moon_hpc, planets)
ax.set_title(f"Artemis-II Solar Eclipse {obstime}")
fig.tight_layout()


################################################################################
# Overplotting Coronagraph Images
# ===============================
# In this section we will fetch images of the near corona from LASCO
# and overplot them on the eclipse photo.

# First step is to fetch the images from Helioviewer.
lasco_c2_file = hvpy.save_file(hvpy.getJP2Image(obstime.datetime,
                                                 DataSource.LASCO_C2.value),
                                filename=get_and_create_download_dir() + "/LASCO_C2.jp2", overwrite=True)
lasco_c2_map = Map(lasco_c2_file)
lasco_c3_file = hvpy.save_file(hvpy.getJP2Image(obstime.datetime,
                                                 DataSource.LASCO_C3.value),
                                filename=get_and_create_download_dir() + "/LASCO_C3.jp2", overwrite=True)
lasco_c3_map = Map(lasco_c3_file)

################################################################################
# Next we reproject the LASCO images to the same WCS as the Artemis eclipse photo.
with SphericalScreen(coords["artemis_ii"]):
    c3_map_img = lasco_c3_map.reproject_to(artemis_map_final.wcs)
    c2_map_img = lasco_c2_map.reproject_to(artemis_map_final.wcs)


################################################################################
# As the final step we will crop the LASCO C3 image to the limb of the moon.

# Calculate coordinates for each pixel in the map.
all_hpc = sunpy.map.all_coordinates_from_map(c3_map_img)

# Create a coordinate object for the center of the moon in the reporjected C3 frame.
moon_cen = SkyCoord(moon_hpc.Tx, moon_hpc.Ty, frame=c3_map_img.coordinate_frame)

# Calculate the angular offset from the center of the moon for each pixel.
moon_cen_offsets = all_hpc.separation(moon_cen)

# Create a mask which is True for all offsets greater than the
# observed angular width of the moon.
c3_map_img.mask = np.logical_or(
    moon_cen_offsets >= moon_obs,
    # Also mask out the parts of the image with no data
    c3_map_img.data < 10,
)

# Mask out the parts of the C2 image with no data
c2_map_img.mask = c2_map_img.data < 10

# Overplot both LASCO images, with autoalign off as we already reprojected them.
c3_map_img.plot(axes=ax, autoalign=False)
c2_map_img.plot(axes=ax, autoalign=False)

# sphinx_gallery_thumbnail_number = -1
