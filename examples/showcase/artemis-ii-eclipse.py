"""
========================
Artemis-II Solar Eclipse
========================

This example demonstrates how to perform astrometric calibration of a solar eclipse image taken by a camera aboard the Artemis-II spacecraft.
Starting from raw JPEG images with EXIF metadata, we extract the observation time.
We then use the known positions of the Moon, Sun, and solar system planets retrieved from JPL Horizons via sunpy.coordinates.get_horizons_coord to build an initial helioprojective WCS using `sunpy.map.make_fitswcs_header`.
The camera roll angle is refined by comparing the predicted and detected pixel positions of Saturn, Mars, and Mercury, identified automatically using `photutils.detection.DAOStarFinder`.
Finally, the residual radial barrel distortion is modeling using a single SIP coefficient k1 derived from the planet positions.
The resulting calibrated sunpy.map.Map allows solar corona features to be precisely located.

"""
from pathlib import Path

import exifread
import matplotlib
import numpy as np
import requests
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from photutils.detection import DAOStarFinder
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord, solar_system_ephemeris
from astropy.stats import mad_std
from astropy.time import Time
from astropy.wcs import WCS

from sunpy.coordinates import Helioprojective, SphericalScreen, get_horizons_coord
from sunpy.map import Map, make_fitswcs_header

# Accurate planetary ephemeris
solar_system_ephemeris.set('de440s')

##############################################################################
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

fig, ax = plt.subplots()
ax.imshow(artemis_image_rbg, origin="lower")
ax.set_axis_off()

##############################################################################
# Extract meta data

with Path(filename).open("rb") as f:
    tags = exifread.process_file(f)

obsdate, obstime= tags['EXIF DateTimeDigitized'].values.split(" ")
obsdate = obsdate.replace(":", "-")
obstime = Time(f"{obsdate}T{obstime}")

hours, _ = [int(part) for part in tags['EXIF OffsetTime'].values.split(":")]
offset = hours*u.hour

obstime = obstime
# obstime = obstime + offset # It seems like the timezone or offset is set incorrectly

##############################################################################
# Extract ROI around the moon

slice_y = slice(1466,3979)
slice_x = slice(2100,4667)
roi = artemis_image[slice_y, slice_x]

##############################################################################
# Get coordinates at observation time and 48 hour interval

NAIF_IDS = {
    "artemis_ii": -1024,
    "moon": 301,
    "earth": 399,
    "sun": 10
}

times = np.linspace(obstime - (24*u.hour), obstime + (24*u.hour), 100)

coords =  {key: get_horizons_coord(str(value), obstime) for key, value in NAIF_IDS.items()}
tracks =  {key: get_horizons_coord(str(value), times) for key, value in NAIF_IDS.items()}

##############################################################################
# Plot general orbit location with zoom the location of Artemtis-II to inspect
# and make sure alignment make sense for the eclipse image.

sun_artemis = np.vstack([coords[n].cartesian.xyz for n in ['sun', 'artemis_ii']])

# Zoom region limits (matching your current ax[1] limits)
x1, x2 = 0.9930, 0.9992
y1, y2 = -0.10856, -0.10806

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 5))

for ax in [ax0, ax1]:
    ax.plot(sun_artemis[:,0], sun_artemis[:,2], 'k-',
            label="Sun Artemis-II line", linewidth=0.5)
    for name, coord in coords.items():
        line = ax.plot(coord.cartesian.xyz[0], coord.cartesian.xyz[2], 'o',
                       label=name.title().replace('i', 'I'))
        ax.plot(tracks[name].cartesian.xyz[0], tracks[name].cartesian.xyz[2],
                color=line[0].get_color())

ax0.legend()
ax1.set_xlim(x1, x2)
ax1.set_ylim(y1, y2)

# Draw zoom box on ax0
rect = Rectangle((x1, y1), x2-x1, y2-y1,
                  linewidth=1, edgecolor='k', facecolor='none', linestyle='--')
ax0.add_patch(rect)

# Draw connecting lines between box and zoom panel
mark_inset(ax0, ax1, loc1=2, loc2=3, fc="none", ec="k", linewidth=0.5, linestyle='--')

##############################################################################
# Edge detection and Hough filtering

edges = canny(roi, sigma=2)

hough_radii = np.arange(np.floor(np.mean(edges.shape) / 4), np.ceil(np.mean(edges.shape) / 2), 10)
hough_res = hough_circle(edges, hough_radii)

accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

##############################################################################
# Plot edge detection results

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(9, 3))
ax[0].imshow(artemis_image[slice_y, slice_x])
ax[0].set_title("Original")
ax[1].imshow(edges)
ax[1].set_title("Canny")
circ = Circle(
    [cx, cy], radius=radii, facecolor="none", edgecolor="red", linewidth=2, linestyle="dashed", label="Hough fit"
)
ax[2].imshow(artemis_image[slice_y, slice_x])
ax[2].add_patch(circ)
ax[2].set_title("Original with fit")
fig.legend()


##############################################################################
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
# Make map

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

def plot_artemis_map(amap, moon_coord, planets):
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": amap}, figsize=(9, 4))
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
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    return fig, ax

##############################################################################
# Plot map and positions of Saturn, Mar and Mercury to see if the WCS is good

PLANET_NAIFS = {
    "mercury": 199,
    # "venus": 299,
    # "earth": 399,
    "mars": 499,
    # "jupiter": 599,
    "saturn": 699,
    # "uranus": 799,
    # "neptune": 899
}
planets = {name: get_horizons_coord(str(id), obstime) for name, id in PLANET_NAIFS.items()}

fig, ax = plot_artemis_map(artemis_map, moon_hpc, planets)

##############################################################################
# Can see a pretty clear roll so use position of Saturn to obtain an estimate of the roll
# Abuse `DAOStarFinder` to find the planets

data = artemis_map.data  # SunPy map data

bkg_sigma = mad_std(data)

daofind = DAOStarFinder(fwhm = 20.0, threshold = 0.75, brightest=3)
sources = daofind(data)

planets_x = sources['xcentroid']
planets_y = sources['ycentroid']

star_coords = artemis_map.pixel_to_world(planets_x * u.pix, planets_y * u.pix)

fig, ax = plot_artemis_map(artemis_map, moon_hpc, planets)
with SphericalScreen(coords["artemis_ii"]):
    ax.plot_coord(star_coords, 's', markerfacecolor='none')

##############################################################################
# There is a clear rotation so lets use the expected and actual positions of
# the planets to determine the roll angle.
#
# Need to know which pixel position correspond to which planet from the map
# above can see that in terms of distance from the Moon they are Saturn, Mars
# and Mercury in order. So we will sort the by the separation angle between
# the moon center and the planet

# Saturn, Mars, Mercury
with SphericalScreen(coords["artemis_ii"]):
    sep = moon_hpc.transform_to(star_coords.frame).separation(star_coords)
planet_index = np.argsort(sep)

moon_pixel = CartesianRepresentation(*artemis_map.wcs.world_to_pixel(moon_hpc), 0)*u.pix

planets_temp = SkyCoord([planets[name] for name in ["saturn", "mars", "mercury"]])
planets_pixel = CartesianRepresentation(*artemis_map.wcs.world_to_pixel(planets_temp), 0) * u.pix

actual_planets_pixel = CartesianRepresentation(planets_x[planet_index], planets_y[planet_index], [0] * 3) * u.pix

vec_expected = planets_pixel - moon_pixel
vec_actual = actual_planets_pixel - moon_pixel
roll_angles = -np.arccos(vec_expected.dot(vec_actual) / (vec_expected.norm() * vec_actual.norm()))


weights = sep[planet_index].to(u.deg).value
roll_angles_weighted = np.average(roll_angles, weights=weights)
print(roll_angles.to('deg'))
print(roll_angles.mean().to('deg'))
print(f"Weighted roll: {np.degrees(roll_angles_weighted):.4f} deg")
##############################################################################
# Use derived roll and make new map

header_roll = make_fitswcs_header(
    artemis_image,
    moon_hpc,
    reference_pixel=u.Quantity([im_cx, im_cy]),
    scale=u.Quantity([plate_scale, plate_scale]),
    rotation_angle=-roll_angles_weighted.to('deg')
)

artemis_map_roll = Map(artemis_image, header_roll)

##############################################################################
# Plot map and positions of Saturn, Mar and Mercury if the WCS is good.
# Seems to be some residual distortion

fig, ax = plot_artemis_map(artemis_map_roll, moon_hpc, planets)

##############################################################################
# Lets see if it might be lense type distortion

cx, cy = artemis_map_roll.wcs.wcs.crpix
r_actual, r_predicted = [], []
for i, name in enumerate(["saturn", "mars", "mercury"]):
    hpc = planets[name].transform_to(artemis_map_roll.coordinate_frame)
    px, py = artemis_map_roll.wcs.world_to_pixel(hpc)
    ax, ay = planets_x[planet_index][i], planets_y[planet_index][i]
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
k1_estimates_reliable = k1_estimates[1:]  # Mars and Mercury only
r_reliable = r_predicted[1:]
k1 = np.average(k1_estimates_reliable, weights=r_reliable)
print(f"k1 (Mars+Mercury only): {k1:.6e} pix^-2")

##############################################################################
# Create a SIP header and WCS see if the SIP position are better

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
     ax, ay   = planets_x[planet_index][i], planets_y[planet_index][i]
     print(f"{name}: residual without SIP=({ax-px_nosip[0]:.1f}, {ay-px_nosip[1]:.1f})  "
           f"with SIP=({ax-px_sip[0]:.1f}, {ay-px_sip[1]:.1f})")

##############################################################################
# Switch to astropy as sunpy map seem to be droppig the SIP headers

fig, ax = plt.subplots(subplot_kw={"projection": wcs_sip}, figsize=(9, 5))
ax.imshow(artemis_image, origin="lower", cmap="gray")

for name, coord in planets.items():
     hpc = coord.transform_to(artemis_map_roll.coordinate_frame)
     px, py = wcs_sip.all_world2pix([[hpc.Tx.to(u.deg).value,
                                      hpc.Ty.to(u.deg).value]], 0)[0]
     ax.plot(px, py, 'o', markerfacecolor='none', label=name.title())

# lunar limb
theta = np.linspace(0, 2*np.pi, 100)
limb_tx = (moon_hpc.Tx + np.sin(theta*u.rad)*moon_obs).to(u.deg).value
limb_ty = (moon_hpc.Ty + np.cos(theta*u.rad)*moon_obs).to(u.deg).value
limb_px, limb_py = wcs_sip.all_world2pix(np.column_stack([limb_tx, limb_ty]), 0).T
ax.plot(limb_px, limb_py, 'b-', label="Lunar Limb")
ax.plot_coord(moon_hpc, 'b+', label="Lunar Center" )

ax.legend()
ax.set_xlim(0, artemis_image.shape[1])
ax.set_ylim(0, artemis_image.shape[0])

#######################################################################
# A bug in sunpy map mean for the moment need to manually apply the
# sip correction to the planet positions

def apply_sip_to_coords(coords_dict, wcs_sip, sunpy_map):
    """Return a new dict of SkyCoords with SIP distortion applied."""
    corrected = {}
    for name, coord in coords_dict.items():
        hpc = coord.transform_to(sunpy_map.coordinate_frame)
        tx = np.atleast_1d(hpc.Tx.to(u.deg).value)
        ty = np.atleast_1d(hpc.Ty.to(u.deg).value)
        px, py = wcs_sip.all_world2pix(np.column_stack([tx, ty]), 0).T
        corrected[name] = sunpy_map.pixel_to_world(px*u.pix, py*u.pix)
    return corrected

planets_sip = apply_sip_to_coords(planets, wcs_sip, artemis_map_roll)
fig, ax = plot_artemis_map(artemis_map_roll, moon_hpc, planets_sip)
ax.set_title(f"Artemis-II Solar Eclipse {obstime}")

# sphinx_gallery_thumbnail_number = -1
