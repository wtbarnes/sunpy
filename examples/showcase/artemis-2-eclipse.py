"""
========================
Artemis-II Solar Eclipse
========================

Analyse Artemis-II solar eclipse images

"""
from pathlib import Path

import exifread
import matplotlib
import numpy as np
import requests
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
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

solar_system_ephemeris.set('de440s')

##############################################################################
# Download and read in the raw image data

url = "https://images-assets.nasa.gov/image/art002e009301/art002e009301~orig.jpg"
# url = "https://images-assets.nasa.gov/image/art002e009575/art002e009575~orig.jpg"
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
# Correct lens distortion

# height, width = artemis_image_rbg.shape[:2]
# focal_length = tags["EXIF FocalLength"].values[0]
# aperture = tags["EXIF FNumber"].values[0]
#
# db = lensfunpy.Database()
# camera = db.find_cameras('NIKON CORPORATION', 'NIKON Z 9')[0]
# lens = db.find_lenses(camera, lens='35mm f/2D')[0]
#
# print(f"Camera : {camera}")
# print(f"Lens   : {lens}")
#
# def remap(channel, coords):
#     """
#     channel : 2-D array (H, W)
#     coords  : lensfunpy coord array for this channel, shape (H, W, 2)
#               where [..., 0] = x (column) and [..., 1] = y (row)
#     """
#     row_coords = coords[:, :, 1]  # y → row index
#     col_coords = coords[:, :, 0]  # x → col index
#     return map_coordinates(
#         channel,
#         [row_coords, col_coords],
#         order=3,  # bicubic — change to 1 for bilinear, 5 for lanczos-like
#         mode='nearest',  # edge handling
#         prefilter=True,
#     ).clip(0, 255).astype(np.uint8)
#
# mod = lensfunpy.Modifier(lens, camera.crop_factor, width, height)
# mod.initialize(focal_length, aperture)
#
# undist_coords = mod.apply_geometry_distortion()  # (H, W, 2)
#
# artemis_image_undistorted = np.stack([
#     remap(artemis_image_rbg[:, :, ch], undist_coords)
#     for ch in range(3)
# ], axis=-1)
#
# artemis_image = rgb2gray(artemis_image_undistorted)

##############################################################################
# Extract ROI around the moon

if filename == "art002e009575~orig.jpg":
    slice_x = slice(2060, 3450)
    slice_y =  slice(995,2310)
elif filename == "art002e009301~orig.jpg":
    slice_y = slice(1466,3979)
    slice_x = slice(2100,4667)

roi = artemis_image[slice_y, slice_x]

##############################################################################
# Get coordinates at observation times and tracks

NAIF_IDS = {
    "artemis_ii": -1024,
    "moon": 301,
    "earth": 399,
    "sun": 10
}

times = np.linspace(obstime - (24*u.hour), obstime + (24*u.hour), 100)

coords =  {key: get_horizons_coord(str(value), obstime) for key, value in NAIF_IDS.items()}
tracks =  {key: get_horizons_coord(str(value), times) for key, value in NAIF_IDS.items()}

sun_artemis = np.vstack([coords[n].cartesian.xyz for n in ['sun', 'artemis_ii']])

fig, ax = plt.subplots(1, 2)
ax[0].plot(sun_artemis[:,0], sun_artemis[:,2], 'k-', label="Sun Artemis-II line", linewidth=0.5)
ax[1].plot(sun_artemis[:,0], sun_artemis[:,2], 'k-', label="Sun Artemis-II line", linewidth=0.5)
for name, coord in coords.items():
    for i in [0, 1]:
        line = ax[i].plot(coord.cartesian.xyz[0], coord.cartesian.xyz[2], 'o',
                          label=name.title().replace('i', 'I'))
        ax[i].plot(tracks[name].cartesian.xyz[0], tracks[name].cartesian.xyz[2], color=line[0].get_color())

ax[0].legend()
ax[1].set_xlim(0.9930, 0.9992)
ax[1].set_ylim(-0.10856, -0.10806)

##############################################################################
# Edge detection and Hough filtering

edges = canny(roi, sigma=2)

hough_radii = np.arange(np.floor(np.mean(edges.shape) / 4), np.ceil(np.mean(edges.shape) / 2), 10)
hough_res = hough_circle(edges, hough_radii)

accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

##############################################################################
# Plot edge detection results

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(9.5, 6))
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
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": amap}, figsize=(10, 5), dpi=150)
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
# above can see that interns of distance from the Moon they are Saturn, Mars
# and Mercury in order. So we will sort the pixels by the separation angle
# between the moon center and the planet

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
# Lets see if it might be some residual lense distortion

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

# Use only Mars and Mercury — Saturn too close to center to constrain k1
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

fig, ax = plt.subplots(subplot_kw={"projection": wcs_sip}, figsize=(12, 8))
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
# For sunpy map need to manually apply the sip correction to the planet
# positions

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


##############################################################################
# Try to use the positions of the planets to fit the wcs more accurately

# wcs_fitted = fit_wcs_from_points(actual_planets_pixel.xyz[:2,:].to_value(u.pixel),
#                                  planets_temp.transform_to(artemis_map_roll.coordinate_frame),
#                                  projection=artemis_map_roll.wcs)
#
# m = Map(artemis_image, wcs_fitted)
#
# fig, ax = plot_artemis_map(m, moon_hpc, planets)

##############################################################################
# Try to use stars to fit WCS

# data = artemis_map_roll.data  # SunPy map data
#
# bkg_sigma = mad_std(data)
#
# daofind = DAOStarFinder(fwhm = 2.5, threshold = 5.0 * bkg_sigma)
# sources = daofind(data)
#
# stars_x = sources['xcentroid']
# stars_y = sources['ycentroid']
#
# star_coords = artemis_map_roll.pixel_to_world(stars_x * u.pix, stars_y * u.pix)
#
# fig, ax = plot_artemis_map(artemis_map_roll, moon_hpc, planets)
# ax.plot_coord(star_coords, 's', markerfacecolor='none')
# ax.scatter(stars_x, stars_y, marker='+', color='r')

##############################################################################
# Try to use stars to fit WCS
# Filter for bright stars (e.g., V magnitude < 8)

# with SphericalScreen(coords["artemis_ii"]):
#     center_icrs = artemis_map_roll.center.spherical.transform_to('icrs')

# center_icrs = SkyCoord(ra=planets["saturn"].icrs.ra, dec=planets["saturn"].icrs.dec, frame="icrs")
# v = Vizier(columns=["*"], column_filters={"VTmag": "<11"}, row_limit=-1)
# results = v.query_region(center_icrs, radius=30*u.deg, catalog="I/259/tyc2")
# catalog = results[0]
# catalog.sort("VTmag")
#
# catalog_coords = SkyCoord(
#     ra=catalog["RA(ICRS)"],
#     dec=catalog["DE(ICRS)"],
#     distance=100*u.pc,
#     frame="icrs"
# )
#
# with SphericalScreen(coords["artemis_ii"]):
#     catalog_coords_hpc = catalog_coords.transform_to(artemis_map_roll.coordinate_frame)
#
# fig, ax = plot_artemis_map(artemis_map_roll, moon_hpc, planets)
# ax.plot_coord(star_coords, 's', markerfacecolor='none')
# ax.plot_coord(catalog_coords_hpc[:300], 'm+')
#
# idx, sep, _ = star_coords.match_to_catalog_sky(catalog_coords_hpc)
#
# # Keep only close matches
# sep_threshold = 200 * u.arcsec # adjust based on your current WCS residual
# good = sep < sep_threshold
#
# fig, ax = plot_artemis_map(artemis_map_roll, moon_hpc, planets)
# ax.plot_coord(star_coords[good], 's', markerfacecolor='none')
# ax.plot_coord(catalog_coords_hpc[idx[good]], 'm+')
#
# pixel_coords = np.array([stars_x[good], stars_y[good]])
# sky_coords_matched = catalog_coords_hpc[idx[good]]
#
# wcs_fitted = fit_wcs_from_points(
#      pixel_coords,
#      sky_coords_matched,
#      projection=artemis_map_roll.wcs,
# )
#
# m_stars = Map(artemis_image, wcs_fitted)
#
# fig, ax = plot_artemis_map(m_stars, moon_hpc, planets)
# ax.scatter(stars_x[good], stars_y[good], marker='x', color='r')
# with SphericalScreen(coords["artemis_ii"]):
#     # ax.plot_coord(star_coords[good], 's', markerfacecolor='none')
#     ax.plot_coord(catalog_coords[idx[good]], 's', markerfacecolor='none')
