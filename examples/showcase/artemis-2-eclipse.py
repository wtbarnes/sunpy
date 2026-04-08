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
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import simple_norm

from sunpy.coordinates import Helioprojective, get_horizons_coord
from sunpy.map import Map, make_fitswcs_header

#NAIF IDS
NAIF_IDS = {
    "artemis_2": -1024,
    "moon": 301,
    "earth": 399,
    "sun": 10
}

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
ax.imshow(artemis_image, origin="lower", cmap="grey")

##############################################################################
# Extract meta data

with Path(filename).open("rb") as f:
    tags = exifread.process_file(f)

obsdate, obstime= tags['EXIF DateTimeDigitized'].values.split(" ")
obsdate = obsdate.replace(":", "-")
obstime = Time(f"{obsdate}T{obstime}")

hours, _ = [int(part) for part in tags['EXIF OffsetTime'].values.split(":")]
offset = hours*u.hour

# obstime = obstime + offset # seems worse if I use this

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

times = np.linspace(obstime - (24*u.hour), obstime + (24*u.hour), 100)

coords =  {key: get_horizons_coord(str(value), obstime) for key, value in NAIF_IDS.items()}
tracks =  {key: get_horizons_coord(str(value), times) for key, value in NAIF_IDS.items()}

sun_artemis = np.vstack([coords[n].cartesian.xyz for n in ['sun', 'artemis_2']])

fig, ax = plt.subplots(1, 2)
ax[0].plot(sun_artemis[:,0], sun_artemis[:,2], 'k-', label="Sun-Artemis-II", linewidth=0.5)
ax[1].plot(sun_artemis[:,0], sun_artemis[:,2], 'k-', label="Sun- Artemis-II", linewidth=0.5)
for name, coord in coords.items():
    for i in [0, 1]:
        line = ax[i].plot(coord.cartesian.xyz[0], coord.cartesian.xyz[2], 'o', label=name)
        ax[i].plot(tracks[name].cartesian.xyz[0], tracks[name].cartesian.xyz[2], color=line[0].get_color())

ax[0].legend()
ax[1].set_xlim(0.9930, 0.9992)
ax[1].set_ylim(-0.10856, -0.10806)
##############################################################################
# Edge detection and Hough filtering

edges = canny(roi, sigma=5)

hough_radii = np.arange(np.floor(np.mean(edges.shape) / 4), np.ceil(np.mean(edges.shape) / 2), 10)
hough_res = hough_circle(edges, hough_radii)

accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

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
plt.legend()
plt.show()

##############################################################################
# Build up meta data required to make a map

im_cx = (cx[0] + slice_x.start) * u.pix
im_cy = (cy[0] + slice_y.start) * u.pix
im_radius = radii[0] * u.pix

moon = SkyCoord(coords['moon'], observer=coords['artemis_2'])
R_moon = 0.2725076 *  u.R_earth  # IAU mean radius
dist_moon = SkyCoord(coords['artemis_2']).separation_3d(moon)
moon_obs = np.arcsin(R_moon / dist_moon).to("arcsec")
print(moon_obs)

plate_scale = moon_obs / im_radius
print(plate_scale)

##############################################################################
# Chi-by-eye needs to be derived from one or all the planets
solar_rotation_angle = 21.0*u.deg
print(solar_rotation_angle)

##############################################################################
# Make map

frame = Helioprojective(observer=coords['artemis_2'], obstime=obstime)
moon_hpc = coords['moon'].transform_to(frame)

header = make_fitswcs_header(
    artemis_image,
    moon_hpc,
    reference_pixel=u.Quantity([im_cx, im_cy]),
    scale=u.Quantity([plate_scale, plate_scale]),
    rotation_angle=solar_rotation_angle,
)

artemis_map = Map(artemis_image, header)

fig, ax = plt.subplots(1,1, subplot_kw={"projection":artemis_map}, figsize=(10, 5), dpi=150)
artemis_map.plot(axes=ax, norm=simple_norm(artemis_image, 'linear', percent=95))
artemis_map.draw_grid(axes=ax)
artemis_map.draw_limb(axes=ax)

ax.plot_coord(moon_hpc, 'b+', label="Lunar Center")
theta= np.linspace(0, 360, 100)*u.deg
lunar_limb = np.vstack([moon_hpc.Tx + np.sin(theta)*moon_obs, moon_hpc.Ty +np.cos(theta)*moon_obs])
ax.plot_coord(SkyCoord(*lunar_limb, frame=artemis_map.coordinate_frame), label="Lunar Limb")


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

# [ax.plot_coord(coord, 'o', label=name.title()) for name, coord in planets.items()]
# ax.legend()

xlim = ax.get_xlim()
ylim = ax.get_ylim()
for name, coord in planets.items():
    # planet_radius = artemis_map.reference_coordinate.separation(coord.transform_to(artemis_map.coordinate_frame))
    # planet_coords = np.vstack([moon_hpc.Tx + np.sin(theta) * planet_radius, moon_hpc.Ty + np.cos(theta) * planet_radius])
    # ax.plot_coord(SkyCoord(*planet_coords, frame=artemis_map.coordinate_frame), label=name.title())
    ax.plot_coord(coord, 'x', label=name.title())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
