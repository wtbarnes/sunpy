"""
=====================
Artemis II trajectory
=====================

This example visualizes the trajectory of the Artemis II spacecraft.

The trajectory is visualized in two different coordinate frames.  The plots
also highlight the segment of the trajectory when Artemis II was in eclipse.
"""
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.constants import R_earth
from astropy.coordinates import solar_system_ephemeris

from sunpy.coordinates import get_horizons_coord, sun
from sunpy.time import parse_time

##############################################################################
# First, define times spanning the Artemis II mission, with higher resolution
# across eclipse transitions (i.e., the four contacts)

start = parse_time("2026-Apr-02 02:00")
times = start + np.concatenate([np.arange(0, 7112, 5) * u.min,
                                np.arange(7112, 7116, 0.1) * u.min,  # entering eclipse
                                np.arange(7120, 7168, 5) * u.min,
                                np.arange(7168, 7172, 0.1) * u.min,  # exiting eclipse
                                np.arange(7175, 12834, 5) * u.min])

##############################################################################
# Use JPL Horizons to retrieve relevant coordinates, and then use the
# coordinate framework to convert to ecliptic coordinates.  There is no
# convenient coordinate frame for the Earth-Moon system, so convert to a
# heliocentric frame for now.

# Use a JPL ephemeris because astropy's built-in ephemeris is not accurate enough
solar_system_ephemeris.set('de440s')

# Get the Artemis II coordinate in Heliocentric Mean Ecliptic
artemis = get_horizons_coord("Artemis II", times).heliocentricmeanecliptic

# Get other relevant coordinates
# Specify NAIF IDs for Earth and Moon due to ambiguity
earth = get_horizons_coord(399, times).heliocentricmeanecliptic
moon = get_horizons_coord(301, times).heliocentricmeanecliptic
em_barycenter = get_horizons_coord("Earth-Moon Barycenter", times).heliocentricmeanecliptic

##############################################################################
# Calculate when Artemis II was in eclipse.

fraction = sun.eclipse_amount(artemis)
eclipsed = np.flatnonzero(fraction > 0)  # partial and total eclipse

##############################################################################
# Shift the coordinates to have the Earth-Moon barycenter as the origin,
# convert to units of Earth radii, and keep only the X and Y components for
# later plotting.

artemis_x, artemis_y, _ = ((artemis.cartesian - em_barycenter.cartesian).xyz / R_earth).decompose()
earth_x, earth_y, _ = ((earth.cartesian - em_barycenter.cartesian).xyz / R_earth).decompose()
moon_x, moon_y, _ = ((moon.cartesian - em_barycenter.cartesian).xyz / R_earth).decompose()

##############################################################################
# Plot the Artemis II trajectory in fixed ecliptic X-Y coordinates. The motion
# of the Earth relative to the Earth-Moon barycenter is not discernible on
# this plot.  The segment of the trajectory when Artemis II was in eclipse is
# highlighted.

fig, ax = plt.subplots()

ax.plot(earth_x, earth_y, ls='dashed', color='b', label='Earth')
ax.plot(earth_x[-1], earth_y[-1], '.', color='b')

ax.plot(moon_x, moon_y, ls='dashed', color='g', label='Moon')
ax.plot(moon_x[-1], moon_y[-1], '.', color='g')

ax.plot(artemis_x, artemis_y, color='k', label='Artemis II')
ax.plot(artemis_x[-1], artemis_y[-1], '.', color='k')
ax.plot(artemis_x[eclipsed], artemis_y[eclipsed], color='m', lw=3, label='eclipse')

ax.set_title('Fixed coordinate frame')
ax.set_xlabel('Ecliptic X (Earth radii)')
ax.set_ylabel('Ecliptic Y (Earth radii)')
ax.set_aspect('equal')
ax.legend(loc='center right')

##############################################################################
# Transform the X and Y components so that the we are in the frame co-rotating
# with the Moon's orbital motion.

angle = np.arctan2(moon_y, moon_x)
c, s = np.cos(-angle), np.sin(-angle)

artemis_xp, artemis_yp = artemis_x * c - artemis_y * s, artemis_x * s + artemis_y * c
earth_xp, earth_yp = earth_x * c - earth_y * s, earth_x * s + earth_y * c
moon_xp, moon_yp = moon_x * c - moon_y * s, moon_x * s + moon_y * c

##############################################################################
# Plot the Artemis II trajectory in coordinates co-rotating with the Moon's
# orbital motion.

fig, ax = plt.subplots()

ax.plot(earth_xp, earth_yp, ls='dashed', color='b', label='Earth')
ax.plot(earth_xp[-1], earth_yp[-1], '.', color='b')

ax.plot(moon_xp, moon_yp, ls='dashed', color='g', label='Moon')
ax.plot(moon_xp[-1], moon_yp[-1], '.', color='g')

ax.plot(artemis_xp, artemis_yp, color='k', label='Artemis II')
ax.plot(artemis_xp[-1], artemis_yp[-1], '.', color='k')
ax.plot(artemis_xp[eclipsed], artemis_yp[eclipsed], color='m', lw=3, label='eclipse')

ax.set_title('Coordinate frame co-rotating with the Moon')
ax.set_xlabel('X (Earth radii)')
ax.set_ylabel('Y (Earth radii)')
ax.set_aspect('equal')
ax.legend(loc='center')

plt.show()

# sphinx_gallery_thumbnail_number = 2
