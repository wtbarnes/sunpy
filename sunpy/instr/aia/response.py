__author__ = 'TDWilkinson'

"""
Contains functions useful for analysing AIA data.

Goal: Use SunPy to infer plasma properties like temperature and density in multiwavelength images taken by the AIA. Tow routines are necessary to calculate the response functions (while utilyzing ChiantiPy):

Wavelength response functions: calculate the amount of flux per wavelength

Temperature response functions: calculate the sensitivity of light from the plasma per temperature

other important variables:
area
emissivity
"""

# tools to import as necessary:
import os.path
import datetime
import csv
import copy
import socket
from itertools import dropwhile

import numpy as np
from scipy import interpolate
from scipy.integrate import trapz, cumtrapz
import astropy.units as u
import pandas as pd

import chiantipy as ch

from sunpy.net import hek
from sunpy.time import parse_time
from sunpy import config
from sunpy import lightcurve
from sunpy.util.net import check_download_file
from sunpy import sun

# general format
def get_function(variable1, optional = None):
    """
    Statement of usefulness of function.
    Parameters
    ----------
    variable1 : what module of import is used.
        explanation
    optional: (optional) string
        A string specifying optional vairable
        e.g. strings
    """
