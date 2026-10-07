# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""Shared imports and style constants for Doggybone.py and the DoggyboneUI mixins.

The DoggyboneLib modules are (re)loaded by Doggybone.py before this file is imported, so the
plain imports below always get the current version.
"""

import json
import logging
import re
import os
import os.path as osp
import csv
import importlib
import locale
from copy import deepcopy

import numpy as np
import qt, ctk, vtk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import errorDisplay

from DoggyboneLib import measurement_logic, landmark_logic, osteotomy_logic, guide_logic
from DoggyboneLib import measurements, landmarks
from DoggyboneLib.measurements import MEASUREMENTS
from DoggyboneLib.landmarks import LANDMARKS

MODULE_PATH = osp.dirname(osp.dirname(osp.abspath(__file__)))   # the Doggybone module folder

# Status colours used everywhere in the module (light pastel, dark text - readable in light and dark themes)
BONE_OPACITY_TIP = ("3D opacity of the bone - the same value in every section (segmentation, measurements, "
                    "osteotomy preview, guides). Lower it to see landmarks, spheres and axes inside the bone.")
STATUS_OK_STYLE = "QLabel { background-color: #C6EFCE; color: #1B3D1F; padding: 2px 4px; }"
# measured values: a light tint of their section's header colour
MEASUREMENT_VALUE_STYLE = "QLabel { background-color: #E6DEF8; color: #322659; padding: 2px 4px; }"   # Measurements (#6b46c1)
OSTEOTOMY_VALUE_STYLE = "QLabel { background-color: #F8DDCC; color: #5A2A0C; padding: 2px 4px; }"     # Osteotomy (#c05621)
STATUS_WARN_STYLE = "QLabel { background-color: #FFE5B4; color: #4A3000; padding: 2px 4px; }"
STATUS_ERROR_STYLE = "QLabel { background-color: #F8CBCB; color: #5A1010; padding: 2px 4px; font-weight: bold; }"
