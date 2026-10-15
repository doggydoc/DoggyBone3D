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

# --- status fields: coloured by what they say (all sections) ------------------------------------
_STATUS_PROBLEM = re.compile(
    r"(warning|outdated|not exported|not written|not built|not applied|not aligned|not confirmed|not checked|"
    r"cannot|could not|can't|failed|error|too close|does not fit|doesn't fit|not fit|missing|rejected|skipped|"
    r"uncertain|conflict|check the|check them|check it|check before|\bfirst[.!]|^no |nothing|deviates|"
    r"\bno (pins?|corrected|cut|guide|correction|bone|fragment|footprint|preview|landmarks?|segment|plane)\b|"
    r"is empty|are empty|not found|(?<!ring )not possible|sleeves touch\)|"
    r"only \d+ of|leaves the bone|end of the fragment|misses|below the saw guide|outside|"
    r"place the|insert the|set the missing|press 'extrude'|extrude .* again)",
    re.IGNORECASE)
_STATUS_OK = re.compile(
    r"(\u2713|exported to|\bsaved\b|\baligned\b|\bcreated\b|\bverified\b|volume body|pin sleeves|"
    r"guide body built|preview active|predicted after correction|corrected position|segmentation done|"
    r"\bready\b|pin rows:|placed|marked|set \(|removed|restored|\bdone\b|finished|filled|added)",
    re.IGNORECASE)


def status_style_for(text):
    """STATUS_WARN_STYLE for a problem, STATUS_OK_STYLE for a success, "" for plain information."""
    text = text or ""
    if _STATUS_PROBLEM.search(text):
        return STATUS_WARN_STYLE
    if _STATUS_OK.search(text):
        return STATUS_OK_STYLE
    return ""


_STATUS_RANK = {"": 0, STATUS_OK_STYLE: 1, STATUS_WARN_STYLE: 2, STATUS_ERROR_STYLE: 3}


def _short_line(line, limit=110, first_clause=False):
    """One status line, shortened: (optionally) only its first clause, at most `limit` characters."""
    line = line.strip()
    if first_clause:
        for sep in ("; ", ". "):
            cut = line.find(sep, 20)
            if cut > 0:
                line = line[:cut]
    if len(line) > limit:
        line = line[:limit].rsplit(" ", 1)[0] + " \u2026"
    return line


def compact_status(text, max_lines=6):
    """The short form of a long status: the headline, every line with a problem and every line with a
    check mark - the rest (plain information) only in the full text (tooltip)."""
    lines = [l for l in (text or "").split("\n") if l.strip()]
    if len(lines) <= 2 and len(text or "") <= 220:
        return text or ""
    keep = [_short_line(lines[0], first_clause=True)]
    for line in lines[1:]:
        if _STATUS_PROBLEM.search(line) or "\u2713" in line:
            keep.append(_short_line(line))
    dropped = len(lines) > len(keep) or any(len(l.strip()) > 110 for l in lines)
    if len(keep) > max_lines:
        keep = keep[:max_lines]
        dropped = True
    if dropped:
        keep.append("(details: hold the mouse over this field)")
    return "\n".join(keep)


class StatusLabel(qt.QLabel):
    """Status field of a section: green when all is well, orange when there is a problem - decided
    from the text itself. A warning / error colour set explicitly just before the text is kept
    (the more severe of the two wins); a colour set after the text replaces it.
    compact_lines: show only a short form (headline, problems, check marks); the full text is in
    the tooltip and in full_text (read that, not .text, to build on the status)."""

    def __init__(self, text="", compact_lines=None):
        qt.QLabel.__init__(self, "")
        self._explicit = None
        self._compact_lines = compact_lines
        self.full_text = ""
        self.setText(text)

    def setStyleSheet(self, style):
        self._explicit = style
        qt.QLabel.setStyleSheet(self, style)

    def setText(self, text):
        text = "" if text is None else str(text)
        self.full_text = text
        auto = status_style_for(text)
        explicit = getattr(self, "_explicit", None)
        style = explicit if explicit is not None and _STATUS_RANK.get(explicit, 0) > _STATUS_RANK.get(auto, 0) else auto
        shown = text
        if getattr(self, "_compact_lines", None):
            shown = compact_status(text, self._compact_lines)
            if shown != text:
                import html
                self.setToolTip("<qt>" + html.escape(text).replace("\n", "<br>") + "</qt>")
            else:
                self.setToolTip("")
        qt.QLabel.setText(self, shown)
        qt.QLabel.setStyleSheet(self, style)
        self._explicit = None
