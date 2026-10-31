# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

import math
import numpy as np

from DoggyboneLib.helpers import *

ALDFA_REFERENCE_DEG = 94.0   # canine reference aLDFA used for the varus/valgus label
ALDFA_NORMAL_TOLERANCE_DEG = 0.5   # within reference +/- this the aLDFA is labelled "normal" (as the tibia labels)
# canine tibial mechanical joint angles (frontal plane), Dismukes et al. 2007: mMPTA 93.3 +/- 1.8 deg;
# mMDTA value to be checked against the paper
TIBIA_MPTA_REF_DEG = 93.3
TIBIA_MDTA_REF_DEG = 95.8

def sphere_landmark(point_dict, radius_dict, label):
    '''
    Returns (center, radius) for a manually-placed sphere landmark: the landmark's own placed
    position as the center, and its user-adjusted radius. Used wherever visually matching a
    sphere against the bone surface (femur head, femoral condyles) is more practical than an
    automatic best-fit from several points.
    '''
    return point_dict[label], radius_dict[label]

class BaseMeasurement:
    '''
    Base class for all measurements. Child classes need to implement the _measure method that takes a
    dictionary of landmark positions and returns a float value and a string description. Calls to the
    measurement should be done via ().
    '''
    # Names of any fitted spheres this measurement computes (e.g. "femur head"), for subclasses
    # that fit one. Used by the GUI to know which spheres to show/hide, independent of whether
    # a fit has actually been computed yet.
    SPHERE_LABELS = []

    # Descriptive labels of the two anatomical axes this measurement compares (e.g. "Tibial
    # axis", "Talar axis"), in the same order they're compared. Used by the GUI to know which
    # axis lines to show/hide, independent of whether they've actually been computed yet.
    AXIS_LABELS = []

    # Optional display/placement order of this measurement's landmarks in the GUI. Names not
    # listed keep their natural (first-queried) order after the listed ones. Does not affect the
    # measurement itself.
    LANDMARK_ORDER = []

    # Landmarks the measurement uses when they are placed but does not need (e.g. the distal partial
    # axis of the femur): _measure asks for them with 'name in point_dict'.
    OPTIONAL_LANDMARKS = []

    def __init__(self, name):
        self.name = name                 # internal key (used by the planning / export)
        self.display_name = name         # shown in the GUI
        self.side = None
        self.landmarks = []
        self.description = ""
        self.spheres = {}  # label -> (center, radius), populated by _measure on success
        self.axes = {}  # label -> (start_point, end_point), populated by _measure on success

    def get_spheres(self):
        return self.spheres

    def result_lines(self, value, description):
        """Optional multi-line result for the GUI (None: the default '<value> <description>')."""
        return None

    def get_axes(self):
        return self.axes

    # Optional: {axis label: (landmark name, landmark name)} for axes that are simply the line between
    # two landmarks - these are drawn as soon as both are placed, before the measurement is complete.
    AXIS_ENDPOINTS = {}

    def partial_axes(self, placed_points):
        """{label: (start, end)} of the AXIS_ENDPOINTS whose two landmarks are placed."""
        return {label: (placed_points[a], placed_points[b]) for label, (a, b) in self.AXIS_ENDPOINTS.items()
                if a in placed_points and b in placed_points}

    def set_side(self, side):
        self.side = side

    def register_landmarks(self, landmarks):
        # Hacky way to find all used landmark names
        class FakeDict:
            def __init__(self):
                self.queried_keys = []

            def __getitem__(self, key):
                self.queried_keys.append(key)
                return np.random.rand(3)

            def __contains__(self, key):          # optional landmarks are asked for with 'in'
                return True

        fake_dict = FakeDict()
        self._measure(fake_dict, fake_dict)
        # _measure legitimately reads some landmarks more than once (e.g. once to build a
        # reference "normal" vector, again for the actual comparison vector), so dedupe while
        # preserving first-seen order - otherwise a landmark referenced N times would be
        # registered and listed N times.
        used_landmark_names = list(dict.fromkeys(fake_dict.queried_keys))
        if self.LANDMARK_ORDER:
            ordered = [name for name in self.LANDMARK_ORDER if name in used_landmark_names]
            used_landmark_names = ordered + [name for name in used_landmark_names if name not in ordered]

        # Register landmark objects
        for name in used_landmark_names:
            landmark_found = False
            for landmark in landmarks:
                if landmark.name == name:
                    self.landmarks.append(landmark)
                    landmark_found = True
                    break
            if not landmark_found:
                raise ValueError(f"Could not register landmark with name {name}")

        # Register update callback for all landmarks
        for landmark in self.landmarks:
            landmark.add_change_callback(self.maybe_update)

    def get_landmarks(self):
        return self.landmarks

    def maybe_update(self):
        pass

    def __call__(self):
        # Check if all landmarks were placed
        optional = set(self.OPTIONAL_LANDMARKS)
        for landmark in self.landmarks:
            if not landmark.placed and landmark.name not in optional:
                return False, None, "Not all landmarks defined"
        used = [l for l in self.landmarks if l.placed]
        point_dict = {l.name: l.get_position() for l in used}
        radius_dict = {l.name: getattr(l, "radius", None) for l in used}
        try:
            angle, message = self._measure(point_dict, radius_dict)
        except ValueError as error:          # degenerate landmark geometry (see helpers.angle)
            return False, None, f"Cannot measure: {error}"
        return True, angle, message

class TibiaTorsionMeasurement(BaseMeasurement):
    AXIS_LABELS = ["Distal cochlear axis", "Proximal tibial condylar axis"]

    def __init__(self):
        super().__init__('Tibia Torsion')
    
    def _measure(self, point_dict, radius_dict):
        normal = vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])
        vector_distal = vector_with_two_points(point_dict["medial cochlea"], point_dict["lateral cochlea"])
        vector_proximal = vector_with_two_points(point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"])
        self.axes = {
            "Distal cochlear axis": (point_dict["medial cochlea"], point_dict["lateral cochlea"]),
            "Proximal tibial condylar axis": (point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"]),
        }
        
        vector_distal_proj = project_vector_to_plane_from_normal(normal, vector_distal)
        vector_proximal_proj = project_vector_to_plane_from_normal(normal, vector_proximal)
        
        a = angle(vector_distal_proj, vector_proximal_proj)*180/math.pi
        
        t = np.cross(vector_proximal_proj, vector_distal_proj)
        q = np.dot(t,normal)
        
        if self.side.lower() == "right":
            return a, "Internal torsion" if q > 0 else "External torsion"
        elif self.side.lower() == "left":
            return a, "External torsion" if q > 0 else "Internal torsion"
        else:
            raise ValueError(f"Unknown side {self.side}")

class VarusValgusTibiaMeasurement(BaseMeasurement):
    """Frontal-plane alignment of the tibia, independent of the tibial torsion.

    Each joint line is measured against the tibial axis in ITS OWN frontal plane (the plane that
    contains the axis and that joint line): the angle between a line and an axis does not change
    when the line turns about the axis, so a torsion of the proximal against the distal tibia
    cannot leak into the varus/valgus value.
      mMPTA: proximal joint line (lateral -> medial articulation point) vs. the mechanical axis
             (intercondylar eminence -> talus centre), medial-proximal angle
      mMDTA: distal joint line (lateral -> medial cochlea articulation point) vs. the mechanical
             axis, medial-distal angle
    The value used by the osteotomy planner is the frontal angle between the two joint lines on
    the anatomical axis, aMPTA + aMDTA - 180 deg (0 = parallel joint lines; normal ~ MPTA_REF + MDTA_REF - 180).
    Smaller than normal = varus, larger = valgus. The anatomical angles (shaft axis through the two
    diaphyseal midpoints) are shown as well."""
    AXIS_LABELS = ["Tibial mechanical axis", "Tibial shaft axis", "Proximal joint line", "Distal joint line"]
    LANDMARK_ORDER = ["intercondylar eminence", "Talus centre", "proximal tibia midpoint", "distal tibia midpoint",
                      "lateral condyle articulation point tibia", "medial condyle articulation point tibia",
                      "lateral cochlea articulation point tibia", "medial cochlea articulation point tibia"]

    def __init__(self):
        super().__init__('Varus Valgus Tibia')
        self.display_name = "Varus/Valgus Tibia (mMPTA / mMDTA)"
        self._angles = None

    def result_lines(self, value, description):
        if self._angles is None:
            return None
        m_mpta, m_mdta, a_mpta, a_mdta = self._angles
        deg = "\N{DEGREE SIGN}"
        reference = TIBIA_MPTA_REF_DEG + TIBIA_MDTA_REF_DEG - 180.0
        return [
            f"Joint lines (aMPTA + aMDTA - 180, torsion-free): {value:.2f}{deg} ({description}; normal ~{reference:.1f}{deg})",
            f"mMPTA: {m_mpta:.2f}{deg} ({self._label(m_mpta, TIBIA_MPTA_REF_DEG)} proximal; ref. {TIBIA_MPTA_REF_DEG:.1f}{deg})",
            f"mMDTA: {m_mdta:.2f}{deg} ({self._label(m_mdta, TIBIA_MDTA_REF_DEG)} distal; ref. {TIBIA_MDTA_REF_DEG:.1f}{deg})",
            f"aMPTA / aMDTA (shaft axis): {a_mpta:.2f}{deg} ({self._label(a_mpta, TIBIA_MPTA_REF_DEG)}) / "
            f"{a_mdta:.2f}{deg} ({self._label(a_mdta, TIBIA_MDTA_REF_DEG)})",
        ]

    @staticmethod
    def _label(value, reference, tolerance=0.5):
        if value < reference - tolerance:
            return "varus"
        if value > reference + tolerance:
            return "valgus"
        return "normal"

    def _measure(self, point_dict, radius_dict):
        eminence = np.asarray(point_dict["intercondylar eminence"], dtype=float)
        talus = np.asarray(point_dict["Talus centre"], dtype=float)
        shaft_proximal = np.asarray(point_dict["proximal tibia midpoint"], dtype=float)
        shaft_distal = np.asarray(point_dict["distal tibia midpoint"], dtype=float)
        prox_lat = np.asarray(point_dict["lateral condyle articulation point tibia"], dtype=float)
        prox_med = np.asarray(point_dict["medial condyle articulation point tibia"], dtype=float)
        dist_lat = np.asarray(point_dict["lateral cochlea articulation point tibia"], dtype=float)
        dist_med = np.asarray(point_dict["medial cochlea articulation point tibia"], dtype=float)
        self.axes = {
            "Tibial mechanical axis": (eminence, talus),
            "Tibial shaft axis": (shaft_distal, shaft_proximal),
            "Proximal joint line": (prox_lat, prox_med),
            "Distal joint line": (dist_lat, dist_med),
        }
        proximal_line = prox_med - prox_lat            # lateral -> medial
        distal_line = dist_med - dist_lat
        mech_down = talus - eminence                   # proximal -> distal
        shaft_down = shaft_distal - shaft_proximal
        deg = 180.0 / math.pi
        # angle between a line and the axis: invariant under rotation about the axis (torsion-free)
        m_mpta = angle(proximal_line, mech_down) * deg
        m_mdta = angle(distal_line, -mech_down) * deg
        a_mpta = angle(proximal_line, shaft_down) * deg
        a_mdta = angle(distal_line, -shaft_down) * deg
        self._angles = (m_mpta, m_mdta, a_mpta, a_mdta)
        # planner value: frontal angle between the joint lines on the ANATOMICAL axis
        joint_lines = a_mpta + a_mdta - 180.0
        reference = TIBIA_MPTA_REF_DEG + TIBIA_MDTA_REF_DEG - 180.0
        return joint_lines, "Varus" if joint_lines < reference else "Valgus"


class TibialPlateauAngleMeasurement(BaseMeasurement):
    """Tibial plateau angle (TPA): slope of the medial tibial plateau (caudal -> cranial point of the
    medial condyle's articular surface) against the perpendicular to the functional axis
    (intercondylar eminence -> talus centre), measured in the tibia's own sagittal plane (the plane
    through the functional axis perpendicular to the proximal condylar axis) - independent of how
    the tibia lies in the scanner and of the tibial torsion. Positive = plateau slopes caudally."""
    AXIS_LABELS = ["Functional axis", "Medial plateau line"]
    LANDMARK_ORDER = ["cranial medial tibial plateau", "caudal medial tibial plateau", "intercondylar eminence",
                      "Talus centre", "condylus medialis tibiae", "condylus lateralis tibiae"]

    def __init__(self):
        super().__init__("Tibial Plateau Angle")
        self.display_name = "Tibial Plateau Angle (TPA)"

    def _measure(self, point_dict, radius_dict):
        cranial = np.asarray(point_dict["cranial medial tibial plateau"], dtype=float)
        caudal = np.asarray(point_dict["caudal medial tibial plateau"], dtype=float)
        eminence = np.asarray(point_dict["intercondylar eminence"], dtype=float)
        talus = np.asarray(point_dict["Talus centre"], dtype=float)
        medial = np.asarray(point_dict["condylus medialis tibiae"], dtype=float)
        lateral = np.asarray(point_dict["condylus lateralis tibiae"], dtype=float)
        self.axes = {
            "Functional axis": (eminence, talus),
            "Medial plateau line": (caudal, cranial),
        }
        up = eminence - talus
        up = up / max(np.linalg.norm(up), 1e-9)
        ml = lateral - medial
        ml = ml - np.dot(ml, up) * up                  # sagittal plane normal, perpendicular to the axis
        plateau = cranial - caudal
        if np.linalg.norm(ml) > 1e-9:
            ml = ml / np.linalg.norm(ml)
            plateau = plateau - np.dot(plateau, ml) * ml
        rise = float(np.dot(plateau, up))              # > 0: cranial point more proximal
        run = float(np.linalg.norm(plateau - rise * up))
        tpa = math.degrees(math.atan2(rise, max(run, 1e-9)))
        return abs(tpa), "TPA (plateau slopes caudally)" if tpa >= 0 else "TPA (plateau slopes cranially)"


class TibiotalarRotationMeasurement(BaseMeasurement):
    AXIS_LABELS = ["Distal tibial axis", "Talar axis"]

    def __init__(self):
        super().__init__("Tibiotalar Rotation")
    
    def _measure(self, point_dict, radius_dict):
        normal = vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])
        vector_distaltibia = vector_with_two_points(point_dict["medial cochlea"], point_dict["lateral cochlea"])
        vector_talus = vector_with_two_points(point_dict["medial talus"], point_dict["lateral talus"])
        self.axes = {
            "Distal tibial axis": (point_dict["medial cochlea"], point_dict["lateral cochlea"]),
            "Talar axis": (point_dict["medial talus"], point_dict["lateral talus"]),
        }
        
        vector_distaltibia_proj = project_vector_to_plane_from_normal(normal, vector_distaltibia)
        vector_talus_proj = project_vector_to_plane_from_normal(normal, vector_talus)
        
        
        a = angle(vector_distaltibia_proj, vector_talus_proj)*180/math.pi
        
        t = np.cross(vector_distaltibia_proj, vector_talus_proj)
        q = np.dot(t,normal)
        
        if self.side.lower() == "right":
            return a, "Internal rotation" if q > 0 else "External rotation"
        elif self.side.lower() == "left":
            return a, "External rotation" if q > 0 else "Internal rotation"
        else:
            raise ValueError(f"Unknown side {self.side}")

class FemorotibialRotationMeasurement(BaseMeasurement):
    SPHERE_LABELS = ["medial femur condyle", "lateral femur condyle"]
    AXIS_LABELS = ["Femoral condylar axis", "Proximal tibial axis"]

    def __init__(self):
        super().__init__("Femorotibial Rotation")

    def _measure(self, point_dict, radius_dict):
        normal = vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])

        medial_condyle, medial_radius = sphere_landmark(point_dict, radius_dict, "medial femur condyle")
        lateral_condyle, lateral_radius = sphere_landmark(point_dict, radius_dict, "lateral femur condyle")
        self.spheres = {
            "medial femur condyle": (medial_condyle, medial_radius),
            "lateral femur condyle": (lateral_condyle, lateral_radius),
        }
        self.axes = {
            "Femoral condylar axis": (medial_condyle, lateral_condyle),
            "Proximal tibial axis": (point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"]),
        }
        vector_distal_femur = vector_with_two_points(medial_condyle, lateral_condyle)
        vector_prox_tibia = vector_with_two_points(point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"])
        
        vector_distal_femur_proj = project_vector_to_plane_from_normal(normal, vector_distal_femur)
        vector_prox_tibia_proj = project_vector_to_plane_from_normal(normal, vector_prox_tibia)
        
        a = angle(vector_distal_femur_proj, vector_prox_tibia_proj)*180/math.pi
        
        t = np.cross(vector_distal_femur_proj, vector_prox_tibia_proj)
        q = np.dot(t,normal)
        
        if self.side.lower() == "right":
            return a, "Internal rotation" if q > 0 else "External rotation"
        elif self.side.lower() == "left":
            return a, "External rotation" if q > 0 else "Internal rotation"
        else:
            raise ValueError(f"Unknown side {self.side}")

# ---- femur: distal partial axis and CORA (optional, part of aLDFA / Antetorsion) ---------------------
# One point in the distal metaphysis: the distal partial axis is CONSTRUCTED through it at the aLDFA
# target to the condylar axis (joint-line method after Paley - for a deformity far distally, where the
# distal segment is too short for two shaft points). A second point further proximal in the distal
# segment: the distal partial axis is MEASURED (both points centred in the cross-section). The CORA is
# the intersection of the proximal (shaft 33 % / 50 %) and the distal partial axis in the frontal plane.
FEMUR_DISTAL_AXIS_POINT = "Femur distal axis point"
FEMUR_DISTAL_ANCHOR = "Femur distal metaphysis"
FEMUR_DISTAL_OPTIONAL = [FEMUR_DISTAL_AXIS_POINT, FEMUR_DISTAL_ANCHOR]
FEMUR_CORA = "CORA"
AXIS_FEMUR_DISTAL = "Distal partial axis"
FEMUR_CORA_MARKER_RADIUS = 2.0
FEMUR_CORA_RELIABLE_DEG = 5.0      # below this angulation 1 mm axis offset moves the CORA by > 11 mm
FEMUR_DISTAL_SEGMENT_TOLERANCE_DEG = 2.0

_aldfa_target_provider = None


def set_aldfa_target_provider(provider):
    """provider() -> the planned aLDFA target in degrees (the femur target of the planning)."""
    global _aldfa_target_provider
    _aldfa_target_provider = provider


def aldfa_target():
    if _aldfa_target_provider is not None:
        try:
            value = float(_aldfa_target_provider())
            if 45.0 < value < 135.0:
                return value
        except Exception:
            pass
    return ALDFA_REFERENCE_DEG


def _unit3(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def femur_distal_cora(point_dict, medial, lateral):
    """Distal partial axis + CORA from the optional points, or None if the metaphysis point is not set.
    Returns a dict with the drawn axes / marker and the values for the result lines."""
    if FEMUR_DISTAL_ANCHOR not in point_dict:
        if FEMUR_DISTAL_AXIS_POINT in point_dict:
            point_dict[FEMUR_DISTAL_AXIS_POINT]              # registration: both names are queried
        return None
    anchor = np.asarray(point_dict[FEMUR_DISTAL_ANCHOR], float)
    second = point_dict[FEMUR_DISTAL_AXIS_POINT] if FEMUR_DISTAL_AXIS_POINT in point_dict else None
    target = aldfa_target()
    p33 = np.asarray(point_dict["Femur midpoint 33%"], float)
    p50 = np.asarray(point_dict["Femur midpoint 50%"], float)
    medial, lateral = np.asarray(medial, float), np.asarray(lateral, float)
    e1 = _unit3(lateral - medial)                        # condylar axis, medial -> lateral
    p = _unit3(p33 - p50)                                # proximal partial axis, towards proximal
    e2 = _unit3(p - np.dot(p, e1) * e1)                  # frontal plane, perpendicular to the condyles
    n = np.cross(e1, e2)                                 # frontal plane normal (cranio-caudal)
    deg = lambda a, b: float(np.degrees(np.arccos(np.clip(np.dot(_unit3(a), _unit3(b)), -1.0, 1.0))))
    current = deg(p, e1)                                 # aLDFA of the proximal partial axis
    measured = second is not None
    if measured:
        u = _unit3(np.asarray(second, float) - anchor)
        if np.dot(u, p) < 0:
            u = -u
    else:
        t = np.radians(target)
        u = np.cos(t) * e1 + np.sin(t) * e2
    u_front = _unit3(u - np.dot(u, n) * n)
    distal_aldfa = deg(u_front, e1)                      # aLDFA of the distal partial axis (frontal)
    frontal = current - distal_aldfa                     # angulation between the partial axes, + varus
    u_sag = _unit3(u - np.dot(u, e1) * e1)
    sagittal = float(np.degrees(np.arctan2(np.dot(u_sag, n), np.dot(u_sag, e2))))   # vs the proximal axis
    joint_centre = 0.5 * (medial + lateral)
    info = {"measured": measured, "target": target, "current": current, "distal_aldfa": distal_aldfa,
            "frontal": frontal, "sagittal": sagittal,
            "anchor_level": float(np.dot(anchor - joint_centre, e2))}
    axes, spheres = {}, {}
    if abs(frontal) >= 0.5:
        proj = lambda q: q - np.dot(q - p50, n) * n
        a0 = proj(anchor)
        w = proj(anchor + u) - a0
        s, _r = np.linalg.lstsq(np.stack([p, -w], axis=1), a0 - p50, rcond=None)[0]
        cora = p50 + s * p                               # on the proximal partial axis
        info["cora"] = cora
        info["cora_level"] = float(np.dot(cora - joint_centre, e2))
        axes["Femoral shaft axis"] = (cora, p33)
        axes[AXIS_FEMUR_DISTAL] = (anchor - 10.0 * u, anchor + np.dot(cora - anchor, u) * u)
        spheres[FEMUR_CORA] = (cora, FEMUR_CORA_MARKER_RADIUS)
    else:
        axes[AXIS_FEMUR_DISTAL] = (anchor - 10.0 * u, anchor + 40.0 * u)
    info["axes"], info["spheres"] = axes, spheres
    return info


def femur_cora_lines(info):
    if not info:
        return ["Distal partial axis: set 'Femur distal metaphysis' (and optionally 'Femur distal axis point') "
                "for the CORA"]
    lines = []
    if info["measured"]:
        lines.append(f"Distal partial axis (2 points, measured): aLDFA of the distal segment {info['distal_aldfa']:.1f}°")
    else:
        lines.append(f"Distal partial axis (1 point, at the target {info['target']:.1f}° to the condylar axis)")
    if "cora" not in info:
        lines.append("No CORA: the partial axes are parallel in the frontal plane")
        return lines
    kind = "varus" if info["frontal"] > 0 else "valgus"
    lines.append(f"CORA (red marker): {info['cora_level']:.1f} mm proximal of the condylar axis, "
                 f"{info['cora_level'] - info['anchor_level']:.1f} mm above the metaphysis point - "
                 f"partial axes {abs(info['frontal']):.1f}° {kind} (frontal)")
    correction = info["current"] - info["target"]
    lines.append(f"Correction to the target {info['target']:.1f}°: {correction:+.1f}° at the CORA")
    if info["measured"]:
        rest = info["distal_aldfa"] - info["target"]
        if abs(rest) > FEMUR_DISTAL_SEGMENT_TOLERANCE_DEG:
            lines.append(f"The distal segment itself stands {rest:+.1f}° off the target: after the correction "
                         f"the partial axes stay {abs(rest):.1f}° apart (deformity partly within the distal segment)")
        lines.append(f"Sagittal: distal partial axis {info['sagittal']:+.1f}° to the proximal axis (+ = cranial)")
    if info["cora_level"] < info["anchor_level"]:
        lines.append("CORA lies DISTAL of the metaphysis point - check the points and the target")
    if abs(info["frontal"]) < FEMUR_CORA_RELIABLE_DEG:
        lines.append(f"UNRELIABLE: only {abs(info['frontal']):.1f}° - 1 mm axis offset moves the CORA by "
                     f"{1.0 / np.tan(np.radians(max(abs(info['frontal']), 0.1))):.0f} mm")
    return lines


class VarusValgusFemurMeasurement(BaseMeasurement):
    SPHERE_LABELS = ["medial femur condyle", "lateral femur condyle", FEMUR_CORA]
    AXIS_LABELS = ["Femoral shaft axis", "Femoral condylar axis", AXIS_FEMUR_DISTAL]
    OPTIONAL_LANDMARKS = FEMUR_DISTAL_OPTIONAL
    LANDMARK_ORDER = ["Femur midpoint 33%", "Femur midpoint 50%", "medial femur condyle", "lateral femur condyle",
                      FEMUR_DISTAL_AXIS_POINT, FEMUR_DISTAL_ANCHOR]

    def result_lines(self, value, description):
        return [f"{value:.2f}\N{DEGREE SIGN} {description}"] + femur_cora_lines(getattr(self, "cora_info", None))

    def _add_distal_axis(self, point_dict, medial, lateral):
        self.cora_info = femur_distal_cora(point_dict, medial, lateral)
        if self.cora_info:
            self.axes.update(self.cora_info["axes"])
            self.spheres.update(self.cora_info["spheres"])

    def __init__(self):
        super().__init__("aLDFA (Varus/Valgus Femur)")

    def _measure(self, point_dict, radius_dict):
        medial_condyle, medial_radius = sphere_landmark(point_dict, radius_dict, "medial femur condyle")
        lateral_condyle, lateral_radius = sphere_landmark(point_dict, radius_dict, "lateral femur condyle")
        self.spheres = {
            "medial femur condyle": (medial_condyle, medial_radius),
            "lateral femur condyle": (lateral_condyle, lateral_radius),
        }
        self.axes = {
            "Femoral shaft axis": (point_dict["Femur midpoint 50%"], point_dict["Femur midpoint 33%"]),
            "Femoral condylar axis": (medial_condyle, lateral_condyle),
        }

        # aLDFA is traditionally read on an AP (anteroposterior) radiograph, which is why this
        # used to project both axes onto a coronal plane assumed from a fixed anteroposterior
        # direction in the scanner's frame. That assumption doesn't hold for every scan - a
        # limb tilted well out of that assumed plane produces a materially distorted angle. Since
        # this is a full 3D CT-derived measurement rather than a 2D radiograph, the raw 3D angle
        # between the two axes is used directly instead - it needs no assumption about how the
        # limb was positioned in the scanner.
        # The shaft vector points proximally (50% -> 30%): verified against a deliberately
        # constructed severe-varus test case (moving the proximal landmark medially). Flipping it
        # gives the supplementary angle (180 - angle), which read Valgus for that known-varus
        # case - the earlier "point distally" convention was calibrated against the now-removed
        # plane projection and doesn't carry over to the raw 3D angle.
        vector_axis_femur_vv = vector_with_two_points(point_dict["Femur midpoint 50%"], point_dict["Femur midpoint 33%"])
        vector_dist_femur_vv = vector_with_two_points(medial_condyle, lateral_condyle)

        # aLDFA: the femoral (anatomical) axis stands at 90 deg on the condylar axis. Varus
        # increases this lateral angle above 90 deg, valgus decreases it below 90 deg - this is a
        # magnitude comparison, not a handedness/chirality question, so it needs no side-dependent
        # sign logic (unlike the rotation measurements elsewhere in this module).
        aldfa = angle(vector_axis_femur_vv, vector_dist_femur_vv)*180/math.pi
        self._add_distal_axis(point_dict, medial_condyle, lateral_condyle)
        # Canine reference aLDFA ~94 deg: above = distal femoral varus, below = valgus, at the
        # reference (+/- ALDFA_NORMAL_TOLERANCE_DEG) = normal.
        if abs(aldfa - ALDFA_REFERENCE_DEG) <= ALDFA_NORMAL_TOLERANCE_DEG:
            return aldfa, "aLDFA - normal"
        return aldfa, "aLDFA - Varus" if aldfa > ALDFA_REFERENCE_DEG else "aLDFA - Valgus"

class AntetorsionMeasurement(BaseMeasurement):
    SPHERE_LABELS = ["femur head", "medial femur condyle", "lateral femur condyle", FEMUR_CORA]
    AXIS_LABELS = ["Femoral condylar axis", "Femoral neck axis", "Femoral shaft axis", AXIS_FEMUR_DISTAL]
    OPTIONAL_LANDMARKS = FEMUR_DISTAL_OPTIONAL
    LANDMARK_ORDER = ["femur head", "femur neck", "Femur midpoint 33%", "Femur midpoint 50%",
                      "medial femur condyle", "lateral femur condyle", FEMUR_DISTAL_AXIS_POINT, FEMUR_DISTAL_ANCHOR]

    def __init__(self):
        super().__init__("Antetorsion")
        self.display_name = "Antetorsion + aLDFA"
        self._aldfa = None

    def result_lines(self, value, description):
        lines = [f"Antetorsion: {value:.2f}\N{DEGREE SIGN}" + ("" if description == "Antetorsion" else f" ({description})")]
        if self._aldfa is not None:
            aldfa_value, aldfa_description = self._aldfa
            kind = aldfa_description.split(" - ", 1)[-1] if " - " in aldfa_description else aldfa_description
            lines.append(f"aLDFA: {aldfa_value:.2f}\N{DEGREE SIGN} ({kind})")
            lines += femur_cora_lines(getattr(self, "cora_info", None))
        return lines

    def _measure(self, point_dict, radius_dict):
        center, head_radius = sphere_landmark(point_dict, radius_dict, "femur head")
        medial_condyle, medial_radius = sphere_landmark(point_dict, radius_dict, "medial femur condyle")
        lateral_condyle, lateral_radius = sphere_landmark(point_dict, radius_dict, "lateral femur condyle")
        self.spheres = {
            "femur head": (center, head_radius),
            "medial femur condyle": (medial_condyle, medial_radius),
            "lateral femur condyle": (lateral_condyle, lateral_radius),
        }
        # Femoral (anatomical) shaft axis: drawn only between the two placed midpoints (50 % / 33 %),
        # like in the aLDFA measurement - no extension beyond the landmarks.
        self.axes = {
            "Femoral condylar axis": (medial_condyle, lateral_condyle),
            "Femoral neck axis": (center, point_dict["femur neck"]),
            "Femoral shaft axis": (point_dict["Femur midpoint 50%"], point_dict["Femur midpoint 33%"]),
        }
        # aLDFA from the same landmarks (shaft 33 % / 50 %, condyles) - shown together with the
        # antetorsion; the stand-alone aLDFA measurement is unchanged
        aldfa = VarusValgusFemurMeasurement()
        aldfa.side = self.side
        self._aldfa = aldfa._measure(point_dict, radius_dict)
        self.cora_info = aldfa.cora_info
        if self.cora_info:
            self.axes.update(self.cora_info["axes"])
            self.spheres.update(self.cora_info["spheres"])
        normal = vector_with_two_points(point_dict["Femur midpoint 50%"], point_dict["Femur midpoint 33%"])
        vector_distal_femur = vector_with_two_points(medial_condyle, lateral_condyle)
        vector_femur_neck = vector_with_two_points(center, point_dict["femur neck"])
        
        vector_distal_femur_proj = project_vector_to_plane_from_normal(normal, vector_distal_femur)
        vector_femur_neck_proj = project_vector_to_plane_from_normal(normal, vector_femur_neck)
        
        a = angle(vector_femur_neck_proj, vector_distal_femur_proj)*180/math.pi
        
        t = np.cross(vector_distal_femur_proj, vector_femur_neck_proj)
        q = np.dot(t,normal)
        
        if self.side.lower() == "right":
            return a, "Retrotorsion" if q > 0 else "Antetorsion"
        elif self.side.lower() == "left":
            return a, "Antetorsion" if q > 0 else "Retrotorsion"
        else:
            raise ValueError(f"Unknown side {self.side}")
        

class TibialMetatarsalAngleMeasurement(BaseMeasurement):
    '''
    Angle between the tibial condylar axis (proximal reference) and the talus-metatarsal axis
    (distal reference), both projected onto the plane perpendicular to the tibial long axis. In
    a normal limb the foot sits roughly in that same plane as the condylar axis, so this isolates
    rotational deviation of the foot relative to the knee - robust to how much the foot happens
    to be flexed/extended, since flexion/extension mostly changes the metatarsal vector's
    component along the tibial axis, which the projection removes. Same pattern as Tibia
    Torsion/Tibiotalar Rotation, which are also projected onto this same tibial-axis plane.
    '''
    AXIS_LABELS = ["Tibial long axis", "Tibial condylar axis", "Talus-metatarsal axis"]

    def __init__(self):
        super().__init__("Tibial Metatarsal Angle")       # internal key (planning / export) - unchanged
        self.display_name = "Tibio-metatarsal rotation"

    def _measure(self, point_dict, radius_dict):
        normal = vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])
        vector_proximal = vector_with_two_points(point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"])
        vector_distal = vector_with_two_points(point_dict["Talus centre"], point_dict["Metatarsal centre"])
        self.axes = {
            "Tibial long axis": (point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"]),
            "Tibial condylar axis": (point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"]),
            "Talus-metatarsal axis": (point_dict["Talus centre"], point_dict["Metatarsal centre"]),
        }

        vector_proximal_proj = project_vector_to_plane_from_normal(normal, vector_proximal)
        vector_distal_proj = project_vector_to_plane_from_normal(normal, vector_distal)
        tma = angle(vector_proximal_proj, vector_distal_proj) * 180 / math.pi
        # Below 90 deg: the metatarsal axis has rotated toward the condylar axis's own lateral
        # direction (external). Above 90 deg: rotated the other way (internal). Verified via
        # controlled rotation-based synthetic geometry - needs no side-dependent sign logic,
        # since "medial -> lateral" is already a correctly side-oriented vector by construction,
        # same reasoning as the aLDFA magnitude-only comparison.
        is_outward = tma < 90
        deviation = abs(tma - 90)
        return deviation, "External rotation" if is_outward else "Internal rotation"
