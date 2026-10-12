# DoggyBone3D - limb side (laterality) from the placed landmarks.
"""Which hind limb do the landmarks of one list belong to?

Left and right limbs are mirror images. Any positioning in the scanner (even a wrongly entered
patient position in the DICOM header) only ROTATES the limb, a mirror is the only thing that
turns a right limb into a left one. The side therefore follows from the handedness of the
anatomical frame built from the landmarks (RAS is right-handed):

    lateral . (cranial x proximal)  > 0  ->  right limb,   < 0  ->  left limb

Femur   proximal = shaft axis (50 % -> 33 % midpoint)
        lateral  = medial -> lateral condyle
        cranial  = opposite of the caudal offset of the condyles from the shaft axis
        check    : the femoral head must lie MEDIAL of the shaft axis
        (the head is NOT used for cranial/caudal: its cranio-caudal position depends on the
        antetorsion, which is reduced or reversed in many patella-luxation patients)
Tibia   proximal = distal tibia midpoint -> intercondylar eminence (or proximal midpoint)
        lateral  = condylus medialis -> lateralis tibiae
        cranial  = caudal -> cranial medial tibial plateau
        check    : the distal medial -> lateral direction (cochlea / talus) must not point
                   against the proximal one (torsion up to 90 degrees is accepted)

Pure numpy - no Slicer imports, so it also runs in the unit tests.
"""
import numpy as np

MIN_CAUDAL_OFFSET_MM = 8.0      # femur: condyles must lie clearly caudal of the shaft axis
MIN_PLATEAU_MM = 4.0            # tibia: cranio-caudal extent of the two plateau landmarks
MIN_ML_MM = 5.0                 # medial <-> lateral landmark distance across the axis
MAX_DISTAL_ML_ANGLE = 90.0      # tibia: proximal vs. distal medio-lateral direction

FEMUR_REQUIRED = ("Femur midpoint 33%", "Femur midpoint 50%", "medial femur condyle", "lateral femur condyle")
TIBIA_PAIRS_PROXIMAL = (("condylus medialis tibiae", "condylus lateralis tibiae"),
                        ("medial condyle articulation point tibia", "lateral condyle articulation point tibia"))
TIBIA_PAIRS_DISTAL = (("medial cochlea", "lateral cochlea"),
                      ("medial cochlea articulation point tibia", "lateral cochlea articulation point tibia"),
                      ("medial talus", "lateral talus"))


def _get(points, name):
    value = points.get(name)
    return None if value is None else np.asarray(value, float)


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def _perp(v, axis):
    return v - np.dot(v, axis) * axis


def _pair(points, pairs):
    for medial, lateral in pairs:
        m, l = _get(points, medial), _get(points, lateral)
        if m is not None and l is not None:
            return m, l
    return None


def _side_of(lateral, cranial, proximal):
    return "Right" if float(np.dot(lateral, np.cross(cranial, proximal))) > 0 else "Left"


def _result(bone):
    return {"bone": bone, "side": None, "reliable": False, "warnings": [], "values": {}}


def femur_side(points):
    """None when the femur landmarks are not placed yet, else a result dict."""
    p30, p50, med, lat = (_get(points, n) for n in FEMUR_REQUIRED)
    if any(v is None for v in (p30, p50, med, lat)):
        return None
    res = _result("Femur")
    P = _unit(p30 - p50)
    ml = None if P is None else _perp(lat - med, P)
    if P is None or np.linalg.norm(ml) < MIN_ML_MM:
        res["warnings"].append("femur: shaft midpoints or condyles (almost) coincide")
        return res
    L = ml / np.linalg.norm(ml)
    offset = _perp((med + lat) / 2.0 - p50, P)
    offset = offset - np.dot(offset, L) * L                   # cranio-caudal part only
    caudal = float(np.linalg.norm(offset))
    res["values"]["condyles caudal of shaft axis (mm)"] = round(caudal, 1)
    if caudal < MIN_CAUDAL_OFFSET_MM:
        res["warnings"].append(f"femur: condyles only {caudal:.1f} mm caudal of the shaft axis - side uncertain")
        return res
    res["side"] = _side_of(L, -offset / caudal, P)
    res["reliable"] = True
    head = _get(points, "femur head")
    if head is not None:
        medial_mm = -float(np.dot(_perp(head - p50, P), L))
        res["values"]["femur head medial of shaft axis (mm)"] = round(medial_mm, 1)
        if medial_mm < 0:
            res["reliable"] = False
            res["warnings"].append("femur: the head lies LATERAL of the shaft axis - medial / lateral "
                                   "condyle swapped?")
    return res


def tibia_side(points):
    """None when the tibia landmarks are not placed yet, else a result dict."""
    distal = _get(points, "distal tibia midpoint")
    proximal = _get(points, "intercondylar eminence")
    if proximal is None:
        proximal = _get(points, "proximal tibia midpoint")
    pair = _pair(points, TIBIA_PAIRS_PROXIMAL)
    cranial, caudal = _get(points, "cranial medial tibial plateau"), _get(points, "caudal medial tibial plateau")
    if distal is None or proximal is None or pair is None or cranial is None or caudal is None:
        return None
    res = _result("Tibia")
    P = _unit(proximal - distal)
    ml = None if P is None else _perp(pair[1] - pair[0], P)
    if P is None or np.linalg.norm(ml) < MIN_ML_MM:
        res["warnings"].append("tibia: axis points or condyles (almost) coincide")
        return res
    L = ml / np.linalg.norm(ml)
    c = _perp(cranial - caudal, P)
    c = c - np.dot(c, L) * L
    length = float(np.linalg.norm(c))
    res["values"]["plateau cranio-caudal (mm)"] = round(length, 1)
    if length < MIN_PLATEAU_MM:
        res["warnings"].append(f"tibia: plateau points only {length:.1f} mm apart cranio-caudally - side uncertain")
        return res
    res["side"] = _side_of(L, c / length, P)
    res["reliable"] = True
    pair_d = _pair(points, TIBIA_PAIRS_DISTAL)
    if pair_d is not None:
        Ld = _unit(_perp(pair_d[1] - pair_d[0], P))
        if Ld is not None:
            angle = float(np.degrees(np.arccos(np.clip(np.dot(L, Ld), -1.0, 1.0))))
            res["values"]["proximal vs. distal medio-lateral (deg)"] = round(angle, 1)
            if angle > MAX_DISTAL_ML_ANGLE:
                res["reliable"] = False
                res["warnings"].append("tibia: proximal and distal medial / lateral point in opposite "
                                       "directions - medial / lateral swapped at one end?")
    return res


def detect(points):
    """points: {landmark name: (x, y, z) RAS} of ONE landmark list.

    Returns {"side": "Left" / "Right" / None, "status": ..., "bones": [...], "messages": [...]}
    status: "incomplete" - neither femur nor tibia complete
            "uncertain"  - computed, but the geometry gives no clear answer
            "check"      - a plausibility check failed (landmarks probably swapped)
            "conflict"   - femur and tibia give different sides
            "ok"         - side determined"""
    bones = [b for b in (femur_side(points), tibia_side(points)) if b is not None]
    messages = [w for b in bones for w in b["warnings"]]
    out = {"side": None, "status": "incomplete", "bones": bones, "messages": messages}
    if not bones:
        return out
    sides = {b["side"] for b in bones if b["side"] is not None}
    if len(sides) > 1:
        out["status"] = "conflict"
        out["messages"] = ["femur says " + bones[0]["side"].lower() + ", tibia says "
                           + bones[1]["side"].lower() + " - landmarks of both legs mixed, or swapped"] + messages
    elif any(b["side"] is not None and not b["reliable"] for b in bones):
        out["status"], out["side"] = "check", sides.pop()
    elif sides:
        out["status"], out["side"] = "ok", sides.pop()
    else:
        out["status"] = "uncertain"
    return out


def summary(result):
    """One line per bone with the measured values - for tooltips / logs."""
    lines = []
    for b in result["bones"]:
        values = ", ".join(f"{k} {v}" for k, v in b["values"].items())
        lines.append(f"{b['bone']}: {b['side'] or 'undetermined'}" + (f" ({values})" if values else ""))
    return "\n".join(lines + result["messages"])
