# DoggyBone3D - side (laterality) detection tests (no GUI, synthetic landmarks):
#   Slicer:  exec(open(r"<path>/test_laterality.py").read())      or
#   shell:   PythonSlicer -m unittest <path>/test_laterality.py
import os, types, unittest
import numpy as np

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DoggyboneLib")


def _load(name):
    module = types.ModuleType("_dg_" + name)
    path = os.path.normpath(os.path.join(_LIB, name + ".py"))
    module.__file__ = path
    exec(compile(open(path).read(), path, "exec"), module.__dict__)
    return module


lat = _load("laterality")

# RIGHT hind limb in its anatomical frame: +x lateral, +y cranial, +z proximal (mm)
RIGHT = {
    "Femur midpoint 30%": (0, 4, 140), "Femur midpoint 50%": (0, 2, 100), "femur head": (-18, 6, 205),
    "femur neck": (-6, 2, 190), "medial femur condyle": (-14, -28, 12), "lateral femur condyle": (14, -28, 12),
    "distal tibia midpoint": (0, 0, -200), "proximal tibia midpoint": (0, 0, -50),
    "intercondylar eminence": (0, 0, 0), "condylus medialis tibiae": (-15, -12, -5),
    "condylus lateralis tibiae": (15, -12, -5), "cranial medial tibial plateau": (-12, 10, -2),
    "caudal medial tibial plateau": (-12, -14, -4), "medial cochlea": (-8, 0, -205),
    "lateral cochlea": (8, 0, -205)}
FEMUR = {k: v for k, v in RIGHT.items() if k.startswith(("Femur", "femur")) or "femur condyle" in k}
TIBIA = {k: v for k, v in RIGHT.items() if k not in FEMUR}


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def _pose(points, R, t, mirror=False):
    m = np.array([-1.0, 1.0, 1.0]) if mirror else np.ones(3)
    return {k: R @ (np.asarray(v, float) * m) + t for k, v in points.items()}


class LateralityTest(unittest.TestCase):
    def test_any_position_and_mirror(self):
        rng = np.random.default_rng(7)
        for subset in (RIGHT, FEMUR, TIBIA):
            for _ in range(50):
                R, t = _rotation(rng), rng.normal(size=3) * 200
                right, left = lat.detect(_pose(subset, R, t)), lat.detect(_pose(subset, R, t, mirror=True))
                self.assertEqual((right["status"], right["side"]), ("ok", "Right"))
                self.assertEqual((left["status"], left["side"]), ("ok", "Left"))

    def test_incomplete(self):
        self.assertEqual(lat.detect({})["status"], "incomplete")
        partial = dict(FEMUR)
        del partial["lateral femur condyle"]
        self.assertEqual(lat.detect(partial)["status"], "incomplete")

    def test_femur_condyles_swapped(self):
        points = dict(FEMUR)
        points["medial femur condyle"], points["lateral femur condyle"] = FEMUR["lateral femur condyle"], FEMUR["medial femur condyle"]
        result = lat.detect(points)
        self.assertEqual(result["status"], "check")
        self.assertTrue(any("LATERAL" in m for m in result["messages"]))

    def test_tibia_one_end_swapped(self):
        points = dict(TIBIA)
        points["medial cochlea"], points["lateral cochlea"] = TIBIA["lateral cochlea"], TIBIA["medial cochlea"]
        self.assertEqual(lat.detect(points)["status"], "check")

    def test_femur_and_tibia_of_different_legs(self):
        mirrored = {k: (-v[0], v[1], v[2]) for k, v in TIBIA.items()}
        self.assertEqual(lat.detect({**FEMUR, **mirrored})["status"], "conflict")

    def test_no_cranial_information(self):
        flat = {k: (v[0], 0.0, v[2]) for k, v in FEMUR.items()}      # condyles ON the shaft axis
        result = lat.detect(flat)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["side"])

    def test_antetorsion_does_not_matter(self):
        for head_y in (-15.0, 0.0, 25.0):                             # retro- ... strong antetorsion
            points = dict(FEMUR)
            points["femur head"] = (-18, head_y, 205)
            self.assertEqual(lat.detect(points)["side"], "Right")


if __name__ == "__main__":
    unittest.main()
