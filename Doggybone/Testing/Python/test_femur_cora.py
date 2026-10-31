# DoggyBone3D - distal planes and CORAs of the femur, part of aLDFA / Antetorsion (no GUI, no VTK):
#   Slicer:  exec(open(r"<path>/test_femur_cora.py").read())      or
#   shell:   python3 -m unittest <path>/test_femur_cora.py
import os, sys, unittest
import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from DoggyboneLib import measurement_logic as ml

E1, Z = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
N = np.cross(E1, Z)                       # frontal plane normal used by the measurement (e1 x e2)
ANCHOR = ml.FEMUR_DISTAL_ANCHOR


def direction(deg):
    return np.cos(np.radians(deg)) * E1 + np.sin(np.radians(deg)) * Z


def femur(aldfa, cora_z=30.0, axis_offset=0.0, anchor=None):
    """Condyles on the x axis (joint centre at the origin); femoral axis at `aldfa` through (0, 0, cora_z),
    shifted by `axis_offset` along N; metaphysis point by default 18 mm below the CORA at 94 deg."""
    cora = np.array([0.0, 0.0, cora_z])
    p = direction(aldfa)
    shift = axis_offset * N
    points = {"medial femur condyle": np.array([-15.0, 0, 0]), "lateral femur condyle": np.array([15.0, 0, 0]),
              "Femur midpoint 33%": cora + 120 * p + shift, "Femur midpoint 50%": cora + 70 * p + shift,
              ANCHOR: cora - 18 * direction(94.0) if anchor is None else np.asarray(anchor, float),
              "femur head": cora + 160 * p + np.array([-10.0, 5, 0]), "femur neck": cora + 150 * p}
    radii = {"medial femur condyle": 9.0, "lateral femur condyle": 9.0, "femur head": 10.0}
    return points, radii, cora


def distance_to_plane(corners, q):
    c = np.asarray(corners, float)
    normal = np.cross(c[1] - c[0], c[2] - c[1])
    return abs(float(np.dot(np.asarray(q, float) - c[0], normal / np.linalg.norm(normal))))


class Landmark:
    def __init__(self, name, position=None):
        self.name, self.placed, self._p = name, position is not None, position

    def get_position(self):
        return self._p

    def add_change_callback(self, cb):
        pass


class FemurCoraTest(unittest.TestCase):
    def tearDown(self):
        ml.set_aldfa_target_provider(None)

    def aldfa(self, points, radii, side="left"):
        m = ml.VarusValgusFemurMeasurement()
        m.side = side                     # condyles medial -x -> lateral +x: N is cranial for "left"
        value, _ = m._measure(points, radii)
        return m, value

    def test_frontal_cora_on_the_blue_plane(self):
        points, radii, cora = femur(102.0)
        m, value = self.aldfa(points, radii)
        self.assertAlmostEqual(value, 102.0, places=6)            # the aLDFA itself is unchanged
        info = m.cora_info
        np.testing.assert_allclose(info["cora"], cora, atol=1e-6)
        self.assertAlmostEqual(info["cora_level"], 30.0, places=6)
        self.assertAlmostEqual(info["frontal"], 8.0, places=6)
        blue = m.get_planes()[ml.PLANE_FEMUR_TARGET][0]
        self.assertLess(distance_to_plane(blue, points[ANCHOR]), 1e-6)
        self.assertLess(distance_to_plane(blue, cora), 1e-6)

    def test_cranio_caudal_offset_of_the_point_does_not_move_the_frontal_cora(self):
        points, radii, cora = femur(104.0)
        points[ANCHOR] = points[ANCHOR] + 3.0 * N
        m, _ = self.aldfa(points, radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)

    def test_orange_plane_and_sagittal_cora(self):
        # femoral axis 2 mm towards N, metaphysis point 4 mm towards N at 20 mm height: the orange plane
        # leans atan(4/20); the femoral axis pierces it at 2 / (4/20) = 10 mm above the condylar axis
        points, radii, _cora = femur(94.0, axis_offset=2.0, anchor=(0.0, 0.0, 20.0) + 4.0 * N)
        m, _ = self.aldfa(points, radii)
        info = m.cora_info
        self.assertNotIn("cora", info)                             # frontal: at the target already
        self.assertAlmostEqual(info["sagittal"], np.degrees(np.arctan(4.0 / 20.0)), places=6)
        self.assertAlmostEqual(info["sagittal_cora_level"], 10.0, places=6)
        orange = m.get_planes()[ml.PLANE_FEMUR_SAGITTAL][0]
        for q in (points["medial femur condyle"], points["lateral femur condyle"], points[ANCHOR], info["sagittal_cora"]):
            self.assertLess(distance_to_plane(orange, q), 1e-6)
        self.assertTrue(any("procurvatum" in line for line in ml.femur_cora_lines(info)))
        m, _ = self.aldfa(points, radii, side="right")
        self.assertTrue(any("recurvatum" in line for line in ml.femur_cora_lines(m.cora_info)))

    def test_coras_lie_on_the_femoral_axis(self):
        points, radii, _cora = femur(100.0, axis_offset=2.0, anchor=(-2.0, 0.0, 18.0) + 3.0 * N)
        m, _ = self.aldfa(points, radii)
        p50, p33 = points["Femur midpoint 50%"], points["Femur midpoint 33%"]
        axis = (p33 - p50) / np.linalg.norm(p33 - p50)
        for key in ("cora", "sagittal_cora"):
            q = m.cora_info[key] - p50
            self.assertLess(np.linalg.norm(q - np.dot(q, axis) * axis), 1e-6)
        self.assertIn(ml.FEMUR_CORA, m.spheres)
        self.assertIn(ml.FEMUR_CORA_SAGITTAL, m.spheres)

    def test_planned_target(self):
        ml.set_aldfa_target_provider(lambda: 96.0)
        points, radii, cora = femur(90.0, cora_z=25.0)
        points[ANCHOR] = cora - 18 * direction(96.0)
        m, _ = self.aldfa(points, radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)
        self.assertAlmostEqual(m.cora_info["frontal"], -6.0, places=6)

    def test_without_the_point_nothing_changes(self):
        points, radii, _ = femur(102.0)
        del points[ANCHOR]
        m, value = self.aldfa(points, radii)
        self.assertAlmostEqual(value, 102.0, places=6)
        self.assertIsNone(m.cora_info)
        self.assertEqual(m.get_planes(), {})

    def test_antetorsion_carries_the_planes(self):
        points, radii, cora = femur(102.0)
        m = ml.AntetorsionMeasurement()
        m.side = "left"
        m._measure(points, radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)
        self.assertEqual(set(m.get_planes()), set(ml.PLANES_FEMUR_DISTAL))
        self.assertTrue(any("Orange plane" in line for line in m.result_lines(20.0, "Antetorsion")))

    def test_registration_and_optional_landmark(self):
        points, radii, _ = femur(102.0)
        for cls in (ml.VarusValgusFemurMeasurement, ml.AntetorsionMeasurement):
            m = cls()
            m.side = "left"
            m.register_landmarks([Landmark(n, None if n == ANCHOR else points[n]) for n in points])
            self.assertIn(ANCHOR, [l.name for l in m.landmarks])
            ready, _value, _text = m()                       # the optional point unplaced: still measured
            self.assertTrue(ready)

    def test_sagittal_value_for_the_planning_and_export(self):
        points, radii, _cora = femur(94.0, axis_offset=2.0, anchor=(0.0, 0.0, 20.0) + 4.0 * N)
        m, _ = self.aldfa(points, radii)
        self.assertAlmostEqual(m.values["sagittal"], np.degrees(np.arctan(4.0 / 20.0)), places=6)
        names = [row[0] for row in m.export_rows()]
        self.assertIn("Femur sagittal angle (condylar axis - metaphysis plane to femoral axis)", names)
        del points[ANCHOR]
        m, _ = self.aldfa(points, radii)
        self.assertEqual(m.values, {})
        self.assertEqual(m.export_rows(), [])

    def test_femur_config_uses_the_sagittal_value_optionally(self):
        from DoggyboneLib.bones import FEMUR
        self.assertEqual(FEMUR.sagittal_target_measurement, "aLDFA (Varus/Valgus Femur)")
        self.assertEqual(FEMUR.sagittal_target_key, "sagittal")
        self.assertTrue(FEMUR.sagittal_target_optional)
        self.assertTrue(FEMUR.sagittal_input)

    def test_axis_ends_at_the_knee_and_far_coras_are_not_drawn(self):
        # nearly at the target, the point 3 mm beside the axis: the axes meet far beyond the knee
        points, radii, _cora = femur(95.5, anchor=(3.0, 0.0, 12.0))
        m, _ = self.aldfa(points, radii)
        info = m.cora_info
        self.assertTrue(info.get("cora_outside"))
        self.assertNotIn(ml.FEMUR_CORA, m.spheres)
        low, _top = m.axes["Femoral shaft axis"]
        self.assertAlmostEqual(float(low[2]), 0.0, delta=0.5)          # knee: condylar level (z = 0)
        self.assertTrue(any("No frontal CORA on the femur" in line for line in ml.femur_cora_lines(info)))

    def test_bad_target_falls_back_to_reference(self):
        ml.set_aldfa_target_provider(lambda: 0.0)
        self.assertEqual(ml.aldfa_target(), ml.ALDFA_REFERENCE_DEG)


if __name__ == "__main__" or "slicer" in sys.modules:
    unittest.main(argv=["femur_cora"], exit=False, verbosity=1)
