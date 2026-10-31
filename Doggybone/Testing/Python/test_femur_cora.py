# DoggyBone3D - distal partial axes and CORA of the femur, part of aLDFA / Antetorsion (no GUI, no VTK):
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


def direction(deg):
    return np.cos(np.radians(deg)) * E1 + np.sin(np.radians(deg)) * Z


def femur(aldfa, cora_z=30.0, distal=94.0, sagittal_deg=0.0, sagittal_offset=0.0):
    """Synthetic distal deformity: condyles on the x axis; the proximal segment at `aldfa`, the distal
    segment at `distal` (frontal) tilted by `sagittal_deg` towards N; both meet at (0, 0, cora_z)."""
    cora = np.array([0.0, 0.0, cora_z])
    p = direction(aldfa)
    s = np.radians(sagittal_deg)
    v = np.cos(s) * direction(distal) + np.sin(s) * N
    shift = sagittal_offset * N
    points = {"medial femur condyle": np.array([-15.0, 0, 0]), "lateral femur condyle": np.array([15.0, 0, 0]),
              "Femur midpoint 33%": cora + 120 * p, "Femur midpoint 50%": cora + 70 * p,
              ml.FEMUR_DISTAL_ANCHOR: cora - 18 * v + shift, ml.FEMUR_DISTAL_AXIS_POINT: cora - 6 * v + shift,
              "femur head": cora + 160 * p + np.array([-10.0, 5, 0]), "femur neck": cora + 150 * p}
    radii = {"medial femur condyle": 9.0, "lateral femur condyle": 9.0, "femur head": 10.0}
    return points, radii, cora, v


def without(points, *names):
    return {k: v for k, v in points.items() if k not in names}


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

    def aldfa(self, points, radii):
        m = ml.VarusValgusFemurMeasurement()
        m.side = "left"                     # condyles medial -x -> lateral +x: N is cranial for "left"
        value, _ = m._measure(points, radii)
        return m, value

    def test_one_point_constructs_the_frontal_axis(self):
        points, radii, cora, _v = femur(102.0)
        for alone in (ml.FEMUR_DISTAL_ANCHOR, ml.FEMUR_DISTAL_AXIS_POINT):
            other = ml.FEMUR_DISTAL_AXIS_POINT if alone == ml.FEMUR_DISTAL_ANCHOR else ml.FEMUR_DISTAL_ANCHOR
            m, value = self.aldfa(without(points, other), radii)
            self.assertAlmostEqual(value, 102.0, places=6)            # the aLDFA itself is unchanged
            info = m.cora_info
            self.assertFalse(info["two_points"])
            self.assertNotIn("sagittal", info)
            np.testing.assert_allclose(info["cora"], cora, atol=1e-6)
            self.assertAlmostEqual(info["cora_level"], 30.0, places=6)
            self.assertAlmostEqual(info["frontal"], 8.0, places=6)
            self.assertIn(ml.FEMUR_CORA, m.spheres)
            self.assertEqual(set(m.get_planes()), {ml.PLANE_FEMUR_TARGET})

    def test_two_points_frontal_at_target_sagittal_measured(self):
        points, radii, cora, v = femur(104.0, cora_z=35.0, sagittal_deg=8.0)
        m, _value = self.aldfa(points, radii)
        info = m.cora_info
        self.assertTrue(info["two_points"])
        np.testing.assert_allclose(info["cora"], cora, atol=1e-6)               # frontal, through the proximal point
        expected = np.degrees(np.arctan2(np.dot(v, N), np.dot(v, Z)))
        self.assertAlmostEqual(info["sagittal"], expected, places=6)
        self.assertGreater(info["sagittal"], 7.0)
        np.testing.assert_allclose(info["sagittal_cora"], cora, atol=1e-6)
        self.assertIn(ml.FEMUR_CORA_SAGITTAL, m.spheres)
        planes = m.get_planes()
        self.assertEqual(set(planes), set(ml.PLANES_FEMUR_DISTAL))
        # the target plane contains the frontal CORA, the sagittal plane both points and the sagittal CORA
        def on_plane(label, q):
            c = np.asarray(planes[label][0])
            normal = np.cross(c[1] - c[0], c[2] - c[1])
            return abs(np.dot(q - c[0], normal / np.linalg.norm(normal)))
        self.assertLess(on_plane(ml.PLANE_FEMUR_TARGET, info["cora"]), 1e-6)
        self.assertLess(on_plane(ml.PLANE_FEMUR_TARGET, points[ml.FEMUR_DISTAL_AXIS_POINT]), 1e-6)
        for q in (points[ml.FEMUR_DISTAL_ANCHOR], points[ml.FEMUR_DISTAL_AXIS_POINT], info["sagittal_cora"]):
            self.assertLess(on_plane(ml.PLANE_FEMUR_SAGITTAL, q), 1e-6)

    def test_sagittal_sign_follows_the_side(self):
        points, radii, _cora, _v = femur(104.0, sagittal_deg=8.0)
        m = ml.VarusValgusFemurMeasurement()
        m.side = "right"
        m._measure(points, radii)
        self.assertLess(m.cora_info["sagittal"], -7.0)
        self.assertTrue(any("recurvatum" in line for line in ml.femur_cora_lines(m.cora_info)))
        m, _ = self.aldfa(points, radii)
        self.assertTrue(any("procurvatum" in line for line in ml.femur_cora_lines(m.cora_info)))

    def test_order_of_the_two_points_does_not_matter(self):
        points, radii, cora, _v = femur(104.0, cora_z=35.0, sagittal_deg=8.0)
        m1, _ = self.aldfa(points, radii)
        swapped = dict(points)
        swapped[ml.FEMUR_DISTAL_ANCHOR], swapped[ml.FEMUR_DISTAL_AXIS_POINT] = \
            points[ml.FEMUR_DISTAL_AXIS_POINT], points[ml.FEMUR_DISTAL_ANCHOR]
        m2, _ = self.aldfa(swapped, radii)
        np.testing.assert_allclose(m2.cora_info["cora"], m1.cora_info["cora"], atol=1e-9)
        self.assertAlmostEqual(m2.cora_info["sagittal"], m1.cora_info["sagittal"], places=9)

    def test_frontal_axis_uses_the_target_not_the_measured_frontal_direction(self):
        # the two points lean 97 deg frontally - the frontal axis still stands at the target 94 deg
        points, radii, _cora, _v = femur(104.0, distal=97.0)
        m, _ = self.aldfa(points, radii)
        self.assertAlmostEqual(m.cora_info["frontal"], 10.0, places=6)
        self.assertAlmostEqual(m.cora_info["measured_frontal_aldfa"], 97.0, places=6)

    def test_planned_target(self):
        ml.set_aldfa_target_provider(lambda: 96.0)
        points, radii, cora, _v = femur(90.0, cora_z=25.0, distal=96.0)
        m, _ = self.aldfa(without(points, ml.FEMUR_DISTAL_AXIS_POINT), radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)
        self.assertAlmostEqual(m.cora_info["frontal"], -6.0, places=6)

    def test_sagittal_offset_does_not_move_the_frontal_cora(self):
        points, radii, cora, _v = femur(104.0, sagittal_offset=3.0)
        m, _ = self.aldfa(points, radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)

    def test_without_the_optional_points_nothing_changes(self):
        points, radii, _cora, _v = femur(102.0)
        m, value = self.aldfa(without(points, *ml.FEMUR_DISTAL_OPTIONAL), radii)
        self.assertAlmostEqual(value, 102.0, places=6)
        self.assertIsNone(m.cora_info)
        self.assertNotIn(ml.FEMUR_CORA, m.spheres)

    def test_antetorsion_carries_the_axes(self):
        points, radii, cora, _v = femur(102.0, sagittal_deg=6.0)
        m = ml.AntetorsionMeasurement()
        m.side = "left"
        m._measure(points, radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)
        self.assertIn(ml.FEMUR_CORA, m.spheres)
        self.assertEqual(set(m.get_planes()), set(ml.PLANES_FEMUR_DISTAL))
        lines = m.result_lines(20.0, "Antetorsion")
        self.assertTrue(any("Orange plane" in line for line in lines))

    def test_registration_and_optional_landmarks(self):
        points, radii, _cora, _v = femur(102.0)
        for cls in (ml.VarusValgusFemurMeasurement, ml.AntetorsionMeasurement):
            m = cls()
            m.side = "left"
            landmarks = [Landmark(n, None if n in ml.FEMUR_DISTAL_OPTIONAL else points[n]) for n in points]
            m.register_landmarks(landmarks)
            registered = [l.name for l in m.landmarks]
            self.assertIn(ml.FEMUR_DISTAL_ANCHOR, registered)
            self.assertIn(ml.FEMUR_DISTAL_AXIS_POINT, registered)
            ready, _value, _text = m()                       # optional points unplaced: still measured
            self.assertTrue(ready)

    def test_bad_target_falls_back_to_reference(self):
        ml.set_aldfa_target_provider(lambda: 0.0)
        self.assertEqual(ml.aldfa_target(), ml.ALDFA_REFERENCE_DEG)


if __name__ == "__main__" or "slicer" in sys.modules:
    unittest.main(argv=["femur_cora"], exit=False, verbosity=1)
