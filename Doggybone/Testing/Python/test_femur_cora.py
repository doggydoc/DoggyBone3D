# DoggyBone3D - distal partial axis and CORA of the femur, part of aLDFA / Antetorsion (no GUI, no VTK):
#   Slicer:  exec(open(r"<path>/test_femur_cora.py").read())      or
#   shell:   python3 -m unittest <path>/test_femur_cora.py
import os, sys, unittest
import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from DoggyboneLib import measurement_logic as ml


def femur(aldfa, cora_z=30.0, distal=94.0, sagittal_offset=0.0):
    """Synthetic distal deformity: condyles on the x axis, the distal segment stands at `distal`, the
    proximal segment at `aldfa`; both meet at (0, 0, cora_z)."""
    e1, z = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    direction = lambda deg: np.cos(np.radians(deg)) * e1 + np.sin(np.radians(deg)) * z
    cora = np.array([0.0, 0.0, cora_z])
    p, u = direction(aldfa), direction(distal)
    shift = np.array([0, sagittal_offset, 0])
    points = {"medial femur condyle": np.array([-15.0, 0, 0]), "lateral femur condyle": np.array([15.0, 0, 0]),
              "Femur midpoint 33%": cora + 120 * p, "Femur midpoint 50%": cora + 70 * p,
              ml.FEMUR_DISTAL_ANCHOR: cora - 18 * u + shift, ml.FEMUR_DISTAL_AXIS_POINT: cora - 6 * u + shift,
              "femur head": cora + 160 * p + np.array([-10.0, 5, 0]), "femur neck": cora + 150 * p}
    radii = {"medial femur condyle": 9.0, "lateral femur condyle": 9.0, "femur head": 10.0}
    return points, radii, cora


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
        value, _ = m._measure(points, radii)
        return m, value

    def test_one_point_constructs_the_axis_at_the_target(self):
        points, radii, cora = femur(102.0)
        m, value = self.aldfa(without(points, ml.FEMUR_DISTAL_AXIS_POINT), radii)
        self.assertAlmostEqual(value, 102.0, places=6)            # the aLDFA itself is unchanged
        info = m.cora_info
        self.assertFalse(info["measured"])
        np.testing.assert_allclose(info["cora"], cora, atol=1e-6)
        self.assertAlmostEqual(info["cora_level"], 30.0, places=6)
        self.assertIn(ml.FEMUR_CORA, m.spheres)
        self.assertIn(ml.AXIS_FEMUR_DISTAL, m.axes)

    def test_two_points_measure_the_axis(self):
        # the distal segment itself deviates (97 deg): measured axis, CORA where the axes really meet
        points, radii, cora = femur(104.0, cora_z=35.0, distal=97.0)
        m, _value = self.aldfa(points, radii)
        info = m.cora_info
        self.assertTrue(info["measured"])
        self.assertAlmostEqual(info["distal_aldfa"], 97.0, places=6)
        self.assertAlmostEqual(info["frontal"], 7.0, places=6)
        np.testing.assert_allclose(info["cora"], cora, atol=1e-6)
        lines = ml.femur_cora_lines(info)
        self.assertTrue(any("stands +3.0° off the target" in line for line in lines))

    def test_planned_target(self):
        ml.set_aldfa_target_provider(lambda: 96.0)
        points, radii, cora = femur(90.0, cora_z=25.0, distal=96.0)
        m, _ = self.aldfa(without(points, ml.FEMUR_DISTAL_AXIS_POINT), radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)
        self.assertAlmostEqual(m.cora_info["frontal"], -6.0, places=6)

    def test_sagittal_offset_does_not_move_the_frontal_cora(self):
        points, radii, cora = femur(104.0, sagittal_offset=3.0)
        m, _ = self.aldfa(points, radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)

    def test_without_the_optional_points_nothing_changes(self):
        points, radii, _ = femur(102.0)
        m, value = self.aldfa(without(points, *ml.FEMUR_DISTAL_OPTIONAL), radii)
        self.assertAlmostEqual(value, 102.0, places=6)
        self.assertIsNone(m.cora_info)
        self.assertNotIn(ml.FEMUR_CORA, m.spheres)

    def test_antetorsion_carries_the_cora(self):
        points, radii, cora = femur(102.0)
        m = ml.AntetorsionMeasurement()
        m.side = "left"
        m._measure(without(points, ml.FEMUR_DISTAL_AXIS_POINT), radii)
        np.testing.assert_allclose(m.cora_info["cora"], cora, atol=1e-6)
        self.assertIn(ml.FEMUR_CORA, m.spheres)
        self.assertTrue(any("CORA" in line for line in m.result_lines(20.0, "Antetorsion")))

    def test_registration_and_optional_landmarks(self):
        points, radii, _ = femur(102.0)
        for cls in (ml.VarusValgusFemurMeasurement, ml.AntetorsionMeasurement):
            m = cls()
            m.side = "left"
            names = [n for n in points] + ["femur head"]
            landmarks = [Landmark(n, None if n in ml.FEMUR_DISTAL_OPTIONAL else points[n]) for n in dict.fromkeys(names)]
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
