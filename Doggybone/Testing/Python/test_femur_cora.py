# DoggyBone3D - femur CORA from the distal partial axis at the aLDFA target (no GUI, no VTK):
#   Slicer:  exec(open(r"<path>/test_femur_cora.py").read())      or
#   shell:   python3 -m unittest <path>/test_femur_cora.py
import os, sys, unittest
import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from DoggyboneLib import measurement_logic as ml


def femur(aldfa, cora_z=30.0, target=94.0, sagittal_offset=0.0):
    """Synthetic distal varus: condyles on the x axis, the distal segment stands at `target`, the
    proximal segment at `aldfa`; both meet at (0, 0, cora_z)."""
    e1, z = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    direction = lambda deg: np.cos(np.radians(deg)) * e1 + np.sin(np.radians(deg)) * z
    cora = np.array([0.0, 0.0, cora_z])
    p, u = direction(aldfa), direction(target)
    points = {"medial femur condyle": np.array([-15.0, 0, 0]), "lateral femur condyle": np.array([15.0, 0, 0]),
              "Femur midpoint 33%": cora + 120 * p, "Femur midpoint 50%": cora + 70 * p,
              ml.FEMUR_DISTAL_ANCHOR: cora - 18 * u + np.array([0, sagittal_offset, 0])}
    radii = {"medial femur condyle": 9.0, "lateral femur condyle": 9.0}
    return points, radii, cora


class FemurCoraTest(unittest.TestCase):
    def tearDown(self):
        ml.set_aldfa_target_provider(None)

    def test_varus_cora(self):
        points, radii, cora = femur(102.0)
        m = ml.FemurCoraMeasurement()
        value, _ = m._measure(points, radii)
        self.assertAlmostEqual(value, 8.0, places=6)
        np.testing.assert_allclose(m.info["cora"], cora, atol=1e-6)
        self.assertAlmostEqual(m.info["cora_level"], 30.0, places=6)
        self.assertIn(ml.FEMUR_CORA, m.spheres)
        self.assertTrue(any("varus to correct" in line for line in m.result_lines(value, "")))

    def test_valgus_and_planned_target(self):
        ml.set_aldfa_target_provider(lambda: 96.0)
        points, radii, cora = femur(90.0, cora_z=25.0, target=96.0)
        m = ml.FemurCoraMeasurement()
        value, _ = m._measure(points, radii)
        self.assertAlmostEqual(value, -6.0, places=6)
        np.testing.assert_allclose(m.info["cora"], cora, atol=1e-6)

    def test_sagittal_offset_of_the_anchor_does_not_move_the_cora(self):
        points, radii, cora = femur(104.0, sagittal_offset=3.0)
        m = ml.FemurCoraMeasurement()
        m._measure(points, radii)
        np.testing.assert_allclose(m.info["cora"], cora, atol=1e-6)

    def test_no_deformity_no_cora(self):
        points, radii, _cora = femur(94.0)
        m = ml.FemurCoraMeasurement()
        value, _ = m._measure(points, radii)
        self.assertAlmostEqual(value, 0.0, places=6)
        self.assertNotIn("cora", m.info)
        self.assertNotIn(ml.FEMUR_CORA, m.spheres)

    def test_bad_target_falls_back_to_reference(self):
        ml.set_aldfa_target_provider(lambda: 0.0)
        self.assertEqual(ml.aldfa_target(), ml.ALDFA_REFERENCE_DEG)


if __name__ == "__main__" or "slicer" in sys.modules:
    unittest.main(argv=["femur_cora"], exit=False, verbosity=1)
