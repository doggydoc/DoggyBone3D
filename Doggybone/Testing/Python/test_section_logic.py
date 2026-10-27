# DoggyBone3D - cross-section centre tests (no GUI, no VTK):
#   shell:   python3 -m unittest <path>/test_section_logic.py
import os, sys, unittest
import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from DoggyboneLib import section_logic as S


def ellipse(cx, cy, a, b, n=64, reverse=False):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], axis=1)
    return pts[::-1] if reverse else pts


class SectionTest(unittest.TestCase):

    def test_area_centroid(self):
        area, c = S.polygon_area_centroid(ellipse(3.0, -2.0, 6.0, 4.0))
        np.testing.assert_allclose(c, [3.0, -2.0], atol=1e-9)
        self.assertAlmostEqual(abs(area), np.pi * 24, delta=0.2)

    def test_area_centroid_not_vertex_mean(self):
        """Unevenly spaced contour points (dense on one side) must not pull the centre."""
        t = np.concatenate([np.linspace(0, np.pi, 200, endpoint=False), np.linspace(np.pi, 2 * np.pi, 10, endpoint=False)])
        loop = np.stack([5 * np.cos(t), 5 * np.sin(t)], axis=1)
        _a, c = S.polygon_area_centroid(loop)
        np.testing.assert_allclose(c, [0.0, 0.0], atol=0.15)
        self.assertGreater(np.linalg.norm(loop.mean(axis=0)), 1.0)        # the plain mean would be off

    def test_bone_next_to_neighbour_and_hollow(self):
        bone_outer = ellipse(0.0, 0.0, 7.0, 4.0)
        bone_canal = ellipse(0.5, 0.0, 3.0, 1.5, reverse=True)
        neighbour = ellipse(0.0, -9.0, 3.0, 3.0)
        loops = [neighbour, bone_canal, bone_outer]
        index, c, area = S.pick_loop_centroid(loops, seed_2d=[1.0, 0.2])           # inside the canal
        self.assertEqual(index, 2)
        np.testing.assert_allclose(c, [0.0, 0.0], atol=1e-9)
        index, c, _ = S.pick_loop_centroid(loops, seed_2d=[6.0, 0.5])            # in the cortex
        self.assertEqual(index, 2)
        index, c, _ = S.pick_loop_centroid(loops, seed_2d=[0.0, -9.5])           # on the neighbouring bone
        self.assertEqual(index, 0)
        np.testing.assert_allclose(c, [0.0, -9.0], atol=1e-9)

    def test_seed_just_outside(self):
        loops = [ellipse(0.0, 0.0, 7.0, 4.0)]
        index, c, _ = S.pick_loop_centroid(loops, seed_2d=[7.8, 0.0])
        self.assertEqual(index, 0)
        index, c, _ = S.pick_loop_centroid(loops, seed_2d=[30.0, 0.0])
        self.assertIsNone(index)

    def test_plane_basis(self):
        for n in ([0, 0, 1], [1, 0, 0], [0.3, -0.8, 0.5]):
            u, v = S.plane_basis(n)
            n = np.asarray(n, float) / np.linalg.norm(n)
            self.assertAlmostEqual(float(u @ n), 0.0, places=9)
            self.assertAlmostEqual(float(v @ n), 0.0, places=9)
            self.assertAlmostEqual(float(u @ v), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
