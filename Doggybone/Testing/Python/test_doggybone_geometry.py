# Doggybone 3D - geometry regression tests (no GUI; run inside Slicer's Python or any Python with vtk + numpy):
#   Slicer:  exec(open(r"<path>/test_doggybone_geometry.py").read())      or
#   shell:   PythonSlicer -m unittest <path>/test_doggybone_geometry.py
import os, sys, unittest, types
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DoggyboneLib")


def _load(name):
    module = types.ModuleType("_dg_" + name)
    path = os.path.normpath(os.path.join(_LIB, name + ".py"))
    module.__file__ = path
    exec(compile(open(path).read(), path, "exec"), module.__dict__)
    return module


ol = _load("osteotomy_logic")


def H(m):
    return ol.matrix_to_numpy(m)


def apply(Hm, p):
    return (Hm @ np.r_[np.asarray(p, float), 1.0])[:3]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def bone(radius=6.0, length=80.0):
    c = vtk.vtkCylinderSource(); c.SetRadius(radius); c.SetHeight(length); c.SetResolution(96); c.CappingOn()
    t = vtk.vtkTransform(); t.RotateX(90)
    f = vtk.vtkTransformPolyDataFilter(); f.SetInputConnection(c.GetOutputPort()); f.SetTransform(t)
    tri = vtk.vtkTriangleFilter(); tri.SetInputConnection(f.GetOutputPort())
    sub = vtk.vtkLinearSubdivisionFilter(); sub.SetInputConnection(tri.GetOutputPort()); sub.SetNumberOfSubdivisions(3)
    sub.Update()
    return sub.GetOutput()


def area(pd):
    if pd is None or pd.GetNumberOfCells() == 0:
        return 0.0
    m = vtk.vtkMassProperties(); m.SetInputData(pd); m.Update()
    return m.GetSurfaceArea()


def points(pd):
    if pd is None or pd.GetNumberOfPoints() == 0:
        return np.zeros((0, 3))
    return vtk_to_numpy(pd.GetPoints().GetData()).astype(float)


SHAFT = unit([0.1, -0.05, 1.0])
AP = unit(np.cross(SHAFT, [1.0, 0.0, 0.0]))
SHAFT_POINT = np.array([3.0, -2.0, 50.0])
HINGE = np.array([9.0, 1.0, 48.0])
ANGLES = (0.0, 30.0, -30.0, 60.0, -80.0, 90.0, -90.0)


class RigidCorrection(unittest.TestCase):
    """anatomical_correction_matrix: rigid, fixed axes, decoupled for any magnitude."""

    def test_rigid(self):
        for w in ANGLES:
            for t in ANGLES:
                for frontal_first in (True, False):
                    Hm = H(ol.anatomical_correction_matrix(SHAFT_POINT, SHAFT, t, HINGE, AP, w, frontal_first))
                    R = Hm[:3, :3]
                    self.assertTrue(np.allclose(R.T @ R, np.eye(3), atol=1e-9))
                    self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)

    def test_pure_torsion_keeps_shaft_line(self):
        for t in ANGLES:
            Hm = H(ol.anatomical_correction_matrix(SHAFT_POINT, SHAFT, t, HINGE, AP, 0.0))
            for s in (-100.0, 0.0, 85.0):
                p = SHAFT_POINT + s * SHAFT
                self.assertLess(np.linalg.norm(apply(Hm, p) - p), 1e-9)

    def test_pure_wedge_keeps_hinge_line(self):
        for w in ANGLES:
            Hm = H(ol.anatomical_correction_matrix(SHAFT_POINT, SHAFT, 0.0, HINGE, AP, w))
            for s in (-40.0, 0.0, 40.0):
                p = HINGE + s * AP
                self.assertLess(np.linalg.norm(apply(Hm, p) - p), 1e-9)

    def test_decoupled_large_angles(self):
        """Distal fragment moves (frontal first): the frontal angle of the condylar axis to the
        FIXED shaft equals the wedge angle, whatever the torsion (up to 90 deg)."""
        ml = unit(np.cross(AP, SHAFT))                 # condylar axis, perpendicular to shaft and AP
        for w in (0.0, 20.0, -35.0, 60.0):
            for t in ANGLES:
                R = H(ol.anatomical_correction_matrix(SHAFT_POINT, SHAFT, t, HINGE, AP, w, True))[:3, :3]
                ml_new = R @ ml
                frontal = np.degrees(np.arctan2(float(ml_new @ SHAFT), float(np.hypot(ml_new @ ml, ml_new @ AP))))
                self.assertAlmostEqual(abs(frontal), abs(w), places=6, msg=f"wedge {w} torsion {t}")


class ThreeRegionSplit(unittest.TestCase):
    """split_bone_three_regions: proximal / wedge / distal are disjoint and complete."""

    def _cases(self):
        for hinge_x in (7.0, 2.0, -3.0):                 # hinge outside and inside the bone
            for deg in (10.0, 30.0):
                for opening in (1.0, -1.0):
                    for flip_p in (1.0, -1.0):          # Markups normals may point either way
                        for flip_d in (1.0, -1.0):
                            a = np.radians(deg)
                            h = np.array([hinge_x, 0.0, 0.0])
                            n_p = flip_p * np.array([0.0, 0.0, 1.0])
                            n_d = flip_d * np.array([-opening * np.sin(a), 0.0, np.cos(a)])
                            yield h, n_p, n_d

    def test_partition(self):
        b = bone()
        total = area(b)
        for h, n_p, n_d in self._cases():
            prox, wedge, dist = ol.split_bone_three_regions(b, h, n_p, h, n_d, [0.0, 0.0, 1.0])
            self.assertAlmostEqual((area(prox) + area(wedge) + area(dist)) / total, 1.0, places=6)
            sp = lambda P: (P - h) @ (n_p * np.sign(n_p[2]))
            sd = lambda P: (P - h) @ (n_d * np.sign(n_d[2]))
            self.assertTrue(np.all(sp(points(prox)) <= 1e-6))
            self.assertTrue(np.all((sp(points(wedge)) >= -1e-6) & (sd(points(wedge)) <= 1e-6)))
            self.assertTrue(np.all((sp(points(dist)) >= -1e-6) & (sd(points(dist)) >= -1e-6)))

    def test_clip_orientation(self):
        b = bone()
        for flip in (1.0, -1.0):
            prox, dist = ol.clip_fragments(b, [0, 0, 5.0], [0, 0, flip], [0, 0, 1.0])
            self.assertLess(points(prox)[:, 2].max(), 5.0 + 1e-6)
            self.assertGreater(points(dist)[:, 2].min(), 5.0 - 1e-6)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"], exit=False)
