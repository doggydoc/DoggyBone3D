# DoggyBone3D - diaphysis centre / guide geometry regression tests (no GUI; Python with vtk + numpy + scipy):
#   shell:   PythonSlicer -m unittest <path>/test_guide_geometry.py
import os, sys, unittest
import numpy as np
import vtk

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from DoggyboneLib import section_centre as SC
from DoggyboneLib import guide_logic as G
from DoggyboneLib import helpers as H


def tube(radius, axis, height=120.0, resolution=64):
    """Closed cylinder (dense triangles) along `axis` through the origin."""
    source = vtk.vtkCylinderSource()                      # along y
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(resolution)
    source.CappingOn()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(source.GetOutputPort())
    sub = vtk.vtkLinearSubdivisionFilter()
    sub.SetInputConnection(tri.GetOutputPort())
    sub.SetNumberOfSubdivisions(3)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    y = np.array([0.0, 1.0, 0.0])
    rot = np.cross(y, a)
    transform = vtk.vtkTransform()
    if np.linalg.norm(rot) > 1e-9:
        transform.RotateWXYZ(float(np.degrees(np.arccos(np.clip(y @ a, -1, 1)))), *rot)
    place = vtk.vtkTransformPolyDataFilter()
    place.SetInputConnection(sub.GetOutputPort())
    place.SetTransform(transform)
    place.Update()
    return place.GetOutput()


def cube(centre, size):
    source = vtk.vtkCubeSource()
    source.SetCenter(*centre)
    source.SetXLength(size)
    source.SetYLength(size)
    source.SetZLength(size)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(source.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


class DiaphysisCentreTest(unittest.TestCase):

    def test_thick_and_thin_tubes(self):
        """The local axis must be found for thick bones too (radius >= ~10 mm: the surface patch is wider
        than long, its main direction transverse), from a point inside and on the surface."""
        axis = np.array([0.2, 1.0, 0.3])
        axis /= np.linalg.norm(axis)
        e1 = np.cross(axis, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        for radius in (6.0, 12.0, 15.0):
            surface = tube(radius, axis)
            for offset in (0.0, 0.5 * radius, radius):
                point = 5.0 * axis + offset * e1
                centre, a, area = SC.diaphysis_centre(surface, point)
                self.assertIsNotNone(centre, f"r={radius}, offset={offset}")
                self.assertGreater(abs(float(a @ axis)), np.cos(np.radians(1.0)))
                np.testing.assert_allclose(centre, 5.0 * axis, atol=0.1)
                # polygon area of the 64-gon, the outline wall not counted
                true = 0.5 * 64 * radius ** 2 * np.sin(2 * np.pi / 64)
                self.assertAlmostEqual(area / true, 1.0, delta=0.015)


class RasterizeTest(unittest.TestCase):

    def test_overlapping_shells_stay_solid(self):
        append = vtk.vtkAppendPolyData()
        append.AddInputData(cube((0.0, 0.0, 0.0), 4.0))
        append.AddInputData(cube((2.0, 0.0, 0.0), 4.0))
        append.Update()
        voxel = 0.25
        origin = np.array([-4.0, -4.0, -4.0])
        dims = (int(10.0 / voxel) + 1, int(8.0 / voxel) + 1, int(8.0 / voxel) + 1)

        def at(mask, p):
            i, j, k = np.rint((np.asarray(p) - origin) / voxel).astype(int)
            return bool(mask[i, j, k])

        even_odd = G._rasterize_closed(append.GetOutput(), origin, voxel, dims)
        self.assertFalse(at(even_odd, (1.0, 0.1, 0.1)))     # the defect: the overlap is a hole
        mask = G._rasterize_shells(append.GetOutput(), origin, voxel, dims)
        for p in ((1.0, 0.1, 0.1), (-1.5, 0.1, 0.1), (3.5, 0.1, 0.1)):
            self.assertTrue(at(mask, p), p)
        self.assertFalse(at(mask, (5.0, 0.1, 0.1)))
        single = G._rasterize_shells(cube((0.0, 0.0, 0.0), 4.0), origin, voxel, dims)
        np.testing.assert_array_equal(single, G._rasterize_closed(cube((0.0, 0.0, 0.0), 4.0), origin, voxel, dims))


class ReductionGuideTest(unittest.TestCase):

    def test_tilted_pin_is_an_error(self):
        pins = [dict(entry=np.array([0.0, 0.0, 0.0]), axis=np.array([0.0, 0.0, 1.0]), r_in=1.0),
                dict(entry=np.array([20.0, 0.0, 0.0]), axis=np.array([1.0, 0.0, 0.05]), r_in=1.0)]
        body, info = G.reduction_bar(pins, [0.0, 0.0, 1.0], voxel=0.5)
        self.assertIsNone(body)
        self.assertIn("error", info)
        pins[1]["axis"] = np.array([0.0, 0.0, -1.0])           # opposite direction
        self.assertIsNone(G.reduction_bar(pins, [0.0, 0.0, 1.0], voxel=0.5)[0])
        self.assertIsNone(G.reduction_web(pins, [0.0, 0.0, 1.0], vtk.vtkOBBTree(), voxel=0.5)[0])
        pins[1]["axis"] = np.array([0.0, 0.05, 1.0])           # nearly parallel: built
        body, info = G.reduction_bar(pins, [0.0, 0.0, 1.0], voxel=0.5)
        self.assertGreater(body.GetNumberOfPoints(), 0)
        self.assertNotIn("error", info)


class _LabelMap(vtk.vtkImageData):
    """vtkImageData with the GetImageToWorldMatrix of Slicer's oriented image (identity)."""

    def GetImageToWorldMatrix(self, matrix):
        matrix.Identity()


class ExternalMaskTest(unittest.TestCase):

    def test_mesh_outside_labelmap(self):
        labelmap = _LabelMap()
        labelmap.SetDimensions(10, 10, 10)
        labelmap.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        labelmap.GetPointData().GetScalars().Fill(1)
        for centre in ((200.0, 200.0, 200.0), (-200.0, -200.0, -200.0)):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*centre)
            sphere.SetRadius(5.0)
            sphere.Update()
            mesh = sphere.GetOutput()
            mask = G.external_mask(mesh, labelmap, 1)
            self.assertEqual(len(mask), mesh.GetNumberOfPoints())
            self.assertFalse(mask.any())


class HelpersTest(unittest.TestCase):

    def test_project_to_plane_zero_normal(self):
        np.testing.assert_allclose(H.project_to_plane([1.0, 2.0, 3.0], [0.0, 0.0, 2.0]), [1.0, 2.0, 0.0])
        with self.assertRaises(ValueError):
            H.project_to_plane([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
