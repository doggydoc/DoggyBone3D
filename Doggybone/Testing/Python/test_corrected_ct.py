# DoggyBone3D - simulated corrected CT tests (no GUI; needs VTK):
#   PythonSlicer -m unittest <path>/test_corrected_ct.py
import os, sys, unittest
import numpy as np
import vtk
from vtk.util import numpy_support

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from DoggyboneLib import corrected_ct as C

SPACING = 2.0
ORIGIN = np.array([-30.0, 10.0, 5.0])
IJK_TO_RAS = np.array([[SPACING, 0, 0, ORIGIN[0]], [0, SPACING, 0, ORIGIN[1]],
                       [0, 0, SPACING, ORIGIN[2]], [0, 0, 0, 1.0]])


def volume(bone=(10, 16)):
    """40^3 CT: soft tissue 20, a bone cube of 1000 (voxels bone[0]..bone[1]-1 in i, j, k)."""
    array = np.full((40, 40, 40), 20, dtype=np.int16)
    a, b = bone
    array[a:b, a:b, a:b] = 1000
    image = vtk.vtkImageData()
    image.SetDimensions(40, 40, 40)
    image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array.ravel(), deep=True,
                                                               array_type=vtk.VTK_SHORT))
    return image, array.copy()


def cube_ras(i0, i1):
    """Closed surface around the voxel centres i0..i1 (same range in i, j, k), in RAS."""
    lo = IJK_TO_RAS @ np.array([i0 - 0.5] * 3 + [1.0])
    hi = IJK_TO_RAS @ np.array([i1 + 0.5] * 3 + [1.0])
    source = vtk.vtkCubeSource()
    source.SetBounds(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(source.GetOutputPort())
    triangles.Update()
    return triangles.GetOutput()


def translation(dx, dy=0.0, dz=0.0):
    m = np.eye(4)
    m[:3, 3] = [dx, dy, dz]
    return m


class CorrectedCtTest(unittest.TestCase):

    def test_fragment_moves(self):
        image, out = volume()
        info = C.simulate_corrected_ct(image, out, IJK_TO_RAS, translation(5 * SPACING), cube_ras(10, 15))
        self.assertEqual(info["fill"], 20.0)
        self.assertTrue(np.all(out[10:16, 10:16, 15:21] == 1000))         # new place (i + 5)
        self.assertTrue(np.all(out[10:16, 10:16, 10:15] == 20))           # old place filled
        self.assertEqual(info["pasted_voxels"], 6 ** 3)
        self.assertEqual(int((out == 1000).sum()), 6 ** 3)                 # no ghost, nothing lost

    def test_original_untouched_and_rotation(self):
        image, out = volume()
        before = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).copy()
        centre = (IJK_TO_RAS @ np.array([12.5, 12.5, 12.5, 1.0]))[:3]
        r = np.eye(4)
        r[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]                     # 90 deg about z through the centre
        h = translation(*centre) @ r @ translation(*(-centre))
        C.simulate_corrected_ct(image, out, IJK_TO_RAS, h, cube_ras(10, 15))
        np.testing.assert_array_equal(numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()), before)
        self.assertTrue(np.all(out[10:16, 10:16, 10:16] == 1000))         # a cube turned about its centre

    def test_wedge_blanked(self):
        image, out = volume()
        array = out.copy()
        array[20:23, 20:23, 20:23] = 1000                                  # 'wedge' near the fragment
        out[:] = array
        image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(array.ravel(), deep=True, array_type=vtk.VTK_SHORT))
        wedge = cube_ras(20, 22)
        C.simulate_corrected_ct(image, out, IJK_TO_RAS, np.eye(4), cube_ras(10, 15), blank_ras=[wedge])
        self.assertTrue(np.all(out[10:16, 10:16, 10:16] == 1000))         # fragment stays (identity)
        self.assertTrue(np.all(out[20:23, 20:23, 20:23] == 20))           # wedge removed

    def test_padding_keeps_the_whole_fragment(self):
        # moved 30 voxels in +i: the cube (i 10..15) would end at i 45 - beyond the 40 voxels of the CT
        image, array = volume()
        h = translation(30 * SPACING)
        pad_low, dims, out_ijk_to_ras = C.padded_geometry((40, 40, 40), IJK_TO_RAS, h, cube_ras(10, 15))
        self.assertEqual(pad_low, (0, 0, 0))
        self.assertGreaterEqual(dims[0], 46)
        self.assertEqual(dims[1:], (40, 40))
        out = C.padded_copy(array, pad_low, dims, -1000)
        info = C.simulate_corrected_ct(image, out, IJK_TO_RAS, h, cube_ras(10, 15), out_ijk_to_ras=out_ijk_to_ras)
        self.assertEqual(info["pasted_voxels"], 6 ** 3)
        self.assertTrue(np.all(out[10:16, 10:16, 40:46] == 1000))         # beyond the old field of view
        self.assertTrue(np.all(out[:, :, 40:][out[:, :, 40:] != 1000] == -1000))
        # padding BEFORE index 0 shifts the origin: the CT stays at the same RAS position
        pad_low, dims, out_ijk_to_ras = C.padded_geometry((40, 40, 40), IJK_TO_RAS, translation(-20 * SPACING), cube_ras(10, 15))
        self.assertGreater(pad_low[0], 0)
        np.testing.assert_allclose(out_ijk_to_ras @ np.array([pad_low[0], 0, 0, 1.0]), IJK_TO_RAS @ np.array([0, 0, 0, 1.0]))
        out = C.padded_copy(array, pad_low, dims, -1000)
        C.simulate_corrected_ct(image, out, IJK_TO_RAS, translation(-20 * SPACING), cube_ras(10, 15), out_ijk_to_ras=out_ijk_to_ras)
        p = pad_low[0]
        self.assertTrue(np.all(out[10:16, 10:16, p - 10:p - 4] == 1000))  # i 10..15 minus 20 voxels
        self.assertEqual(int((out == 1000).sum()), 6 ** 3)

    def test_outside_volume(self):
        image, out = volume()
        info = C.simulate_corrected_ct(image, out, IJK_TO_RAS, translation(1000.0), cube_ras(10, 15))
        self.assertEqual(info["pasted_voxels"], 0)
        self.assertEqual(int((out == 1000).sum()), 0)                      # removed, moved out of the volume


if __name__ == "__main__":
    unittest.main()
