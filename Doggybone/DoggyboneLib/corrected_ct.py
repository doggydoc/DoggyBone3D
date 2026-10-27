# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""Simulated CT of the planned correction (numpy + VTK, no MRML - testable without the GUI).

The moving fragment (closed bone surface from the correction preview, in its ORIGINAL position) is
cut out of a copy of the CT and pasted back where the planned correction (incl. fine shift) puts it:
  1. the fragment's original place - and a removed CWO wedge - is filled with the median soft-tissue
     value around it (no ghost of the old bone),
  2. the CT is resampled through the correction (linear interpolation) and copied in inside the
     fragment's corrected surface.
Only the bone inside the fragment surface moves: soft tissue and a neighbouring bone (fibula) stay where
they are, and the osteotomy gap / overlap is not smoothed. A planning picture, not a diagnostic image.
If the correction moves part of the fragment out of the scanned field of view, the grid is enlarged
(padded_geometry / padded_copy) so the fragment is not cut off.

All arrays are indexed [k, j, i] like slicer.util.arrayFromVolume; matrices are 4x4 numpy arrays."""

import numpy as np
import vtk
from vtk.util import numpy_support

SOFT_TISSUE_RANGE = (-200.0, 150.0)      # values counted as soft tissue for the fill value (HU)
SOFT_TISSUE_DEFAULT = 40.0               # fill value if no soft tissue is found around the fragment


def transform_polydata(polydata, matrix):
    """Copy of `polydata` with every point mapped by the 4x4 `matrix`."""
    m = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(matrix[r][c]))
    t = vtk.vtkTransform()
    t.SetMatrix(m)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputData(polydata)
    f.SetTransform(t)
    f.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(f.GetOutput())
    return out


def bounding_extent(polydatas, dims, margin=1):
    """[i0, i1, j0, j1, k0, k1] around the points of the (IJK) polydatas, clipped to the volume; None
    if they lie completely outside."""
    bounds = np.array([p.GetBounds() for p in polydatas if p is not None and p.GetNumberOfPoints()])
    if not len(bounds):
        return None
    low = np.floor(bounds[:, 0::2].min(axis=0)).astype(int) - margin
    high = np.ceil(bounds[:, 1::2].max(axis=0)).astype(int) + margin
    low = np.maximum(low, 0)
    high = np.minimum(high, np.asarray(dims) - 1)
    if np.any(high < low):
        return None
    return [int(low[0]), int(high[0]), int(low[1]), int(high[1]), int(low[2]), int(high[2])]


def _shape(extent):
    return (extent[5] - extent[4] + 1, extent[3] - extent[2] + 1, extent[1] - extent[0] + 1)


def _sub(array, extent):
    """View of `array` [k, j, i] for the extent."""
    return array[extent[4]:extent[5] + 1, extent[2]:extent[3] + 1, extent[0]:extent[1] + 1]


def surface_mask(polydata_ijk, extent):
    """Boolean [k, j, i] mask of the voxels inside the closed (IJK) surface, for the extent."""
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(polydata_ijk)
    stencil.SetOutputOrigin(0.0, 0.0, 0.0)
    stencil.SetOutputSpacing(1.0, 1.0, 1.0)
    stencil.SetOutputWholeExtent(*extent)
    stencil.Update()
    to_image = vtk.vtkImageStencilToImage()
    to_image.SetInputConnection(stencil.GetOutputPort())
    to_image.SetInsideValue(1)
    to_image.SetOutsideValue(0)
    to_image.SetOutputScalarTypeToUnsignedChar()
    to_image.Update()
    values = numpy_support.vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars())
    return values.reshape(_shape(extent)).astype(bool)


def resample(image_ijk, out_to_in, extent, background):
    """[k, j, i] array of `image_ijk` (vtkImageData, origin 0, spacing 1) sampled at out_to_in @ ijk
    for every voxel of the extent (linear interpolation, `background` outside the image)."""
    m = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(out_to_in[r][c]))
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(image_ijk)
    reslice.SetResliceAxes(m)
    reslice.SetOutputOrigin(0.0, 0.0, 0.0)
    reslice.SetOutputSpacing(1.0, 1.0, 1.0)
    reslice.SetOutputExtent(*extent)
    reslice.SetInterpolationModeToLinear()
    reslice.SetBackgroundLevel(float(background))
    reslice.Update()
    values = numpy_support.vtk_to_numpy(reslice.GetOutput().GetPointData().GetScalars())
    return values.reshape(_shape(extent))


def soft_tissue_value(values, mask, value_range=SOFT_TISSUE_RANGE, default=SOFT_TISSUE_DEFAULT):
    """Median of the voxels OUTSIDE the mask whose value lies in the soft-tissue range."""
    around = values[~mask]
    around = around[(around >= value_range[0]) & (around <= value_range[1])]
    return float(np.median(around)) if around.size else float(default)


def padded_geometry(dims, ijk_to_ras, correction, moving_ras, margin=4):
    """Grid that holds the CT AND the corrected fragment: a correction can move the far end of a
    fragment out of the scanned field of view, which would cut it off. Returns (pad_low, out_dims,
    out_ijk_to_ras): voxels added before index 0 (i, j, k), the new dimensions and the new IJK->RAS
    (same spacing / direction, origin moved by pad_low). No padding needed: (0,0,0), dims, ijk_to_ras."""
    ijk_to_ras = np.asarray(ijk_to_ras, dtype=float)
    corrected = transform_polydata(moving_ras, np.linalg.inv(ijk_to_ras) @ np.asarray(correction, dtype=float))
    b = np.array(corrected.GetBounds(), dtype=float)
    low = np.floor(b[0::2]).astype(int) - margin
    high = np.ceil(b[1::2]).astype(int) + margin
    pad_low = np.maximum(-low, 0)
    pad_high = np.maximum(high - (np.asarray(dims) - 1), 0)
    out_dims = tuple(int(d) for d in np.asarray(dims) + pad_low + pad_high)
    shift = np.eye(4)
    shift[:3, 3] = -pad_low
    return tuple(int(p) for p in pad_low), out_dims, ijk_to_ras @ shift


def padded_copy(array, pad_low, out_dims, pad_value):
    """Copy of the CT array [k, j, i] on the grid of padded_geometry; added voxels get pad_value
    (the CT's own 'outside the scan' value)."""
    out = np.full(tuple(reversed(out_dims)), pad_value, dtype=array.dtype)
    i, j, k = pad_low
    out[k:k + array.shape[0], j:j + array.shape[1], i:i + array.shape[2]] = array
    return out


def simulate_corrected_ct(image_ijk, out_array, ijk_to_ras, correction, moving_ras, blank_ras=(), margin=6,
                          out_ijk_to_ras=None):
    """Write the simulated correction into `out_array` ([k, j, i]): a COPY of the CT that `image_ijk`
    holds unchanged - on the same grid, or on a larger one (out_ijk_to_ras, see padded_geometry) with
    the CT already copied in. correction: 4x4 RAS matrix of the moving fragment (incl. fine shift);
    moving_ras: its closed surface in the ORIGINAL position; blank_ras: further surfaces removed
    without being pasted back (CWO wedge). Returns a dict with the fill value and voxel counts."""
    ijk_to_ras = np.asarray(ijk_to_ras, dtype=float)
    out_ijk_to_ras = ijk_to_ras if out_ijk_to_ras is None else np.asarray(out_ijk_to_ras, dtype=float)
    correction = np.asarray(correction, dtype=float)
    ras_to_ijk = np.linalg.inv(out_ijk_to_ras)                 # surfaces on the OUTPUT grid
    dims = tuple(reversed(out_array.shape))                    # (i, j, k)
    original_ijk = transform_polydata(moving_ras, ras_to_ijk)
    corrected_ijk = transform_polydata(moving_ras, ras_to_ijk @ correction)
    blank_ijk = [original_ijk] + [transform_polydata(p, ras_to_ijk) for p in blank_ras
                                  if p is not None and p.GetNumberOfPoints()]

    # 1. the fragment's (and the wedge's) original place -> soft tissue
    fill, blanked = SOFT_TISSUE_DEFAULT, 0
    blank_extent = bounding_extent(blank_ijk, dims, margin)
    if blank_extent is not None:
        mask = np.zeros(_shape(blank_extent), dtype=bool)
        for polydata in blank_ijk:
            mask |= surface_mask(polydata, blank_extent)
        region = _sub(out_array, blank_extent)
        fill = soft_tissue_value(region, mask)
        region[mask] = fill
        blanked = int(mask.sum())

    # 2. the fragment in its corrected position, resampled from the ORIGINAL CT
    pasted = 0
    paste_extent = bounding_extent([corrected_ijk], dims, 1)
    if paste_extent is not None:
        mask = surface_mask(corrected_ijk, paste_extent)
        # output voxel -> RAS -> back through the correction -> voxel of the original CT
        moved = resample(image_ijk, np.linalg.inv(ijk_to_ras) @ np.linalg.inv(correction) @ out_ijk_to_ras,
                         paste_extent, fill)
        region = _sub(out_array, paste_extent)
        region[mask] = moved[mask].astype(out_array.dtype)
        pasted = int(mask.sum())
    return {"fill": fill, "blanked_voxels": blanked, "pasted_voxels": pasted}
