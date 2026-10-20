# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""Centre of a long bone's diaphysis as the AREA CENTROID of its cross-section.

The section plane is perpendicular to the LOCAL bone axis; that axis is refined iteratively from the
centroids of two neighbouring sections, so an oblique first guess does not bias the result. The
filled OUTER contour is used (medullary cavity included), so a thin or incomplete cortex hardly
matters, and neighbouring bones in the same plane (fibula, patella, fabellae) are left out: only the
contour that contains the picked point - or lies nearest to it - counts.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _frame(n):
    n = _unit(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(n, helper))
    return e1, np.cross(n, e1)


def _section_segments(polydata, point, normal):
    """Line segments (M x 2 x 3) where the plane (point, normal) cuts the surface."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(*[float(v) for v in point])
    plane.SetNormal(*[float(v) for v in normal])
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(polydata)
    cutter.Update()
    out = cutter.GetOutput()
    if out is None or out.GetNumberOfPoints() < 2 or out.GetNumberOfLines() == 0:
        return None
    P = vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    lines = vtk_to_numpy(out.GetLines().GetData())
    # cutter output: [2, i, j, 2, i, j, ...]
    if len(lines) % 3 == 0 and np.all(lines[0::3] == 2):
        pairs = np.stack([lines[1::3], lines[2::3]], axis=1)
    else:
        pairs, k = [], 0
        while k < len(lines):
            m = int(lines[k])
            ids = lines[k + 1:k + 1 + m]
            pairs += [(ids[j], ids[j + 1]) for j in range(m - 1)]
            k += m + 1
        pairs = np.asarray(pairs, int)
    return P[pairs]


def section_centre(polydata, point, normal, reach=40.0, res=0.1, max_distance=6.0):
    """(centre 3D, area mm^2) of the bone cross-section in the plane (point, normal), or (None, 0).
    The section is rasterised (res mm pixels): everything enclosed by the cut outline is filled -
    the medullary canal too - and the filled region that contains `point` (or, for a point picked
    on the surface, the region nearest to it, at most 6 mm away) gives the area centroid.
    Robust against small gaps in the outline."""
    from scipy import ndimage as ndi
    n = _unit(normal)
    origin = np.asarray(point, float)
    seg = _section_segments(polydata, origin, n)
    if seg is None:
        return None, 0.0
    e1, e2 = _frame(n)
    xy = np.stack([(seg - origin) @ e1, (seg - origin) @ e2], axis=-1)        # (M, 2, 2)
    keep = np.all(np.abs(xy) <= reach, axis=(1, 2))
    xy = xy[keep]
    if len(xy) == 0:
        return None, 0.0
    lo = xy.reshape(-1, 2).min(axis=0) - 1.0
    hi = xy.reshape(-1, 2).max(axis=0) + 1.0
    lo = np.minimum(lo, -1.0)
    hi = np.maximum(hi, 1.0)
    shape = tuple(int(np.ceil((hi[k] - lo[k]) / res)) + 1 for k in range(2))
    wall = np.zeros(shape, dtype=bool)
    lengths = np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1)
    steps = np.maximum(2, np.ceil(lengths / (0.5 * res)).astype(int) + 1)
    for count in np.unique(steps):
        sel = steps == count
        t = np.linspace(0.0, 1.0, int(count))[None, :, None]
        pts = xy[sel, 0][:, None, :] + t * (xy[sel, 1] - xy[sel, 0])[:, None, :]
        idx = np.rint((pts.reshape(-1, 2) - lo) / res).astype(int)
        wall[idx[:, 0], idx[:, 1]] = True
    wall = ndi.binary_dilation(wall)                       # close sub-pixel gaps of the outline
    free, _n = ndi.label(~wall)
    border = set(np.unique(np.concatenate([free[0, :], free[-1, :], free[:, 0], free[:, -1]]))) - {0}
    outside = np.isin(free, list(border))
    filled = ~outside                                      # outline + everything it encloses
    regions, count = ndi.label(filled)
    if count == 0:
        return None, 0.0
    q = np.rint((np.zeros(2) - lo) / res).astype(int)
    label = int(regions[q[0], q[1]]) if (0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]) else 0
    if label == 0:
        dist, (ia, ib) = ndi.distance_transform_edt(regions == 0, return_indices=True)
        if dist[q[0], q[1]] * res > max_distance:
            return None, 0.0                               # no bone near the picked point in this plane
        label = int(regions[ia[q[0], q[1]], ib[q[0], q[1]]])
    ii, jj = np.nonzero(regions == label)
    area = len(ii) * res * res
    cx = lo[0] + float(ii.mean()) * res
    cy = lo[1] + float(jj.mean()) * res
    return origin + cx * e1 + cy * e2, area


def local_axis(polydata, point, radius=15.0):
    """Rough local bone axis: main direction of the surface points within `radius` of point."""
    P = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
    near = P[np.linalg.norm(P - np.asarray(point, float), axis=1) <= radius]
    if len(near) < 20:
        return None
    centred = near - near.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    return _unit(vt[0])


def diaphysis_centre(polydata, point, axis_hint=None, step=5.0, iterations=4, max_distance=6.0):
    """(centre, axis, area) of the diaphysis at the height of `point`: area centroid of the section
    perpendicular to the local bone axis, the axis refined from the centroids at +-step mm.
    Returns (None, None, 0) if there is no bone there."""
    p = np.asarray(point, float)
    a = _unit(axis_hint) if axis_hint is not None else local_axis(polydata, p)
    if a is None:
        return None, None, 0.0
    for _k in range(iterations):
        c0, _area = section_centre(polydata, p, a, max_distance=max_distance)
        if c0 is None:
            return None, None, 0.0
        cp, _ = section_centre(polydata, c0 + step * a, a)
        cm, _ = section_centre(polydata, c0 - step * a, a)
        if cp is None or cm is None:
            break
        a_new = _unit(cp - cm)
        if float(np.dot(a_new, a)) < 0.0:
            a_new = -a_new
        change = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a_new, a)))))))
        a = a_new
        if change < 0.2:
            break
    centre, area = section_centre(polydata, p, a, max_distance=max_distance)   # plane through the picked HEIGHT
    if centre is None:
        return None, None, 0.0
    return centre, a, area
