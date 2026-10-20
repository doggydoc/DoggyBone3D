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


def _section_loops(polydata, point, normal):
    """Closed loops (N x 3 arrays) where the plane (point, normal) cuts the surface."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(*[float(v) for v in point])
    plane.SetNormal(*[float(v) for v in normal])
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(polydata)
    merge = vtk.vtkCleanPolyData()              # coincident points of neighbouring cut segments -> one
    merge.SetInputConnection(cutter.GetOutputPort())
    merge.ToleranceIsAbsoluteOn()
    merge.SetAbsoluteTolerance(0.01)
    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(merge.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    out = stripper.GetOutput()
    if out is None or out.GetNumberOfPoints() < 3:
        return []
    P = vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    loops = []
    lines = out.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    while lines.GetNextCell(ids):
        if ids.GetNumberOfIds() >= 4:
            loops.append(np.array([P[ids.GetId(k)] for k in range(ids.GetNumberOfIds())]))
    return loops


def _polygon(loop, origin, e1, e2):
    rel = loop - origin
    return np.stack([rel @ e1, rel @ e2], axis=1)


def _area_centroid(xy):
    x, y = xy[:, 0], xy[:, 1]
    xr, yr = np.roll(x, -1), np.roll(y, -1)
    cross = x * yr - xr * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-9:
        return 0.0, xy.mean(axis=0)
    cx = float(((x + xr) * cross).sum()) / (6.0 * area)
    cy = float(((y + yr) * cross).sum()) / (6.0 * area)
    return abs(area), np.array([cx, cy])


def _inside(xy, q):
    """Point q (2D) inside the polygon xy (ray casting)."""
    x, y = xy[:, 0], xy[:, 1]
    xr, yr = np.roll(x, -1), np.roll(y, -1)
    crosses = ((y > q[1]) != (yr > q[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x + (q[1] - y) * (xr - x) / (yr - y)
    return int(np.count_nonzero(crosses & (q[0] < x_at))) % 2 == 1


def section_centre(polydata, point, normal):
    """(centre 3D, area mm^2) of the bone cross-section in the plane (point, normal), or (None, 0).
    The outer contour that contains `point` is used; if the point lies outside the bone (picked on
    the surface), the contour nearest to it and then the largest contour around that one."""
    n = _unit(normal)
    origin = np.asarray(point, float)
    loops = _section_loops(polydata, origin, n)
    if not loops:
        return None, 0.0
    e1, e2 = _frame(n)
    polys = [_polygon(loop, origin, e1, e2) for loop in loops]
    stats = [_area_centroid(xy) for xy in polys]
    q = np.zeros(2)
    containing = [k for k, xy in enumerate(polys) if stats[k][0] > 0.5 and _inside(xy, q)]
    if not containing:
        nearest = min(range(len(polys)), key=lambda k: float(np.min(np.linalg.norm(polys[k], axis=1))))
        if float(np.min(np.linalg.norm(polys[nearest], axis=1))) > 6.0:
            return None, 0.0                     # no bone near the picked point in this plane
        q = stats[nearest][1]
        containing = [k for k, xy in enumerate(polys) if stats[k][0] > 0.5 and _inside(xy, q)] or [nearest]
    best = max(containing, key=lambda k: stats[k][0])       # the OUTER contour (filled)
    area, c2 = stats[best]
    return origin + c2[0] * e1 + c2[1] * e2, area


def local_axis(polydata, point, radius=15.0):
    """Rough local bone axis: main direction of the surface points within `radius` of point."""
    P = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
    near = P[np.linalg.norm(P - np.asarray(point, float), axis=1) <= radius]
    if len(near) < 20:
        return None
    centred = near - near.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    return _unit(vt[0])


def diaphysis_centre(polydata, point, axis_hint=None, step=5.0, iterations=4):
    """(centre, axis, area) of the diaphysis at the height of `point`: area centroid of the section
    perpendicular to the local bone axis, the axis refined from the centroids at +-step mm.
    Returns (None, None, 0) if there is no bone there."""
    p = np.asarray(point, float)
    a = _unit(axis_hint) if axis_hint is not None else local_axis(polydata, p)
    if a is None:
        return None, None, 0.0
    for _k in range(iterations):
        c0, _area = section_centre(polydata, p, a)
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
    centre, area = section_centre(polydata, p, a)       # plane through the picked HEIGHT
    if centre is None:
        return None, None, 0.0
    return centre, a, area
