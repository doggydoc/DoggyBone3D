# Doggybone 3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

'''
Geometry for corrective osteotomy planning.

This builds the rigid-body correction that carries a bone's distal fragment from its
as-scanned position to a corrected position at an osteotomy site, combining:
  - a wedge (angular) correction for varus/valgus deformity, and
  - a rotational (derotation) correction for torsion/antetorsion deformity,
using anatomically distinct axes: torsion about the true shaft line and angular correction
about the AP/hinge line. This avoids the artificial translation produced by rotating torsion
about an eccentric cortical hinge.

Kept separate from the UI (Doggybone.py) and from measurement_logic.py - the same
split already used elsewhere in this module (segmentation/landmark logic vs. measurement logic).

IMPORTANT - sign convention: the UI derives signed correction angles from measured anatomy and
postoperative target values.  The geometry layer intentionally remains sign-agnostic and applies
the signed angles supplied by the caller.  Left/right clinical sign conventions must therefore
still be verified in the 3D preview before surgical use.
'''
import numpy as np
import vtk


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def plane_frame(shaft_axis, reference_ml_vector):
    '''
    Builds an orthonormal frame (shaft, ml, ap) at the osteotomy site.

    shaft_axis: the osteotomy plane's normal (proximal -> distal direction along the bone).
    reference_ml_vector: an approximate medial->lateral (or lateral->medial - caller decides the
      sign) vector near the osteotomy level, e.g. the vector between the two condyle landmarks
      already placed for the angle measurements (medial/lateral femur condyle, or condylus
      medialis/lateralis tibiae). Only used for its direction in the plane, so it doesn't need
      to be exactly at the osteotomy level or exactly perpendicular to shaft_axis.

    Returns three unit numpy vectors (shaft, ml, ap), right-handed (ap = shaft x ml).
    '''
    shaft = normalize(shaft_axis)
    ml_raw = np.asarray(reference_ml_vector, dtype=float)
    ml_in_plane = ml_raw - np.dot(ml_raw, shaft) * shaft
    ml = normalize(ml_in_plane)
    ap = normalize(np.cross(shaft, ml))
    return shaft, ml, ap


def rotation_matrix_about_line(point, axis, angle_deg):
    """Return a vtkMatrix4x4 for a rotation about an arbitrary 3-D line."""
    point = np.asarray(point, dtype=float)
    axis = normalize(axis)
    angle = np.deg2rad(float(angle_deg))
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    R = np.array([
        [x*x*C+c,   x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, y*y*C+c,   y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, z*z*C+c],
    ], dtype=float)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = point - R @ point
    out = vtk.vtkMatrix4x4()
    for r in range(4):
        for col in range(4):
            out.SetElement(r, col, float(H[r, col]))
    return out


def matrix_to_numpy(matrix):
    return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)


def numpy_to_vtk_matrix(H):
    out = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            out.SetElement(r, c, float(H[r, c]))
    return out


def closest_point_on_line(point, line_point, line_direction):
    """Orthogonal projection of point onto an infinite 3-D line."""
    p = np.asarray(point, dtype=float)
    a = np.asarray(line_point, dtype=float)
    d = normalize(line_direction)
    return a + np.dot(p - a, d) * d


def anatomical_correction_matrix(shaft_point, shaft_axis, rotation_angle_deg,
                                  hinge_point, wedge_axis, wedge_angle_deg, frontal_first=True):
    """Build the rigid correction using two anatomically distinct rotation axes.

    1) Torsion/derotation is performed about the TRUE shaft line
       (shaft_point + t * shaft_axis).
    2) Varus/valgus is performed about the wedge/hinge line
       (hinge_point + t * wedge_axis), normally the anatomical AP direction.

    Rule: the derotation always turns the MOVING fragment about its OWN long axis while that axis
    lies on the shaft line.
    frontal_first=True  (DISTAL fragment moves):   H = H_torsion @ H_wedge - the frontal step aligns
        the fragment with the (fixed) shaft line, then it is derotated about it.
    frontal_first=False (PROXIMAL fragment moves): H = H_wedge @ H_torsion - the shaft axis belongs to
        the moving fragment, so it is derotated first (axis still in place), then angulated.
    Details for the distal case:
    The varus/valgus rotation acts on the fragment in its scanned pose, where the AP axis is
    perpendicular to the plane of shaft and condylar axis, so it changes the frontal angle
    fully. The derotation then turns the aligned fragment about the fixed shaft line, which
    leaves the frontal angle unchanged. The two corrections are therefore decoupled for ANY
    magnitude (60-80 deg torsion included) and match the derived second cutting plane
    (_sync_other_plane: frontal step, then torsion about the same shaft line).
    (Earlier H_wedge @ H_torsion rotated first: after a large torsion the fixed AP axis no
    longer corrected varus/valgus and fed back into the torsion.)
    Both rotations keep their own axes (shaft line / hinge line) - never one common hinge.
    The returned matrix maps the moving fragment from its scanned pose into its corrected pose.
    """
    torsion = rotation_matrix_about_line(shaft_point, shaft_axis, rotation_angle_deg)
    wedge = rotation_matrix_about_line(hinge_point, wedge_axis, wedge_angle_deg)
    if frontal_first:
        H = matrix_to_numpy(torsion) @ matrix_to_numpy(wedge)
    else:
        H = matrix_to_numpy(wedge) @ matrix_to_numpy(torsion)
    return numpy_to_vtk_matrix(H)


def transform_direction(matrix, vector):
    """Transform a direction by the rotational part of a homogeneous matrix."""
    H = matrix_to_numpy(matrix)
    return normalize(H[:3, :3] @ np.asarray(vector, dtype=float))


def correction_matrix(hinge_point, ap_axis, wedge_angle_deg, shaft_axis, rotation_angle_deg):
    """Backward-compatible wrapper for old callers.

    New code should use anatomical_correction_matrix(), because this legacy API
    cannot specify a separate point on the shaft axis.
    """
    return anatomical_correction_matrix(
        hinge_point, shaft_axis, rotation_angle_deg,
        hinge_point, ap_axis, wedge_angle_deg)


def combine_fragments(fixed_polydata, moving_polydata, correction_matrix):
    '''
    Bakes correction_matrix into moving_polydata and appends it to fixed_polydata into one
    merged polydata - a single continuous "as if healed" preview of the corrected bone, rather
    than two separately colored/positioned fragments. Purely a display convenience for the final
    correction preview; the underlying fragment nodes and their live transform (on whichever
    fragment moves - proximal or distal, see onPreviewCorrection's reference-plane choice) remain
    the actual source of truth for the correction.
    '''
    transform = vtk.vtkTransform()
    transform.SetMatrix(correction_matrix)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(moving_polydata)
    transform_filter.Update()

    appender = vtk.vtkAppendPolyData()
    appender.AddInputData(fixed_polydata)
    appender.AddInputData(transform_filter.GetOutput())
    appender.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(appender.GetOutput())
    return result


def clip_fragments(bone_polydata, plane_origin, plane_normal, proximal_to_distal_axis=None):
    """Split bone at a plane and return (proximal, distal) robustly.

    If ``proximal_to_distal_axis`` is supplied, it is the authoritative anatomical direction.
    The plane normal is first oriented to point in that direction.  The two vtkClipPolyData
    outputs are then classified from their actual signed position relative to the plane rather
    than relying on VTK's output/clipped-output convention.  This avoids the historical
    InsideOut/normal-sign inversion that could remove bone above/below the intended wedge.
    """
    origin = np.asarray(plane_origin, dtype=float)
    normal = normalize(plane_normal)
    if proximal_to_distal_axis is not None:
        pd = normalize(proximal_to_distal_axis)
        if np.dot(normal, pd) < 0.0:
            normal = -normal

    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(bone_polydata)
    clipper.SetClipFunction(plane)
    clipper.GenerateClippedOutputOn()
    clipper.SetGenerateClipScalars(False)
    clipper.Update()

    # vtkClipPolyData passes the FULL input point set to both outputs (points not used by any
    # cell included). Averaging over poly.GetPoints() therefore gave identical values for both
    # halves, the comparison below always returned (a, b), and proximal/distal came out swapped.
    # Drop unused points first, so each half contains only its own points.
    def used_points_only(poly):
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(poly)
        cleaner.PointMergingOff()
        cleaner.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(cleaner.GetOutput())
        return out

    a = used_points_only(clipper.GetOutput())
    b = used_points_only(clipper.GetClippedOutput())

    from vtk.util.numpy_support import vtk_to_numpy

    def representative_signed_distance(poly):
        pts = poly.GetPoints()
        if pts is None or pts.GetNumberOfPoints() == 0:
            return 0.0
        # Mean signed distance of the piece's own points identifies which half-space it
        # occupies, independent of vtkClipPolyData's output convention. Vectorised (numpy).
        coords = vtk_to_numpy(pts.GetData()).astype(float)
        return float(np.mean((coords - origin) @ normal))

    da = representative_signed_distance(a)
    db = representative_signed_distance(b)
    # With normal oriented proximal->distal: negative side = proximal, positive side = distal.
    if da <= db:
        return a, b
    return b, a



def split_bone_three_regions(bone_polydata, proximal_origin, proximal_normal,
                             distal_origin, distal_normal, proximal_to_distal_axis):
    """Split a CWO bone into three disjoint anatomical regions.

    Returns ``(proximal_fragment, wedge_to_remove, distal_fragment)``.

    The authoritative proximal/distal direction is ``proximal_to_distal_axis``;
    therefore Markups plane normal signs cannot invert the result.  The wedge is
    explicitly the intersection of:

      * bone distal to the proximal cutting plane, and
      * bone proximal to the distal cutting plane.
    """
    # SEQUENTIAL (mutually exclusive) split: original -> proximal | remainder, then
    # remainder -> wedge | distal. Two independent clips of the original would assign the region
    # beyond the hinge line (proximal of the proximal AND distal of the distal plane) to BOTH
    # fragments whenever the hinge lies inside the bone. Guaranteed here:
    # proximal, wedge, distal pairwise disjoint and together the original bone.
    proximal_fragment, remainder = clip_fragments(
        bone_polydata, proximal_origin, proximal_normal, proximal_to_distal_axis)

    wedge_to_remove, distal_fragment = clip_fragments(
        remainder, distal_origin, distal_normal, proximal_to_distal_axis)

    return proximal_fragment, wedge_to_remove, distal_fragment

def cross_section_centroid(bone_polydata, plane_origin, plane_normal):
    '''
    The area-weighted centroid of the bone's cross-section where `plane` cuts through it - i.e.
    "the middle of the bone" at that exact level, computed from the actual segmented anatomy
    rather than eyeballed in the 3D view. Cuts the closed bone surface with the plane (vtkCutter),
    joins the resulting boundary into closed contours (vtkStripper), fills them
    (vtkContourTriangulator) and area-weights each triangle's centroid - this is the correct
    2D-in-3D area centroid, not just an average of boundary points (which would be biased toward
    however densely the surface mesh happens to be triangulated at the edge).

    Returns a numpy point, or None if the plane doesn't actually intersect the bone (e.g. it's
    been dragged outside the segmented volume) or the cut produced no fillable contour.
    '''
    plane = vtk.vtkPlane()
    plane.SetOrigin(*plane_origin)
    plane.SetNormal(*plane_normal)

    cutter = vtk.vtkCutter()
    cutter.SetInputData(bone_polydata)
    cutter.SetCutFunction(plane)
    cutter.Update()
    if cutter.GetOutput().GetNumberOfPoints() == 0:
        return None

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputConnection(stripper.GetOutputPort())
    triangulator.Update()
    filled = triangulator.GetOutput()
    if filled.GetNumberOfCells() == 0:
        return None

    points = filled.GetPoints()
    total_area = 0.0
    weighted_centroid = np.zeros(3)
    for i in range(filled.GetNumberOfCells()):
        cell = filled.GetCell(i)
        if cell.GetNumberOfPoints() != 3:
            continue
        p0 = np.array(points.GetPoint(cell.GetPointId(0)))
        p1 = np.array(points.GetPoint(cell.GetPointId(1)))
        p2 = np.array(points.GetPoint(cell.GetPointId(2)))
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        weighted_centroid += area * (p0 + p1 + p2) / 3.0
        total_area += area

    if total_area < 1e-9:
        return None
    return weighted_centroid / total_area


def planes_intersection_line(origin1, normal1, origin2, normal2):
    '''
    The line where two planes intersect - used to derive the hinge axis directly from two
    independently, manually positioned cutting planes (proximal and distal), instead of
    computing a hinge point separately and deriving a second plane from it. Any point on the
    returned line is an equally valid rotation pivot for aligning the two planes (see
    signed_angle_between): rotating a rigid body about a line leaves every point of that line
    fixed, so which point along the infinite line is picked doesn't affect the result - the point
    returned is simply the one on the line closest to the midpoint of the two plane origins, a
    reasonable, well-defined choice with no other significance.

    Returns (point_on_line, direction_unit) as numpy arrays, or (None, None) if the two planes
    are parallel (or nearly so) and therefore have no well-defined single intersection line - the
    caller should treat that as "the two planes don't yet define a wedge" and ask the user to
    tilt one of them.
    '''
    n1 = normalize(normal1)
    n2 = normalize(normal2)
    direction = np.cross(n1, n2)
    direction_length = np.linalg.norm(direction)
    if direction_length < 1e-6:
        return None, None
    direction = direction / direction_length

    origin1 = np.asarray(origin1, dtype=float)
    origin2 = np.asarray(origin2, dtype=float)
    d1 = np.dot(n1, origin1)
    d2 = np.dot(n2, origin2)
    midpoint = (origin1 + origin2) / 2.0

    # Three equations pin down a unique point: on plane 1, on plane 2, and (the arbitrary but
    # well-defined choice above) closest to the midpoint of the two origins along the line
    # direction.
    A = np.array([n1, n2, direction])
    b = np.array([d1, d2, np.dot(direction, midpoint)])
    point_on_line = np.linalg.solve(A, b)
    return point_on_line, direction


def signed_angle_between(v_from, v_to, axis):
    '''
    The signed angle in degrees to rotate v_from about `axis` (right-hand rule) so it becomes
    parallel to v_to - i.e. exactly the angle to pass to vtk's RotateWXYZ(angle, *axis). All
    three inputs are treated as directions (need not be pre-normalized); axis should be
    perpendicular to both v_from and v_to, as it is when axis is the intersection line of the two
    planes v_from/v_to are the normals of.

    This is how the wedge correction angle and its sign are read directly off two independently
    placed cutting planes - no separate "apex side" or "invert" choice needed, unlike the
    target-value-driven correction used for a single plane (OWO): the angle and its direction are
    simply whatever the two planes actually are.
    '''
    v_from = normalize(v_from)
    v_to = normalize(v_to)
    axis = normalize(axis)
    cos_angle = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
    sin_angle = float(np.dot(np.cross(v_from, v_to), axis))
    return float(np.degrees(np.arctan2(sin_angle, cos_angle)))
