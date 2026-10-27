# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""Cross-section centre of a long bone - pure numpy part (no VTK, testable without Slicer).

The bone surface is cut with a plane perpendicular to the shaft axis (VTK, in the widget); the cut
gives closed contour loops - the bone itself, neighbouring bones and, in a
hollow segmentation, the medullary canal. pick_loop_centroid() chooses the bone the landmark sits on
and returns the AREA centroid of its OUTER contour (filled: the medullary canal does not shift it)."""

import numpy as np


def plane_basis(normal):
    """Two unit vectors (u, v) spanning the plane perpendicular to `normal`."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def to_plane(points, origin, u, v):
    d = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    return np.stack([d @ u, d @ v], axis=-1)


def polygon_area_centroid(loop_2d):
    """(signed area, centroid) of a closed polygon (first point need not be repeated)."""
    p = np.asarray(loop_2d, dtype=float)
    if len(p) < 3:
        return 0.0, p.mean(axis=0) if len(p) else np.zeros(2)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * cross.sum()
    if abs(area) < 1e-12:
        return 0.0, p.mean(axis=0)
    cx = ((x + xn) * cross).sum() / (6.0 * area)
    cy = ((y + yn) * cross).sum() / (6.0 * area)
    return float(area), np.array([cx, cy])


def point_in_polygon(point, loop_2d):
    """Even-odd rule."""
    x, y = float(point[0]), float(point[1])
    p = np.asarray(loop_2d, dtype=float)
    inside = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi:
            inside = not inside
        j = i
    return inside


def _distance_to_loop(point, loop_2d):
    p = np.asarray(loop_2d, dtype=float)
    a, b = p, np.roll(p, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", np.asarray(point) - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    closest = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(closest - np.asarray(point), axis=1)))


def pick_loop_centroid(loops_2d, seed_2d, max_distance=5.0):
    """Index of the chosen loop and the area centroid of the bone the seed belongs to.

    The bone = the LARGEST loop that contains the seed (outer contour - an inner medullary loop is
    smaller); if no loop contains it (the landmark lies on the surface or just outside), the loop
    nearest to the seed within max_distance mm. Returns (index, centroid_2d, area) or (None, None, 0)."""
    loops = [np.asarray(loop, dtype=float) for loop in loops_2d if len(loop) >= 3]
    if not loops:
        return None, None, 0.0
    stats = [polygon_area_centroid(loop) for loop in loops]
    containing = [i for i, loop in enumerate(loops) if point_in_polygon(seed_2d, loop)]
    if containing:
        index = max(containing, key=lambda i: abs(stats[i][0]))
    else:
        distances = [_distance_to_loop(seed_2d, loop) for loop in loops]
        index = int(np.argmin(distances))
        if distances[index] > max_distance:
            return None, None, 0.0
        # the outer contour of that bone: the largest loop containing the nearest loop's centroid
        around = [i for i, loop in enumerate(loops) if point_in_polygon(stats[index][1], loop)]
        if around:
            index = max(around, key=lambda i: abs(stats[i][0]))
    return index, stats[index][1], abs(stats[index][0])
