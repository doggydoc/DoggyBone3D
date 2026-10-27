# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

import numpy as np
import math

def vector_with_two_points(i,j):

    return (j-i)


def project_vector_to_plane_from_normal(normal, vector):
    if np.linalg.norm(normal) < 1e-9:
        raise ValueError("degenerate plane normal (landmarks coincide or are collinear)")
    normal_component = np.dot(normal, vector)/(np.linalg.norm(normal)**2) * normal
    return (vector - normal_component) # in-plane component


def angle(u,v):

    c = np.dot(u,v)
    d = np.linalg.norm(u)
    e = np.linalg.norm(v)
    if d < 1e-6 or e < 1e-6:
        # two landmarks at the same place / an axis parallel to the projection plane normal:
        # a NaN would silently show as a wrong side label
        raise ValueError("angle undefined - two landmarks coincide or an axis has no length in this plane")
    cos_angle = np.clip(c/(d*e), -1.0, 1.0)
    return math.acos(cos_angle)


def fit_sphere(points):
    '''
    Fits a sphere to a set of at least 4 non-coplanar 3D points and returns the fitted
    (center, radius). Averaging several points sampled on a curved anatomical surface this way
    is more robust against a single mis-placed landmark than relying on one manually chosen
    point (e.g. femur head, femoral condyles), and the fitted sphere can also be drawn in the
    scene so the user can see how well it matches the bone surface.

    Uses the standard closed-form ALGEBRAIC least-squares sphere fit (a single linear solve, no
    external dependency or iteration) rather than a nonlinear geometric fit: expanding
    (x-a)^2+(y-b)^2+(z-c)^2=r^2 gives x^2+y^2+z^2 = 2ax+2by+2cz+(r^2-a^2-b^2-c^2), which is LINEAR
    in the unknowns (a,b,c,d=r^2-a^2-b^2-c^2), so np.linalg.lstsq solves it directly. This
    algebraic fit is a good approximation for the point spreads used here (a handful of points
    over less than half a sphere), and needs no external dependency or initial guess/iteration
    the nonlinear version would.
    '''
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        raise ValueError("fit_sphere needs at least 4 non-coplanar points.")

    A = np.hstack([2.0 * points, np.ones((points.shape[0], 1))])
    b = np.sum(points**2, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 4:
        raise ValueError("fit_sphere: points are coplanar or otherwise degenerate.")
    center = solution[:3]
    radius_squared = solution[3] + float(np.dot(center, center))
    if radius_squared <= 0:
        raise ValueError("fit_sphere: degenerate point configuration (non-positive radius^2).")
    return center, float(np.sqrt(radius_squared))

def fit_sphere_rms(points):
    '''
    fit_sphere plus the fit quality: (center, radius, rms) - rms is the root mean square distance of
    the points from the fitted sphere SURFACE (mm). Small (< ~0.5 mm) = the points lie well on one
    sphere; large = a point is off the joint surface or the surface is not spherical there.
    '''
    points = np.asarray(points, dtype=float)
    center, radius = fit_sphere(points)
    distances = np.linalg.norm(points - center, axis=1) - radius
    return center, radius, float(np.sqrt(np.mean(distances ** 2)))


def project_to_plane(vector, plane_normal):
    '''Component of `vector` in the plane with the given normal (normal need not be unit length).'''
    normal = np.asarray(plane_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    vector = np.asarray(vector, dtype=float)
    return vector - np.dot(vector, normal) * normal


def fit_circle_3d(points):
    '''
    Circle through >= 3 points in 3D (they need not lie exactly in one plane): best plane through
    the points (least squares), points projected into it, algebraic least-squares circle fit there.
    Returns (centre, radius, rms, plane_normal); rms = RMS distance of the points from the circle
    (in the plane) - the fit quality (e.g. the centre of the best-fitting circle of a condyle in the sagittal plane).
    '''
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("fit_circle_3d needs at least 3 points.")
    centroid = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centroid)
    e1, e2, normal = vt[0], vt[1], vt[2]
    xy = np.stack([(points - centroid) @ e1, (points - centroid) @ e2], axis=1)
    A = np.hstack([2.0 * xy, np.ones((len(xy), 1))])
    b = np.sum(xy ** 2, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        raise ValueError("fit_circle_3d: points are collinear.")
    cx, cy = solution[0], solution[1]
    radius_squared = solution[2] + cx * cx + cy * cy
    if radius_squared <= 0:
        raise ValueError("fit_circle_3d: degenerate point configuration.")
    radius = float(np.sqrt(radius_squared))
    rms = float(np.sqrt(np.mean((np.linalg.norm(xy - [cx, cy], axis=1) - radius) ** 2)))
    return centroid + cx * e1 + cy * e2, radius, rms, normal
