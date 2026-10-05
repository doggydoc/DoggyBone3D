# Doggybone 3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

import numpy as np
import math

def vector_with_two_points(i,j):

    return (j-i)

def normalvector_of_three_points(k,l,m):

    return np.cross((l-k),(l-m))
def normalvector_of_two_vectors(plane_vector_1, plane_vector_2):

    return np.cross(plane_vector_1,plane_vector_2)

def project_vector_to_plane_from_normal(normal, vector):
    if np.linalg.norm(normal) < 1e-9:
        raise ValueError("degenerate plane normal (landmarks coincide or are collinear)")
    normal_component = np.dot(normal, vector)/(np.linalg.norm(normal)**2) * normal
    return (vector - normal_component) # in-plane component

def project_vector_to_plane_from_2_vectors(plane_vector_1, plane_vector_2, vector):
    normal = normalvector_of_two_vectors(plane_vector_1,plane_vector_2)
    return project_vector_to_plane_from_normal(normal, vector)


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

def angle_in_plane_with_normal(normal, vector_a, vector_b):

    vec_a_proj = project_vector_to_plane_from_normal(normal, vector_a)
    vec_b_proj = project_vector_to_plane_from_normal(normal, vector_b)

    return (angle(vec_a_proj, vec_b_proj)*180/math.pi)

def angle_in_plane_from_two_vectors(plane_vector_1, plane_vector_2, vector_a, vector_b):
    vec_a_proj = project_vector_to_plane_from_2_vectors(plane_vector_1, plane_vector_2, vector_a)
    vec_b_proj = project_vector_to_plane_from_2_vectors(plane_vector_1, plane_vector_2, vector_b)
    normal=np.cross(plane_vector_1,plane_vector_2)

    return (angle(vec_a_proj, vec_b_proj)*180/math.pi)

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