# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""bone_export - the osteotomised (corrected) bone as ONE closed solid for STL export (no Slicer / Qt).

The fixed and the moved fragment are rasterised together; the gap between their cut faces (opening
wedge, reduction gap) is bridged loop by loop - every bone section of the fixed cut face is joined to
the nearest section of the moved cut face by their convex hull - so the result is one connected piece
(e.g. for pre-contouring a plate on a printed model). Coordinates stay those of the CT (mm)."""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def _matrix(H):
    m = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(H[r][c]))
    return m


def transformed(poly, H):
    transform = vtk.vtkTransform()
    transform.SetMatrix(_matrix(H))
    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputData(poly)
    filt.SetTransform(transform)
    filt.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(filt.GetOutput())
    return out


def cut_loops(poly, origin, normal, inset=0.4, min_points=8):
    """Closed section loops of `poly` in a plane `inset` mm inside the fragment (towards its centre),
    as a list of (N, 3) point arrays."""
    origin, normal = np.asarray(origin, float), np.asarray(normal, float)
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    centre = np.asarray(poly.GetCenter(), float)
    inward = normal if float(np.dot(centre - origin, normal)) >= 0 else -normal
    plane = vtk.vtkPlane()
    plane.SetOrigin(*(origin + inset * inward))
    plane.SetNormal(*normal)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(poly)
    cutter.SetCutFunction(plane)
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(cutter.GetOutputPort())
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    out = connectivity.GetOutput()
    if out.GetNumberOfPoints() == 0:
        return []
    points = vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    region = vtk_to_numpy(out.GetPointData().GetArray("RegionId"))
    return [points[region == r] for r in np.unique(region) if np.count_nonzero(region == r) >= min_points]


def _hull(points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=float), deep=True))
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(vtk_points)
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(cloud)
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(delaunay.GetOutputPort())
    surface.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(surface.GetOutput())
    return out


def bridges(loops_a, loops_b, max_distance=25.0):
    """Hull bridges joining every loop of face A with the nearest loop of face B (by centroid)."""
    result = []
    if not loops_a or not loops_b:
        return result
    centres_b = np.array([loop.mean(axis=0) for loop in loops_b])
    for loop in loops_a:
        distances = np.linalg.norm(centres_b - loop.mean(axis=0), axis=1)
        k = int(np.argmin(distances))
        if distances[k] <= max_distance:
            hull = _hull(np.vstack([loop, loops_b[k]]))
            if hull.GetNumberOfPoints():
                result.append(hull)
    return result


def _rasterised(poly, lo, voxel, dims):
    """Inside of a closed surface on the grid (k, j, i) as uint8 0/1."""
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(*lo)
    stencil.SetOutputSpacing(voxel, voxel, voxel)
    stencil.SetOutputWholeExtent(0, int(dims[0]) - 1, 0, int(dims[1]) - 1, 0, int(dims[2]) - 1)
    to_image = vtk.vtkImageStencilToImage()
    to_image.SetInputConnection(stencil.GetOutputPort())
    to_image.SetInsideValue(1)
    to_image.SetOutsideValue(0)
    to_image.SetOutputScalarTypeToUnsignedChar()
    to_image.Update()
    return vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars()).reshape(dims[::-1]).astype(np.uint8)


def solid_union(polys, voxel=0.35, max_voxels=160e6, smooth=True, exclude=()):
    """Union of closed surfaces as ONE iso-surface; the inside of the closed surfaces in exclude (other
    bones touching this one, e.g. the femur on the tibial plateau) is taken out again. Returns (polydata,
    info)."""
    bounds = np.array([p.GetBounds() for p in polys if p.GetNumberOfPoints()], float)
    lo = bounds[:, 0::2].min(axis=0) - 2.0
    hi = bounds[:, 1::2].max(axis=0) + 2.0
    while np.prod(np.ceil((hi - lo) / voxel) + 1) > max_voxels:
        voxel *= 1.25
    dims = (np.ceil((hi - lo) / voxel) + 1).astype(int)
    volume = np.zeros(dims[::-1], dtype=np.uint8)                 # (k, j, i)
    for poly in polys:
        if poly.GetNumberOfPoints():
            volume |= _rasterised(poly, lo, voxel, dims)
    for poly in exclude:
        if poly is not None and poly.GetNumberOfPoints():
            volume[_rasterised(poly, lo, voxel, dims) > 0] = 0
    image = vtk.vtkImageData()
    image.SetDimensions(*[int(v) for v in dims])
    image.SetOrigin(*lo)
    image.SetSpacing(voxel, voxel, voxel)
    image.GetPointData().SetScalars(numpy_to_vtk(volume.ravel().astype(np.float32), deep=True,
                                                 array_type=vtk.VTK_FLOAT))
    source = image
    if smooth:
        gauss = vtk.vtkImageGaussianSmooth()
        gauss.SetInputData(image)
        gauss.SetStandardDeviations(0.8, 0.8, 0.8)
        gauss.SetRadiusFactors(2.0, 2.0, 2.0)
        gauss.Update()
        source = gauss.GetOutput()
    iso = vtk.vtkFlyingEdges3D()
    iso.SetInputData(source)
    iso.SetValue(0, 0.5)
    iso.ComputeNormalsOff()
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(iso.GetOutputPort())
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    pieces = connectivity.GetNumberOfExtractedRegions()
    largest = vtk.vtkPolyDataConnectivityFilter()
    largest.SetInputConnection(iso.GetOutputPort())
    largest.SetExtractionModeToLargestRegion()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(largest.GetOutputPort())
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return out, {"voxel": float(voxel), "pieces": int(pieces)}


def _contact(poly, fragment, samples=3000):
    """How close the surface poly comes to the fragment: mean of the 5 % smallest point distances."""
    points = vtk_to_numpy(poly.GetPoints().GetData())
    if len(points) > samples:
        points = points[np.linspace(0, len(points) - 1, samples).astype(int)]
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(fragment)
    locator.BuildLocator()
    frag = vtk_to_numpy(fragment.GetPoints().GetData())
    d = np.array([np.linalg.norm(frag[locator.FindClosestPoint(p)] - p) for p in points])
    d.sort()
    return float(d[:max(1, len(d) // 20)].mean())


def corrected_solid(fixed_poly, moving_poly, H, fixed_cut, moving_cut, voxel=0.35, others=(), removals=()):
    """The corrected bone as one solid: fixed fragment + moved fragment (moving_poly moved by H) + bridges
    over the gap between their cut faces. fixed_cut / moving_cut: (origin, normal) of each fragment's cut
    face in ITS OWN (original) coordinates. others: closed surfaces of the OTHER bones in the original
    position (they share the segmentation, e.g. the femur on the tibia) - taken out where they are, and
    once more moved by H, whichever fragment they hang on."""
    moved = transformed(moving_poly, H)
    loops_fixed = cut_loops(fixed_poly, *fixed_cut)
    loops_moving = [((H[:3, :3] @ loop.T).T + H[:3, 3]) for loop in cut_loops(moving_poly, *moving_cut)]
    joints = bridges(loops_fixed, loops_moving)
    exclude = []
    for poly in others:
        if poly is None or not poly.GetNumberOfPoints():
            continue
        # the other bone hangs on the fragment it touches: closer to the moving one -> moves with it
        on_moving = _contact(poly, moving_poly) < _contact(poly, fixed_poly)
        exclude.append(transformed(poly, H) if on_moving else poly)
    # removals: closed bodies cut away from the bone (e.g. the sulcus disc of a trochlear prosthesis),
    # given in the original position - moved by H when they lie on the moving fragment
    for poly, on_moving in removals:
        if poly is not None and poly.GetNumberOfPoints():
            exclude.append(transformed(poly, H) if on_moving else poly)
    solid, info = solid_union([fixed_poly, moved] + joints, voxel=voxel, exclude=exclude)
    info.update(bridges=len(joints), loops=(len(loops_fixed), len(loops_moving)))
    return solid, info
