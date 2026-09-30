"""
guide_logic - surface (mesh) based helpers for patient-specific guide planning.

The footprint is a per-vertex selection on the bone SURFACE mesh (no voxel volume): it is
suggested automatically, edited with brush / lasso tools, and finally extruded into a closed
guide body (lifted off the bone by a fit gap, fixed wall thickness).
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def world_to_local_matrix(centre, shaft, across, d):
    """4x4 (vtk) mapping world -> local (t along shaft, u across, v towards the guide side)."""
    R = np.vstack([shaft, across, d])
    m = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            m.SetElement(r, c, float(R[r, c]))
        m.SetElement(r, 3, float(-R[r] @ np.asarray(centre, dtype=float)))
    return m


def extract_roi_mesh(bone_polydata, centre, shaft, across, d, t_range, u_half, v_half, closest_point=None):
    """Bone surface inside an oriented box around the osteotomy; largest connected piece; clean
    triangles with consistent point normals (orientation fixed later)."""
    transform = vtk.vtkTransform()
    transform.SetMatrix(world_to_local_matrix(centre, shaft, across, d))
    box = vtk.vtkBox()
    box.SetBounds(float(t_range[0]), float(t_range[1]), -float(u_half), float(u_half),
                  -float(v_half), float(v_half))
    box.SetTransform(transform)
    extract = vtk.vtkExtractPolyDataGeometry()
    extract.SetInputData(bone_polydata)
    extract.SetImplicitFunction(box)
    extract.ExtractInsideOn()
    extract.ExtractBoundaryCellsOn()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(extract.GetOutputPort())
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(triangles.GetOutputPort())
    if closest_point is not None:
        connectivity.SetExtractionModeToClosestPointRegion()
        connectivity.SetClosestPoint(*[float(c) for c in closest_point])
    else:
        connectivity.SetExtractionModeToLargestRegion()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(connectivity.GetOutputPort())
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return out


def points_and_normals(mesh):
    P = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    N = vtk_to_numpy(mesh.GetPointData().GetNormals()).astype(float)
    return P, N


def outermost(t, u, v, box, depth=1.0, bin_size=0.5):
    """Boolean mask: within box, vertices of the outermost layer along +v per (t, u) column."""
    result = np.zeros(len(t), dtype=bool)
    idx = np.nonzero(box)[0]
    if len(idx) == 0:
        return result
    ti = np.floor((t[idx] - t[idx].min()) / bin_size).astype(np.int64)
    ui = np.floor((u[idx] - u[idx].min()) / bin_size).astype(np.int64)
    key = ti * (int(ui.max()) + 2) + ui
    order = np.argsort(key)
    starts = np.r_[0, np.nonzero(np.diff(key[order]))[0] + 1]
    vmax_per = np.maximum.reduceat(v[idx][order], starts)
    vmax = np.empty(len(idx))
    vmax[order] = np.repeat(vmax_per, np.diff(np.r_[starts, len(order)]))
    result[idx[v[idx] >= vmax - depth]] = True
    return result


def orient_and_suggest(mesh, centre, shaft, across, d, t_min, t_max, half_width):
    """Orient the point normals OUTWARD (checked on the outer surface facing the guide side) and
    return the suggested footprint selection (bool per vertex)."""
    P, N = points_and_normals(mesh)
    rel = P - np.asarray(centre, dtype=float)
    t, u, v = rel @ shaft, rel @ across, rel @ d
    box = (t >= t_min) & (t <= t_max) & (np.abs(u) <= half_width) & (v > 0.0)
    outer = outermost(t, u, v, box)
    if np.any(outer) and float(np.mean(N[outer] @ d)) < 0.0:
        N = -N
        mesh.GetPointData().SetNormals(numpy_to_vtk(N, deep=True))
        mesh.GetPointData().GetNormals().SetName("Normals")
    selected = outer & ((N @ d) > 0.2)
    return selected


def set_selection(mesh, selected, name="Footprint"):
    array = numpy_to_vtk(np.asarray(selected, dtype=np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    array.SetName(name)
    mesh.GetPointData().RemoveArray(name)
    mesh.GetPointData().AddArray(array)


def get_selection(mesh, name="Footprint"):
    array = mesh.GetPointData().GetArray(name)
    if array is None:
        return np.zeros(mesh.GetNumberOfPoints(), dtype=bool)
    return vtk_to_numpy(array).astype(bool)


def triangles_of(mesh):
    return vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)


def patch_from_selection(mesh, selected, lift=0.0):
    """Triangles whose three vertices are selected, as a separate clean surface patch; points
    optionally lifted along the (outward) normals."""
    P, N = points_and_normals(mesh)
    tris = triangles_of(mesh)
    keep = np.asarray(selected, dtype=bool)[tris].all(axis=1)
    tris = tris[keep]
    patch = vtk.vtkPolyData()
    if len(tris) == 0:
        return patch
    used = np.unique(tris)
    remap = -np.ones(len(P), dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(P[used] + lift * N[used], deep=True))
    cells = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), remap[tris]]).ravel()
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(len(tris), numpy_to_vtkIdTypeArray(cells, deep=True))
    patch.SetPoints(points)
    patch.SetPolys(cell_array)
    normals = numpy_to_vtk(N[used], deep=True)
    normals.SetName("Normals")
    patch.GetPointData().SetNormals(normals)
    return patch


def points_in_polygon(xy, polygon):
    """Vectorised even-odd point-in-polygon test (display coordinates)."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    poly = np.asarray(polygon, dtype=float)
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    for x1, y1, x2, y2 in zip(px, py, qx, qy):
        crosses = ((y1 > y) != (y2 > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < x_at)
    return inside


def extrude_patch(patch, gap, thickness, smoothing_iterations=15):
    """Closed guide body from a surface patch: smoothed, lifted by `gap` along the outward normals
    (fit clearance to the bone), then extruded outward by `thickness` with capping."""
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(patch)
    smooth.SetNumberOfIterations(int(smoothing_iterations))
    smooth.SetPassBand(0.1)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.ComputePointNormalsOn()
    normals.Update()
    smoothed = vtk.vtkPolyData()
    smoothed.DeepCopy(normals.GetOutput())
    # keep the outward orientation of the original patch
    N_orig = vtk_to_numpy(patch.GetPointData().GetNormals())
    N_new = vtk_to_numpy(smoothed.GetPointData().GetNormals())
    if len(N_orig) == len(N_new) and float(np.mean(np.sum(N_orig * N_new, axis=1))) < 0.0:
        flip = vtk.vtkPolyDataNormals()
        flip.SetInputData(smoothed)
        flip.SplittingOff()
        flip.ConsistencyOn()
        flip.FlipNormalsOn()
        flip.Update()
        smoothed.DeepCopy(flip.GetOutput())
    smoothed.GetPointData().SetActiveVectors("Normals")
    lift = vtk.vtkWarpVector()
    lift.SetInputData(smoothed)
    lift.SetScaleFactor(float(gap))
    lift.Update()
    lifted = vtk.vtkPolyData()
    lifted.DeepCopy(lift.GetOutput())
    lifted.GetPointData().SetNormals(smoothed.GetPointData().GetNormals())
    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputData(lifted)
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetScaleFactor(float(thickness))
    extrude.CappingOn()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(extrude.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    final_normals = vtk.vtkPolyDataNormals()
    final_normals.SetInputConnection(clean.GetOutputPort())
    final_normals.SplittingOff()
    final_normals.ConsistencyOn()
    final_normals.AutoOrientNormalsOn()
    final_normals.Update()
    body = vtk.vtkPolyData()
    body.DeepCopy(final_normals.GetOutput())
    return body


def open_edge_count(polydata):
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(polydata)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfCells()


def outer_mask(mesh, centre, shaft, across, d, depth=1.0, bin_size=0.5):
    """Vertices of the OUTER bone surface on the guide side: outermost layer along +d per (t, u)
    column, with the (outward) normal facing the guide side. Excludes inner surfaces of pores /
    cancellous bone and the far side of the bone."""
    P, N = points_and_normals(mesh)
    rel = P - np.asarray(centre, dtype=float)
    t, u, v = rel @ shaft, rel @ across, rel @ d
    return outermost(t, u, v, np.ones(len(P), dtype=bool), depth=depth, bin_size=bin_size) & ((N @ d) > 0.0)


def clean_patch(patch, keep_fraction=0.05):
    """Drop small detached crumbs of a surface patch: keep connected regions with at least
    keep_fraction of the largest region's cells."""
    if patch.GetNumberOfCells() == 0:
        return patch
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(patch)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    sizes = vtk_to_numpy(connectivity.GetRegionSizes())
    if len(sizes) <= 1:
        return patch
    keep = np.nonzero(sizes >= keep_fraction * sizes.max())[0]
    select = vtk.vtkPolyDataConnectivityFilter()
    select.SetInputData(patch)
    select.SetExtractionModeToSpecifiedRegions()
    for region in keep:
        select.AddSpecifiedRegion(int(region))
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(select.GetOutputPort())
    clean.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(clean.GetOutput())
    return out
