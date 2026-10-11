# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""
guide_logic - geometry for patient-specific guide planning (no Slicer / Qt imports).

The footprint is a per-vertex selection on the bone SURFACE mesh, edited with brush / lasso.
The guide body is built from the distance field of the bone (sdf_guide_body): exact fit gap,
smoothed outer surface, saw block(s), pin sleeves and bores, kerfs kept free. Also here: saw
slot planning, saw blocks, pin planning, the reduction pin bar, labels and guide clean-up.
"""
import time
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
    outer_region = None
    if closest_point is not None:
        outer_region = _outer_region_id(triangles, centre, shaft, across, d)
    if outer_region is not None:
        # Outer (periosteal) surface: never the wall of the medullary canal, even when the
        # plane centre lies inside a hollow segmentation.
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.InitializeSpecifiedRegionList()
        connectivity.AddSpecifiedRegion(int(outer_region))
    elif closest_point is not None:
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


def _outer_region_id(triangles, centre, shaft, across, d, rays=24, levels=(-12.0, -6.0, 0.0, 6.0, 12.0),
                     reach=120.0):
    """Connected region of the ROI surface that is the OUTER (periosteal) surface: rays shot from
    outside towards the shaft axis, at several heights and from all around, hit the outer surface
    FIRST - whatever the width of the medullary canal (which widens towards the metaphysis and can
    have a larger mean radius than the diaphyseal cortex). Majority vote over all rays.
    None if no ray hits anything."""
    triangles.Update()
    finder = vtk.vtkPolyDataConnectivityFilter()
    finder.SetInputData(triangles.GetOutput())
    finder.SetExtractionModeToAllRegions()
    finder.ColorRegionsOn()
    finder.Update()
    out = finder.GetOutput()
    if out.GetNumberOfCells() == 0:
        return None
    point_ids = out.GetPointData().GetArray("RegionId")     # the filter colours POINTS
    if point_ids is None:
        return None
    region_of_point = vtk_to_numpy(point_ids).astype(int)

    def region_of_cell_id(cell):
        return int(region_of_point[out.GetCell(int(cell)).GetPointId(0)])
    obb = vtk.vtkOBBTree()
    obb.SetDataSet(out)
    obb.BuildLocator()
    centre = np.asarray(centre, dtype=float)
    shaft, across, d = (unit(np.asarray(v, dtype=float)) for v in (shaft, across, d))
    votes = {}
    hit_points = vtk.vtkPoints()
    hit_cells = vtk.vtkIdList()
    for level in levels:
        axis_point = centre + float(level) * shaft
        for k in range(int(rays)):
            angle = 2.0 * np.pi * k / int(rays)
            direction = np.cos(angle) * d + np.sin(angle) * across
            start = axis_point + reach * direction
            hit_points.Reset()
            hit_cells.Reset()
            obb.IntersectWithLine(start.tolist(), axis_point.tolist(), hit_points, hit_cells)
            if hit_cells.GetNumberOfIds() == 0:
                continue
            # first hit seen from outside = closest to the ray start
            best = min(range(hit_points.GetNumberOfPoints()),
                       key=lambda n: float(np.linalg.norm(np.asarray(hit_points.GetPoint(n)) - start)))
            region = region_of_cell_id(hit_cells.GetId(best))
            votes[region] = votes.get(region, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)

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


def patch_from_selection(mesh, selected, lift=0.0, return_ids=False):
    """Triangles whose three vertices are selected, as a separate clean surface patch; points
    optionally lifted along the (outward) normals."""
    P, N = points_and_normals(mesh)
    tris = triangles_of(mesh)
    keep = np.asarray(selected, dtype=bool)[tris].all(axis=1)
    tris = tris[keep]
    patch = vtk.vtkPolyData()
    if len(tris) == 0:
        return (patch, np.zeros(0, dtype=np.int64)) if return_ids else patch
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
    if return_ids:
        return patch, used
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


def close_selection(mesh, selected, iterations=2):
    """Morphological closing of a vertex selection on the mesh (dilate, then erode, by mesh
    neighbourhood): fills pinholes and smooths a ragged painted border (~0.5 mm per iteration)."""
    tris = triangles_of(mesh)
    sel = np.asarray(selected, dtype=bool).copy()

    def dilate(s):
        hit = s[tris].any(axis=1)
        out = s.copy()
        out[tris[hit].ravel()] = True
        return out

    for _ in range(iterations):
        sel = dilate(sel)
    for _ in range(iterations):
        sel = ~dilate(~sel)
    return sel


def _boundary_edge_array(poly):
    tris = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    edges = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return tris, uniq[counts == 1], uniq[counts > 2]


def make_patch_manifold(patch, hole_size=2.0, max_iterations=15):
    """Surface patch that extrudes into a CLOSED body: fill small holes, then repeatedly drop the
    triangles around 'bow-tie' vertices (vertices where more than two border edges meet) and
    around non-manifold edges, then drop small detached pieces."""
    if patch.GetNumberOfCells() == 0:
        return patch
    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(patch)
    fill.SetHoleSize(float(hole_size))
    fill.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(fill.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    work = vtk.vtkPolyData()
    work.DeepCopy(clean.GetOutput())
    for _ in range(max_iterations):
        tris, border, nonmanifold = _boundary_edge_array(work)
        if len(tris) == 0:
            break
        degree = np.bincount(border.ravel(), minlength=work.GetNumberOfPoints()) if len(border) else np.zeros(work.GetNumberOfPoints(), int)
        bad = degree > 2
        if len(nonmanifold):
            bad[nonmanifold.ravel()] = True
        if not bad.any():
            break
        keep = ~bad[tris].any(axis=1)
        cells = np.hstack([np.full((int(keep.sum()), 1), 3, dtype=np.int64), tris[keep]]).ravel()
        array = vtk.vtkCellArray()
        array.SetCells(int(keep.sum()), numpy_to_vtkIdTypeArray(cells, deep=True))
        nxt = vtk.vtkPolyData()
        nxt.SetPoints(work.GetPoints())
        nxt.SetPolys(array)
        nxt.GetPointData().ShallowCopy(work.GetPointData())
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(nxt)
        cl.Update()
        work = vtk.vtkPolyData()
        work.DeepCopy(cl.GetOutput())
    return clean_patch(work)


def external_mask(mesh, labelmap, label_value, margin=4.0, seal_iterations=3, sample_distances=(1.0, 2.0)):
    """Per-vertex: True where the bone surface faces the OUTSIDE of the bone.

    Voxel test on the segmentation around the mesh: the bone is closed morphologically (seals
    small cortical gaps), the bone cross-section is sealed on the faces of the region of interest
    (so the medullary canal cut open by the region's end faces does not count as outside), and
    the outside is the background connected to the region's border. A vertex is external if a
    point a little outside along its (outward) normal lies in that outside space. Surfaces of
    pores, cancellous bone and the medullary canal are enclosed and therefore not external."""
    import scipy.ndimage as ndi
    P, N = points_and_normals(mesh)
    m = vtk.vtkMatrix4x4()
    labelmap.GetImageToWorldMatrix(m)
    M = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])
    Minv = np.linalg.inv(M)
    ext = labelmap.GetExtent()
    b = [0.0] * 6
    mesh.GetBounds(b)
    corners = np.array([[x, y, z, 1.0] for x in (b[0] - margin, b[1] + margin)
                        for y in (b[2] - margin, b[3] + margin) for z in (b[4] - margin, b[5] + margin)])
    ijk = (Minv @ corners.T).T[:, :3]
    lo = np.maximum(np.floor(ijk.min(axis=0)).astype(int), [ext[0], ext[2], ext[4]])
    hi = np.minimum(np.ceil(ijk.max(axis=0)).astype(int) + 1, [ext[1] + 1, ext[3] + 1, ext[5] + 1])
    dims = (ext[5] - ext[4] + 1, ext[3] - ext[2] + 1, ext[1] - ext[0] + 1)
    full = vtk_to_numpy(labelmap.GetPointData().GetScalars()).reshape(dims)
    bone = full[lo[2] - ext[4]:hi[2] - ext[4], lo[1] - ext[2]:hi[1] - ext[2], lo[0] - ext[0]:hi[0] - ext[0]] == label_value
    solid = ndi.binary_closing(bone, iterations=int(seal_iterations), border_value=0) | bone
    # porous / thin cortex (low threshold): the medullary canal and the pores may still connect to the
    # outside in 3D - fill every enclosed hole slice by slice in all three directions, then in 3D
    for axis in range(3):
        for index in range(solid.shape[axis]):
            sl = [slice(None)] * 3
            sl[axis] = index
            face = solid[tuple(sl)]
            if face.any():
                solid[tuple(sl)] = ndi.binary_fill_holes(face)
    solid = ndi.binary_fill_holes(solid)
    # seal the bone cross-sections on the six faces of the region of interest
    for axis in range(3):
        for index in (0, solid.shape[axis] - 1):
            face = np.take(solid, index, axis=axis)
            filled = ndi.binary_fill_holes(face)
            sl = [slice(None)] * 3
            sl[axis] = index
            solid[tuple(sl)] = filled
    background = ~solid
    labels, _ = ndi.label(background)
    border_labels = np.unique(np.concatenate([
        labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    border_labels = border_labels[border_labels > 0]
    outside = np.isin(labels, border_labels)
    result = np.zeros(len(P), dtype=bool)
    shape = np.array(outside.shape)          # (k, j, i)
    # both sides of the surface are tested (independent of the normals' orientation): a vertex is
    # external if the outside space lies just in front of OR just behind it
    for s in tuple(sample_distances) + tuple(-float(x) for x in sample_distances):
        q = np.c_[P + float(s) * N, np.ones(len(P))]
        idx = np.round((Minv @ q.T).T[:, :3]).astype(int) - lo      # (i, j, k) local
        kji = idx[:, ::-1]
        ok = np.all((kji >= 0) & (kji < shape), axis=1)
        hit = np.zeros(len(P), dtype=bool)
        hit[ok] = outside[kji[ok, 0], kji[ok, 1], kji[ok, 2]]
        result |= hit
    return result


def smooth_selection_rim(mesh, selected, amount_mm, edge_length=0.4):
    """Smooth the outline of a vertex selection: closing (fills notches) then opening (removes
    spikes and thin fringes), radius ~amount_mm in mesh rings."""
    rings = int(round(float(amount_mm) / float(edge_length)))
    if rings <= 0:
        return np.asarray(selected, dtype=bool)
    tris = triangles_of(mesh)
    sel = np.asarray(selected, dtype=bool).copy()

    def dilate(s):
        hit = s[tris].any(axis=1)
        out = s.copy()
        out[tris[hit].ravel()] = True
        return out

    def erode(s):
        return ~dilate(~s)

    for _ in range(rings):
        sel = dilate(sel)
    for _ in range(rings):
        sel = erode(sel)
    for _ in range(rings):
        sel = erode(sel)
    for _ in range(rings):
        sel = dilate(sel)
    return sel


# --- saw slots -----------------------------------------------------------------------------------
# One fixed rule, no options:
#  * slot thickness = blade 1.0 mm + 0.2 mm clearance
#  * two cuts (wedge): the whole kerf lies inside the wedge - the slot wall facing the kept segment
#    is flush with the virtual cutting plane, so both kept segments are exactly as planned.
#    One cut: the slot is centred on the plane.
#  * slot length = bone width in the cut (perpendicular to the saw direction) + margin each side
#  * a rectangular saw block (two plates, top flat) stands on the guide around every slot; its
#    underside follows the bone (+ fit gap); the slot ends are closed by thin caps at the top -
#    these caps are the break-off bridges of the one-piece guide.
# The guide body gets an exactly planar slot (patch clipped with the slot planes, extrusion inside
# the rail zone parallel to the plane). Block parts are closed shells that overlap the body; the
# print preparation software unites overlapping shells of one STL.

BLADE_THICKNESS = 1.0
SLOT_CLEARANCE = 0.2
SLOT_WIDTH = BLADE_THICKNESS + SLOT_CLEARANCE
SLOT_WALL = 2.5          # plate thickness on each side of the slot (grows with the corner bridges, see configure_slots)
SLOT_MARGIN = 1.0        # slot longer than the bone on each side (fixed; then the end caps)
BLOCK_HEIGHT = 10.0      # top of the saw block above the highest bone point under it
BLOCK_MAX_DEPTH = 22.0   # block never reaches further down than this below its top
BRIDGE_WIDTH = 3.0       # closed slot ends / corner bridges along the slot (UI: Corner bridges)
BLOCK_WALL = 3.5         # block guide (open face instead of a slot): wall thickness behind the cutting face
DEFAULT_BRIDGE_WIDTH = BRIDGE_WIDTH             # at this corner bridge size the walls have their base thickness
BASE_SLOT_WALL, BASE_BLOCK_WALL = SLOT_WALL, BLOCK_WALL
BREAK_BRIDGE_LENGTH = 2.5   # saw block pair: corner bridge at each END of each kerf, along the kerf (mm)
BREAK_BRIDGE_HEIGHT = 2.5   # ... and its height below the top of the block (mm) - snaps off with pliers
                            # both = BRIDGE_WIDTH - 0.5 (the kerf is kept free 0.5 mm beyond the slot end)
BRIDGE_HEIGHT = 3.0          # minimum height of a bridge (mm)
BRIDGE_FRACTION = 0.30       # bridges close the top 30 % of the local block height; below open to the bone


def configure_slots(blade_thickness=None, block_height=None, bridge_size=None):
    """Set blade thickness (slot = blade + clearance), saw block height and the size of the corner
    bridges (end caps of the slots, mm along the slot; the block grows sideways by the same amount).
    The corner bridge size also sets the width of the whole frame: every mm above the default adds
    1 mm to the wall on each side of the slot (block guide: behind the cutting face); below the
    default the walls keep their base thickness."""
    global BLADE_THICKNESS, SLOT_WIDTH, BLOCK_HEIGHT, BRIDGE_WIDTH, BREAK_BRIDGE_LENGTH, BREAK_BRIDGE_HEIGHT
    global SLOT_WALL, BLOCK_WALL
    if bridge_size is not None:
        BRIDGE_WIDTH = max(1.5, float(bridge_size))
        BREAK_BRIDGE_LENGTH = BREAK_BRIDGE_HEIGHT = BRIDGE_WIDTH - 0.5
        extra = max(0.0, BRIDGE_WIDTH - DEFAULT_BRIDGE_WIDTH)
        SLOT_WALL = BASE_SLOT_WALL + extra
        BLOCK_WALL = BASE_BLOCK_WALL + extra
    if blade_thickness is not None:
        BLADE_THICKNESS = float(blade_thickness)
        SLOT_WIDTH = BLADE_THICKNESS + SLOT_CLEARANCE
    if block_height is not None:
        BLOCK_HEIGHT = float(block_height)
PIN_THRESHOLD_MM = 15.0


def auto_pin_diameter(bone_diameter):
    """Clinic rule: bones up to 15 mm -> 1.2 mm pins, larger bones -> 1.6 mm pins."""
    if bone_diameter is None:
        return 1.6
    return 1.2 if bone_diameter <= PIN_THRESHOLD_MM else 1.6


def section_contour(bone_polydata, origin, normal, search_radius=40.0):
    """Largest closed cross-section contour of the bone in the plane near origin (N x 3) or None."""
    o = np.asarray(origin, float)
    nn = unit(np.asarray(normal, float))
    plane = vtk.vtkPlane()
    plane.SetOrigin(*o)
    plane.SetNormal(*nn)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(bone_polydata)
    cutter.SetCutFunction(plane)
    strip = vtk.vtkStripper()
    strip.SetInputConnection(cutter.GetOutputPort())
    strip.Update()
    pd = strip.GetOutput()
    if pd.GetNumberOfPoints() < 3:
        return None
    P = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    e1 = unit(np.cross(nn, [1.0, 0.0, 0.0]) if abs(nn[0]) < 0.9 else np.cross(nn, [0.0, 1.0, 0.0]))
    e2 = np.cross(nn, e1)
    best, best_area = None, 0.0
    lines = pd.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    while lines.GetNextCell(ids):
        idx = [ids.GetId(k) for k in range(ids.GetNumberOfIds())]
        if len(idx) < 4:
            continue
        Q = P[idx]
        if np.min(np.linalg.norm(Q - o, axis=1)) > search_radius:
            continue
        x, y = (Q - o) @ e1, (Q - o) @ e2
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if area > best_area:
            best, best_area = Q, area
    return best


def contour_diameter(contour, normal):
    """Mean of the two principal widths of a section contour (mm)."""
    if contour is None:
        return None
    nn = unit(np.asarray(normal, float))
    Q = contour - contour.mean(axis=0)
    Q = Q - np.outer(Q @ nn, nn)
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    return 0.5 * (float(np.ptp(Q @ vt[0])) + float(np.ptp(Q @ vt[1])))


def plan_slot(bone_points, contour, origin, normal, other_plane, v_hint, gap, adjust=None, margin=None,
              near_fraction=None):
    """Geometry of one slot + saw block (dict) or {'error': ...}.
    Frame: C = slot centre, n = plane normal, v = saw direction (bone -> guide), u = n x v."""
    o = np.asarray(origin, float)
    nn = unit(np.asarray(normal, float))
    if contour is None:
        return {"error": "the cutting plane does not cut the bone"}
    c = contour.mean(axis=0)
    # area centroid of the section (contour points may be unevenly spaced - their mean is biased)
    e1 = unit(np.cross(nn, [1.0, 0.0, 0.0]) if abs(nn[0]) < 0.9 else np.cross(nn, [0.0, 1.0, 0.0]))
    e2 = np.cross(nn, e1)
    x, y = (contour - c) @ e1, (contour - c) @ e2
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * float(cross.sum())
    if abs(area) > 1e-6:
        cx = float(((x + np.roll(x, -1)) * cross).sum()) / (6.0 * area)
        cy = float(((y + np.roll(y, -1)) * cross).sum()) / (6.0 * area)
        c = c + cx * e1 + cy * e2
    c = c - float((c - o) @ nn) * nn                      # on the virtual plane
    v = np.asarray(v_hint, float)
    v = v - float(v @ nn) * nn
    if np.linalg.norm(v) < 1e-6:
        return {"error": "no saw direction (paint the footprint across the cut)"}
    v = unit(v)
    # manual adjustment of the saw box with its gizmo (always IN the cutting plane): rotation of the saw
    # direction about the plane normal, extension of the box along the kerf, height above the bone
    adjust = adjust or {}
    theta = float(adjust.get("theta", 0.0))
    if abs(theta) > 1e-9:
        u0_dir = unit(np.cross(nn, v))
        v = unit(np.cos(theta) * v + np.sin(theta) * u0_dir)
    u = unit(np.cross(nn, v))
    side = 0.0
    if other_plane is not None:
        o2 = np.asarray(other_plane[0], float)
        n2 = unit(np.asarray(other_plane[1], float))
        q = c - float((c - o2) @ n2) * n2                 # closest point on the other plane
        d = float((q - c) @ nn)
        if abs(d) > 0.05:
            side = float(np.sign(d))                       # wedge lies on this side of the plane
    half = 0.5 * SLOT_WIDTH
    C = c + side * half * nn
    rel_contour = contour - C
    us = rel_contour @ u
    # slot length: the whole cross-section (default) or only its part on the guide side
    # (near_fraction of the section depth, measured from the cortex the guide sits on)
    if near_fraction is not None and 0.0 < float(near_fraction) < 1.0:
        vs = rel_contour @ v
        near = vs >= float(vs.max()) - float(near_fraction) * float(vs.max() - vs.min())
        if near.sum() >= 2:
            us = us[near]
    m = SLOT_MARGIN if margin is None else float(margin)
    u0, u1 = float(us.min()) - m, float(us.max()) + m
    du = float(adjust.get("du", 0.0))            # the box is extended in the dragged direction, never shortened
    u0, u1 = min(u0, u0 + du), max(u1, u1 + du)
    ua, ub = u0 - BRIDGE_WIDTH, u1 + BRIDGE_WIDTH
    slab = half + SLOT_WALL + 0.3
    R = bone_points - C
    rn, ru, rv = R @ nn, R @ u, R @ v
    near = (np.abs(rn) <= slab) & (ru >= ua - 1.0) & (ru <= ub + 1.0)
    ru, rv = ru[near], rv[near]
    if len(rv) == 0:
        return {"error": "no bone under the saw block"}
    in_slot = (ru >= u0) & (ru <= u1)
    v_apex = float(rv[in_slot].max()) if in_slot.any() else float(rv.max())
    v_top = v_apex + gap + max(3.0, BLOCK_HEIGHT + float(adjust.get("dv", 0.0)))
    step = 0.25
    ubins = np.arange(ua, ub + 0.5 * step, step)
    ubins[-1] = ub
    bottom = np.full(len(ubins), -np.inf)
    order = np.argsort(ru)
    ru_s, rv_s = ru[order], rv[order]
    lo = np.searchsorted(ru_s, ubins - 0.5)
    hi = np.searchsorted(ru_s, ubins + 0.5)
    for k in range(len(ubins)):
        if hi[k] > lo[k]:
            bottom[k] = rv_s[lo[k]:hi[k]].max() + gap
    bottom = np.clip(bottom, v_top - BLOCK_MAX_DEPTH, v_top - 2.0)
    return dict(origin=o, normal=nn, centre=C, u=u, v=v, side=side, u0=u0, u1=u1, ua=ua, ub=ub,
                v_top=v_top, ubins=ubins, bottom=bottom, length=u1 - u0,
                bone_width=float(us.max() - us.min()), width=SLOT_WIDTH)


def _frame_points(slot, uu, vv, nv):
    C, u, v, nn = slot["centre"], slot["u"], slot["v"], slot["normal"]
    return C + np.outer(uu, u) + np.outer(vv, v) + np.outer(nv, nn)


def _extrude_sheet(points, triangles, vector):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=float), deep=True))
    poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    for t in triangles:
        cells.InsertNextCell(3)
        for k in t:
            cells.InsertCellPoint(int(k))
    poly.SetPolys(cells)
    ext = vtk.vtkLinearExtrusionFilter()
    ext.SetInputData(poly)
    ext.SetExtrusionTypeToVectorExtrusion()
    ext.SetVector(*[float(x) for x in vector])
    ext.SetScaleFactor(1.0)
    ext.CappingOn()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(ext.GetOutputPort())
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputConnection(tri.GetOutputPort())
    nrm.SplittingOff()
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    return out


def _clamp_far_layer(shell, n_base, direction, other):
    """Extruded shell: the far layer (points n..2n-1) must stop at the face of the OTHER slot so a
    wedge-side part never reaches into the neighbouring slot (thin end of the wedge)."""
    pts = vtk_to_numpy(shell.GetPoints().GetData()).astype(float).copy()
    m = n_base
    if len(pts) < 2 * m:
        return
    base, far = pts[:m], pts[m:2 * m]
    C2, n2 = other["centre"], other["normal"]
    half = 0.5 * SLOT_WIDTH
    d = np.asarray(direction, float)
    length = np.linalg.norm(far - base, axis=1)
    denom = float(d @ n2)
    if abs(denom) < 1e-6:
        return
    s_base = (base - C2) @ n2
    face = np.where(s_base >= 0.0, half, -half)
    t_hit = (face - s_base) / denom
    t_hit = np.where(t_hit > 0.0, t_hit, np.inf)
    t = np.minimum(length, np.maximum(t_hit, 0.05))
    pts[m:2 * m] = base + t[:, None] * d
    shell.GetPoints().SetData(numpy_to_vtk(pts, deep=True))
    shell.GetPoints().Modified()


def _extrude_sheet_raw(points, triangles, vector):
    """Like _extrude_sheet but returns the raw extrusion (input points first, copies after)."""
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=float), deep=True))
    poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    for t in triangles:
        cells.InsertNextCell(3)
        for k in t:
            cells.InsertCellPoint(int(k))
    poly.SetPolys(cells)
    ext = vtk.vtkLinearExtrusionFilter()
    ext.SetInputData(poly)
    ext.SetExtrusionTypeToVectorExtrusion()
    ext.SetVector(*[float(x) for x in vector])
    ext.SetScaleFactor(1.0)
    ext.CappingOn()
    ext.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(ext.GetOutput())
    return out


def _finish_shell(raw):
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(raw)
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputConnection(tri.GetOutputPort())
    nrm.SplittingOff()
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    return out


def saw_block_shells(slot, other=None, block_side=None):
    """Closed shells of the saw block: two plates (underside follows the bone) and two end caps.
    Every part is built from the slot face OUTWARD, so the slot walls are exact; wedge-side parts
    are clamped at the neighbouring slot (other).
    block_side (+1 / -1 along the slot normal): block guide - ONE plate on that side only, the blade
    runs along its face (no slot, no end caps)."""
    half = 0.5 * SLOT_WIDTH
    nn = slot["normal"]
    ub_, bot, top = slot["ubins"], slot["bottom"], slot["v_top"]
    m = len(ub_)
    wedge_dir = slot["side"] if slot["side"] != 0 else None
    shells = []
    for sgn in ((float(block_side),) if block_side else (1.0, -1.0)):
        P = np.vstack([_frame_points(slot, ub_, bot, np.full(m, sgn * half)),
                       _frame_points(slot, ub_, np.full(m, top), np.full(m, sgn * half))])
        tris = []
        for k in range(m - 1):
            a, b, c2, d = k, k + 1, m + k + 1, m + k
            tris += [(a, b, c2), (a, c2, d)]
        raw = _extrude_sheet_raw(P, tris, sgn * (BLOCK_WALL if block_side else SLOT_WALL) * nn)
        if other is not None and wedge_dir is not None and sgn == wedge_dir:
            _clamp_far_layer(raw, len(P), sgn * nn, other)
        shells.append(_finish_shell(raw))
    if block_side:
        return shells
    # end caps (break-off bridges): from the kept-side outer face across the slot to the wedge side;
    # top BRIDGE_FRACTION of the local block height, open below (towards the bone)
    start = -(wedge_dir or 1.0)
    for ua, ub in ((slot["ua"], slot["u0"] + 0.3), (slot["u1"] - 0.3, slot["ub"])):
        n0 = start * (half + SLOT_WALL)
        hb = bridge_height(slot, ua, ub)
        P = _frame_points(slot, np.array([ua, ub, ub, ua]),
                          np.array([top - hb, top - hb, top, top]), np.full(4, n0))
        raw = _extrude_sheet_raw(P, [(0, 1, 2), (0, 2, 3)], -start * (SLOT_WIDTH + 2 * SLOT_WALL) * nn)
        if other is not None and wedge_dir is not None:
            _clamp_far_layer(raw, 4, -start * nn, other)
        shells.append(_finish_shell(raw))
    return shells


# --- outline, rounded edge, local contact relief ---------------------------------------------------


# --- small free islands --------------------------------------------------------------------------


def largest_selection_component(mesh, selected):
    """Keep only the largest edge-connected part of a vertex selection (drops painted specks)."""
    selected = np.asarray(selected, dtype=bool)
    if selected.sum() == 0:
        return selected
    tris = triangles_of(mesh)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = e[selected[e[:, 0]] & selected[e[:, 1]]]
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        n = len(selected)
        g = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, labels = connected_components(g, directed=False)
    except Exception:
        return selected
    counts = np.bincount(labels[selected])
    return selected & (labels == int(np.argmax(counts)))


def bridge_height(slot, ua, ub, fraction=None):
    """`fraction` (default BRIDGE_FRACTION) of the local block height between ua and ub
    (at least BRIDGE_HEIGHT)."""
    inside = (slot["ubins"] >= min(ua, ub)) & (slot["ubins"] <= max(ua, ub))
    bottom = slot["bottom"][inside] if inside.any() else slot["bottom"]
    height = float(slot["v_top"] - np.mean(bottom))
    return max(BRIDGE_HEIGHT, (BRIDGE_FRACTION if fraction is None else fraction) * height)


# --- fixation pins -------------------------------------------------------------------------------
# All pins are parallel to the MASTER cutting plane (direction = saw direction of the master slot).
# Pins of the moving segment are pre-angled with the inverse correction, so after the reduction
# every pin is parallel. Each pin gets a sleeve (bore = pin + clearance, wall SLEEVE_WALL) standing
# on the bone; the guide body gets a clean hole underneath (clipped, extruded along the pin axis).
PIN_CLEARANCE = 0.10          # bore = pin diameter + 0.1 mm (diametral play)
SLEEVE_WALL = 2.0
SLEEVE_SEGMENTS = 32


def sleeve_radii(pin_diameter):
    r_in = 0.5 * float(pin_diameter) + 0.5 * PIN_CLEARANCE
    return r_in, r_in + SLEEVE_WALL


def _perp_frame(a):
    a = unit(np.asarray(a, float))
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(a, helper))
    return e1, np.cross(a, e1)


def line_hits(obb, start, end):
    """Sorted intersection points of the segment start->end with the surface in obb."""
    pts = vtk.vtkPoints()
    obb.IntersectWithLine([float(x) for x in start], [float(x) for x in end], pts, None)
    P = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], dtype=float)
    if len(P) == 0:
        return P
    order = np.argsort(np.linalg.norm(P - np.asarray(start, float), axis=1))
    return P[order]


def plan_pin(obb, point, axis, pin_diameter, gap, height, reach=90.0):
    """Pin through `point` (on the guide), axis = OUTWARD unit direction (bone -> guide).
    Returns dict with entry (first bone hit), far (last bone hit), bone_length, top, radii."""
    a = unit(np.asarray(axis, float))
    p = np.asarray(point, float)
    start = p + 30.0 * a
    hits = line_hits(obb, start, p - reach * a)
    if len(hits) == 0:
        return {"error": "pin axis misses the bone"}
    entry, far = hits[0], hits[-1]
    r_in, r_out = sleeve_radii(pin_diameter)
    top = entry + (gap + height) * a
    return dict(point=p, axis=a, entry=entry, far=far, bone_length=float(np.linalg.norm(far - entry)),
                hits=len(hits), top=top, r_in=r_in, r_out=r_out, diameter=float(pin_diameter))


def sleeve_shell(obb, pin, gap):
    """Closed annular sleeve: bore r_in, outside r_out, top flat at pin['top'], bottom ring
    following the bone (+ gap) along the pin axis."""
    a = pin["axis"]
    e1, e2 = _perp_frame(a)
    entry, top = pin["entry"], pin["top"]
    t_top = float((top - entry) @ a)
    m = SLEEVE_SEGMENTS
    rings = []
    for r in (pin["r_out"], pin["r_in"]):
        bottom, upper = [], []
        for k in range(m):
            ang = 2.0 * np.pi * k / m
            offset = r * (np.cos(ang) * e1 + np.sin(ang) * e2)
            base = entry + offset
            hits = line_hits(obb, base + (t_top + 5.0) * a, base - 25.0 * a)
            t_bottom = float((hits[0] - entry) @ a) + gap if len(hits) else gap
            t_bottom = min(t_bottom, t_top - 1.0)
            bottom.append(t_bottom)
            upper.append(base + t_top * a)
        # a ring sample that misses the bone (sleeve at the bone edge) would hit something far below
        # (fibula, malleolus) and draw a long spike - keep the bottom within 3 mm of the ring's median
        t_b = np.array(bottom)
        t_b = np.maximum(t_b, float(np.median(t_b)) - 3.0)
        bottom = [entry + r * (np.cos(2.0 * np.pi * k / m) * e1 + np.sin(2.0 * np.pi * k / m) * e2) + t_b[k] * a
                  for k in range(m)]
        rings.append((np.array(bottom), np.array(upper)))
    (ob, ot), (ib, it) = rings
    P = np.vstack([ob, ot, it, ib])           # blocks of m points
    OB, OT, IT, IB = 0, m, 2 * m, 3 * m
    tris = []
    for k in range(m):
        k1 = (k + 1) % m
        for A, Bk in ((OB, OT), (OT, IT), (IT, IB), (IB, OB)):
            a0, a1, b0, b1 = A + k, A + k1, Bk + k, Bk + k1
            tris += [(a0, a1, b1), (a0, b1, b0)]
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(P), deep=True))
    poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    for t in tris:
        cells.InsertNextCell(3)
        for q in t:
            cells.InsertCellPoint(int(q))
    poly.SetPolys(cells)
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(poly)
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.SplittingOff()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    return out


def pin_axis_polydata(pins):
    """Tubes along the pin axes (far cortex + 3 mm up to the sleeve top) for display."""
    append = vtk.vtkAppendPolyData()
    for pin in pins:
        if "error" in pin:
            continue
        line = vtk.vtkLineSource()
        line.SetPoint1(*(pin["far"] - 3.0 * pin["axis"]))
        line.SetPoint2(*(pin["top"] + 2.0 * pin["axis"]))
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(0.5 * pin["diameter"])
        tube.SetNumberOfSides(16)
        tube.CappingOn()
        tube.Update()
        append.AddInputData(tube.GetOutput())
    append.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    return out


def _mst_edges(points):
    """Minimum spanning tree (Prim) over 2D/3D points -> list of (i, j)."""
    n = len(points)
    if n < 2:
        return []
    P = np.asarray(points, dtype=float)
    in_tree = [0]
    edges = []
    while len(in_tree) < n:
        best = None
        for i in in_tree:
            for j in range(n):
                if j in in_tree:
                    continue
                dist = float(np.linalg.norm(P[i] - P[j]))
                if best is None or dist < best[0]:
                    best = (dist, i, j)
        edges.append((best[1], best[2]))
        in_tree.append(best[2])
    return edges


def _ring_edges(points, min_width=3.0):
    """Closed ring: the convex hull of the (2D) pin positions as a loop + the minimum spanning tree so
    inner pins are joined too. -> (edges, ring_built). No ring if the pins lie (nearly) in one line."""
    tree = _mst_edges(points)
    P = np.asarray(points, dtype=float)
    if len(P) < 3:
        return tree, False
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(P)
    except Exception:
        return tree, False
    verts = list(hull.vertices)
    longest = max(float(np.linalg.norm(P[verts[k]] - P[verts[(k + 1) % len(verts)]])) for k in range(len(verts)))
    if len(verts) < 3 or float(hull.volume) / max(longest, 1e-6) < 0.5 * min_width:   # area / length ~ width
        return tree, False
    on_hull = set(int(v) for v in verts)
    # tree links only where they reach an INNER pin - no diagonals across the ring
    edges = {tuple(sorted(e)) for e in tree if not (e[0] in on_hull and e[1] in on_hull)}
    for k in range(len(verts)):
        edges.add(tuple(sorted((int(verts[k]), int(verts[(k + 1) % len(verts)])))))
    return sorted(edges), True


def _hull_order(points):
    """Indices of the convex hull of 2D points in order around it, or None."""
    P = np.asarray(points, dtype=float)
    if len(P) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        return [int(v) for v in ConvexHull(P).vertices]
    except Exception:
        return None


def reduction_bar(pins, axis, standoff=15.0, bar_diameter=6.0, boss_height=10.0, wall=2.0, voxel=0.15,
                  bone_obb=None, gap=0.15):
    """Reduction guide as a BAR over the pins (no bone contact): around every (reduced, ~parallel)
    pin a boss of radius r_in + wall and height boss_height, the bosses joined by round rods of
    diameter bar_diameter along the minimum spanning tree of the pins; through-bores r_in along
    each pin's own axis. The bar's centre lies standoff + boss_height/2 above the highest pin entry.
    With bone_obb (the corrected bone): every boss reaches DOWN TO THE BONE and its underside follows
    the bone surface of that fragment + gap (patient-specific seat) - the guide only fits when both
    fragments are correctly reduced; bone contact only around the pins.
    Signed distance field -> iso-surface (always closed). Returns (polydata, info)."""
    from vtk.util.numpy_support import numpy_to_vtk as _n2v
    a = unit(np.asarray(axis, float))
    e1, e2 = _perp_frame(a)
    frame = np.stack([e1, e2, a])
    entries = np.array([p["entry"] for p in pins], float)
    axes = np.array([unit(p["axis"]) for p in pins], float)
    local = entries @ frame.T
    local_axes = axes @ frame.T
    r_in = float(max(p["r_in"] for p in pins))
    r_boss = r_in + float(wall)
    r_bar = 0.5 * float(bar_diameter)
    half_boss = 0.5 * float(boss_height)
    zc = float(local[:, 2].max()) + float(standoff) + half_boss

    def centre_at(i, z):
        t = (z - local[i, 2]) / max(local_axes[i, 2], 1e-6)
        return local[i, 0] + t * local_axes[i, 0], local[i, 1] + t * local_axes[i, 1]

    nodes = np.array([centre_at(i, zc) for i in range(len(pins))])
    edges = _mst_edges(nodes)
    reach = max(r_bar, r_boss) + 3.0 * voxel + 1.0
    zh = max(r_bar, half_boss) + 3.0 * voxel
    z_top = zc + max(r_bar, half_boss)
    seated = bone_obb is not None
    if seated:
        # bosses run down to the bone: their centre lines drift slightly if the axes are not exactly
        # parallel, so the grid also has to include the pin centres at bone level
        low = [centre_at(i, local[i, 2]) for i in range(len(pins))]
        all_uv = np.vstack([nodes, np.array(low)])
    else:
        all_uv = nodes
    u0, u1 = all_uv[:, 0].min() - reach, all_uv[:, 0].max() + reach
    v0, v1 = all_uv[:, 1].min() - reach, all_uv[:, 1].max() + reach
    z0, z1 = zc - zh, zc + zh
    seat_bottom = None
    if seated:
        z0 = float(local[:, 2].min()) - 8.0
        nu_ = int(np.ceil((u1 - u0) / voxel)) + 1
        nv_ = int(np.ceil((v1 - v0) / voxel)) + 1
        us_ = u0 + voxel * np.arange(nu_)
        vs_ = v0 + voxel * np.arange(nv_)
        seat_bottom = np.full((nv_, nu_), np.nan)            # bone height + gap, per (v, u) column
        hit_points = vtk.vtkPoints()
        hit_cells = vtk.vtkIdList()
        for i in range(len(pins)):
            cu, cv = centre_at(i, local[i, 2])
            for jv in np.where(np.abs(vs_ - cv) <= r_boss + voxel)[0]:
                for ju in np.where(np.abs(us_ - cu) <= r_boss + voxel)[0]:
                    if np.hypot(us_[ju] - cu, vs_[jv] - cv) > r_boss + voxel or not np.isnan(seat_bottom[jv, ju]):
                        continue
                    start = us_[ju] * e1 + vs_[jv] * e2 + (z_top + 5.0) * a
                    end = us_[ju] * e1 + vs_[jv] * e2 + (float(local[i, 2]) - 25.0) * a
                    hit_points.Reset()
                    hit_cells.Reset()
                    bone_obb.IntersectWithLine(start.tolist(), end.tolist(), hit_points, hit_cells)
                    if hit_points.GetNumberOfPoints():
                        heights = [float(np.dot(np.asarray(hit_points.GetPoint(n)), a))
                                   for n in range(hit_points.GetNumberOfPoints())]
                        seat_bottom[jv, ju] = max(heights) + float(gap)     # first surface seen from above
        default = float(local[:, 2].max()) + float(gap)
        seat_bottom = np.where(np.isnan(seat_bottom), default, seat_bottom)
        z0 = min(z0, float(np.nanmin(seat_bottom)) - 3.0 * voxel)
    nu, nv, nz = (int(np.ceil((u1 - u0) / voxel)) + 1, int(np.ceil((v1 - v0) / voxel)) + 1,
                  int(np.ceil((z1 - z0) / voxel)) + 1)
    us = u0 + voxel * np.arange(nu)
    vs = v0 + voxel * np.arange(nv)
    zs = z0 + voxel * np.arange(nz)
    V2, U2 = np.meshgrid(vs, us, indexing="ij")
    sdf = np.empty((nz, nv, nu), dtype=np.float32)
    for k, z in enumerate(zs):
        solid = np.full(U2.shape, np.inf)
        for i, j in edges:                                  # round rods (3D capsules at height zc)
            A = nodes[i]
            B = nodes[j]
            AB = B - A
            L2 = max(float(AB @ AB), 1e-12)
            t = np.clip(((U2 - A[0]) * AB[0] + (V2 - A[1]) * AB[1]) / L2, 0.0, 1.0)
            du = U2 - (A[0] + t * AB[0])
            dv = V2 - (A[1] + t * AB[1])
            solid = np.minimum(solid, np.sqrt(du * du + dv * dv + (z - zc) ** 2) - r_bar)
        holes = np.full(U2.shape, np.inf)
        for i in range(len(pins)):                          # bosses + bores along each pin axis
            cu, cv = centre_at(i, z)
            radial = np.hypot(U2 - cu, V2 - cv)
            if seated:                                      # from the bone (+gap) up to the bar top
                boss = np.maximum(np.maximum(radial - r_boss, seat_bottom - z), z - z_top)
            else:
                boss = np.maximum(radial - r_boss, abs(z - zc) - half_boss)
            solid = np.minimum(solid, boss)
            holes = np.minimum(holes, radial - r_in)
        sdf[k] = np.maximum(solid, -holes)
    image = vtk.vtkImageData()
    image.SetDimensions(nu, nv, nz)
    image.SetOrigin(u0, v0, z0)
    image.SetSpacing(voxel, voxel, voxel)
    image.GetPointData().SetScalars(_n2v(sdf.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
    iso = vtk.vtkFlyingEdges3D()
    iso.SetInputData(image)
    iso.SetValue(0, 0.0)
    iso.ComputeNormalsOff()
    iso.Update()
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            matrix.SetElement(r, c, float(frame[c, r]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    to_world = vtk.vtkTransformPolyDataFilter()
    to_world.SetInputConnection(iso.GetOutputPort())
    to_world.SetTransform(transform)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(to_world.GetOutputPort())
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputConnection(clean.GetOutputPort())
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.SplittingOff()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    bottom = zc - max(r_bar, half_boss) if not seated else zc - r_bar     # rods' underside
    deviation = max(float(np.degrees(np.arccos(np.clip(ax @ a, -1.0, 1.0)))) for ax in axes)
    info = dict(frame=frame, nodes=nodes, edges=edges, zc=zc, bottom=bottom, top=zc + max(r_bar, half_boss),
                r_bar=r_bar, r_boss=r_boss, r_in=r_in, max_axis_deviation_deg=deviation,
                protrusion=[float(zc + max(r_bar, half_boss) - h) for h in local[:, 2]])
    return out, info


def reduction_web(pins, axis, bone_obb, height=15.0, web_width=6.0, wall=2.0, gap=0.15, voxel=0.15, ring=False,
                  filled=False, contact_only=False, clearance=6.0, web_thickness=5.0):
    """Reduction guide seated on the corrected bone: a boss (r_in + wall) around every reduced pin
    and WEBS of width web_width joining the bosses along the minimum spanning tree - both over the
    FULL height: from the bone surface (+gap, patient-specific underside) up to a flat top `height`
    mm above the highest pin entry. Through-bores r_in along each pin's own axis.
    Signed distance field -> iso-surface (always closed). Returns (polydata, info)."""
    from vtk.util.numpy_support import numpy_to_vtk as _n2v
    a = unit(np.asarray(axis, float))
    e1, e2 = _perp_frame(a)
    frame = np.stack([e1, e2, a])
    entries = np.array([p["entry"] for p in pins], float)
    axes = np.array([unit(p["axis"]) for p in pins], float)
    local = entries @ frame.T
    local_axes = axes @ frame.T
    r_in = float(max(p["r_in"] for p in pins))
    r_boss = r_in + float(wall)
    half_web = 0.5 * float(web_width)
    z_top = float(local[:, 2].max()) + float(height)

    def centre_at(i, z):
        t = (z - local[i, 2]) / max(local_axes[i, 2], 1e-6)
        return np.array([local[i, 0] + t * local_axes[i, 0], local[i, 1] + t * local_axes[i, 1]])

    base_nodes = np.array([centre_at(i, local[i, 2]) for i in range(len(pins))])
    top_nodes = np.array([centre_at(i, z_top) for i in range(len(pins))])
    ring_built = False
    if ring or filled:
        edges, ring_built = _ring_edges(top_nodes)
    else:
        edges = _mst_edges(top_nodes)
    hull = _hull_order(top_nodes) if (filled and ring_built) else None
    reach = max(half_web, r_boss) + 3.0 * voxel + 1.0
    all_uv = np.vstack([base_nodes, top_nodes])
    u0, u1 = all_uv[:, 0].min() - reach, all_uv[:, 0].max() + reach
    v0, v1 = all_uv[:, 1].min() - reach, all_uv[:, 1].max() + reach
    nu, nv = int(np.ceil((u1 - u0) / voxel)) + 1, int(np.ceil((v1 - v0) / voxel)) + 1
    us = u0 + voxel * np.arange(nu)
    vs = v0 + voxel * np.arange(nv)
    V2, U2 = np.meshgrid(vs, us, indexing="ij")

    def footprint_2d(nodes, bosses=True, webs=True):
        """distance to the union of boss discs and web strips (<= 0 inside)"""
        dist = np.full(U2.shape, np.inf)
        if bosses:
            for n in nodes:
                dist = np.minimum(dist, np.hypot(U2 - n[0], V2 - n[1]) - r_boss)
        if not webs:
            return dist
        for i, j in edges:
            A, B = nodes[i], nodes[j]
            AB = B - A
            t = np.clip(((U2 - A[0]) * AB[0] + (V2 - A[1]) * AB[1]) / max(float(AB @ AB), 1e-12), 0.0, 1.0)
            dist = np.minimum(dist, np.hypot(U2 - (A[0] + t * AB[0]), V2 - (A[1] + t * AB[1])) - half_web)
        if hull is not None:
            # filled frame: the whole area inside the ring is solid (signed distance to the hull polygon)
            inside = np.zeros(U2.shape, dtype=bool)
            edge_dist = np.full(U2.shape, np.inf)
            for k in range(len(hull)):
                A, B = nodes[hull[k]], nodes[hull[(k + 1) % len(hull)]]
                crosses = ((A[1] > V2) != (B[1] > V2)) & \
                    (U2 < (B[0] - A[0]) * (V2 - A[1]) / ((B[1] - A[1]) if abs(B[1] - A[1]) > 1e-12 else 1e-12) + A[0])
                inside ^= crosses
                AB = B - A
                t = np.clip(((U2 - A[0]) * AB[0] + (V2 - A[1]) * AB[1]) / max(float(AB @ AB), 1e-12), 0.0, 1.0)
                edge_dist = np.minimum(edge_dist, np.hypot(U2 - (A[0] + t * AB[0]), V2 - (A[1] + t * AB[1])))
            dist = np.minimum(dist, np.where(inside, -edge_dist, edge_dist))
        return dist

    # patient-specific underside: bone height (+gap) per column, ray cast from above onto the bone
    need = (footprint_2d(base_nodes) <= voxel) | (footprint_2d(top_nodes) <= voxel)
    seat = np.full(U2.shape, np.nan)
    hit_points = vtk.vtkPoints()
    hit_cells = vtk.vtkIdList()
    z_low = float(local[:, 2].min()) - 30.0
    # rays on a coarser lattice (every `step`-th column, ~0.45 mm) - 9x fewer rays; each fine column then
    # takes the HIGHEST of the 2x2 lattice samples around it (conservative: never dips into the bone)
    step = 3
    from scipy.ndimage import binary_dilation
    need_c = binary_dilation(need, iterations=step + 1)
    lattice = np.full((nv // step + 2, nu // step + 2), np.nan)
    for jv_c in range(lattice.shape[0]):
        jv = min(jv_c * step, nv - 1)
        for ju_c in range(lattice.shape[1]):
            ju = min(ju_c * step, nu - 1)
            if not need_c[jv, ju]:
                continue
            start = us[ju] * e1 + vs[jv] * e2 + (z_top + 5.0) * a
            end = us[ju] * e1 + vs[jv] * e2 + z_low * a
            hit_points.Reset()
            hit_cells.Reset()
            bone_obb.IntersectWithLine(start.tolist(), end.tolist(), hit_points, hit_cells)
            if hit_points.GetNumberOfPoints():
                lattice[jv_c, ju_c] = max(float(np.dot(np.asarray(hit_points.GetPoint(n)), a))
                                          for n in range(hit_points.GetNumberOfPoints())) + float(gap)
    jv_all, ju_all = np.where(need)
    lo_v, lo_u = jv_all // step, ju_all // step
    corners = np.stack([lattice[lo_v, lo_u], lattice[lo_v + 1, lo_u],
                        lattice[lo_v, lo_u + 1], lattice[lo_v + 1, lo_u + 1]])
    with np.errstate(all="ignore"):
        best = np.nanmax(np.where(np.isnan(corners), -np.inf, corners), axis=0)
    best = np.where(np.isinf(best), np.nan, best)
    seat[jv_all, ju_all] = best
    seat_raw = seat.copy()
    if np.all(np.isnan(seat)):
        seat[:] = float(local[:, 2].max()) + float(gap)
    seat = np.where(np.isnan(seat), np.nanmax(seat), seat)       # columns beside the bone: no overhang below
    # conservative on steep flanks / at the step between the fragments: each column takes the highest
    # bone of its neighbours (sampling between columns can otherwise dip a few 1/10 mm into the bone)
    from scipy.ndimage import maximum_filter
    seat = maximum_filter(seat, size=3, mode="nearest")
    # contact only at the pins: the guide stands on the pin columns only (small cylinders seated on the
    # bone); the columns are joined by flat BEAMS, each beam just `clearance` mm above the highest bone
    # below IT (not the highest bone anywhere) - low guide, free space for the plate and screws
    web_levels, pin_tops = None, None


    if contact_only:
        # the webs FOLLOW the bone surface `clearance` mm above it (conservative: highest bone within ~1 mm)
        from scipy.ndimage import distance_transform_edt
        size = max(3, int(round(1.0 / voxel)) | 1)
        bone_hi = maximum_filter(np.where(np.isnan(seat_raw), -np.inf, seat_raw), size=size, mode="nearest")
        finite = np.isfinite(bone_hi)
        if not finite.any():
            bone_hi = np.full(U2.shape, float(local[:, 2].max()) + float(gap))
        elif not finite.all():                           # beside the bone: take the nearest bone height
            indices = distance_transform_edt(~finite, return_distances=False, return_indices=True)
            bone_hi = bone_hi[tuple(indices)]
        web_bottom_field = bone_hi - float(gap) + float(clearance)
        # ONE block with a flat top: its underside (the webs) follows the bone `clearance` mm above it,
        # the top is level - `web_thickness` above the highest web underside; the columns reach the bone
        on_webs = footprint_2d(top_nodes, bosses=False) <= voxel
        highest = float(web_bottom_field[on_webs].max()) if on_webs.any() else float(local[:, 2].max()) + float(clearance)
        z_top = max(highest, float(local[:, 2].max()) + float(clearance)) + float(web_thickness)
        web_top_field = np.full(U2.shape, z_top)
        web_levels = True
        pin_tops = [z_top] * len(pins)
        top_nodes = np.array([centre_at(i, z_top) for i in range(len(pins))])
    z0 = float(np.nanmin(seat[need])) - 3.0 * voxel
    z1 = z_top + 3.0 * voxel
    nz = int(np.ceil((z1 - z0) / voxel)) + 1
    zs = z0 + voxel * np.arange(nz)
    sdf = np.empty((nz, nv, nu), dtype=np.float32)
    cached = None                       # (nodes, footprint, holes) - reused while the pins barely move
    for k, z in enumerate(zs):
        nodes = np.array([centre_at(i, z) for i in range(len(pins))])
        if cached is None or float(np.abs(nodes - cached[0]).max()) > 0.1 * voxel:
            holes = np.full(U2.shape, np.inf)
            for n in nodes:
                holes = np.minimum(holes, np.hypot(U2 - n[0], V2 - n[1]) - r_in)
            if web_levels is None:
                cached = (nodes, footprint_2d(nodes), None, holes)
            else:
                discs = [np.hypot(U2 - n[0], V2 - n[1]) - r_boss for n in nodes]
                cached = (nodes, discs, footprint_2d(nodes, bosses=False), holes)
        _nodes, fp, strips, holes = cached
        if strips is None:
            solid = np.maximum(np.maximum(fp, seat - z), z - z_top)
        else:
            solid = np.full(U2.shape, np.inf)
            for i, disc in enumerate(fp):                  # the columns: from the bone up to their own top
                solid = np.minimum(solid, np.maximum(np.maximum(disc, seat - z), z - pin_tops[i]))
            # the webs between them, following the bone `clearance` mm above it
            solid = np.minimum(solid, np.maximum(np.maximum(strips, web_bottom_field - z), z - web_top_field))
        sdf[k] = np.maximum(solid, -holes)
    image = vtk.vtkImageData()
    image.SetDimensions(nu, nv, nz)
    image.SetOrigin(u0, v0, z0)
    image.SetSpacing(voxel, voxel, voxel)
    image.GetPointData().SetScalars(_n2v(sdf.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
    iso = vtk.vtkFlyingEdges3D()
    iso.SetInputData(image)
    iso.SetValue(0, 0.0)
    iso.ComputeNormalsOff()
    iso.Update()
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            matrix.SetElement(r, c, float(frame[c, r]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    to_world = vtk.vtkTransformPolyDataFilter()
    to_world.SetInputConnection(iso.GetOutputPort())
    to_world.SetTransform(transform)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(to_world.GetOutputPort())
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputConnection(clean.GetOutputPort())
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.SplittingOff()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    deviation = max(float(np.degrees(np.arccos(np.clip(ax @ a, -1.0, 1.0)))) for ax in axes)
    heights = seat[need] - float(gap)
    info = dict(frame=frame, edges=edges, top=z_top, r_in=r_in, r_boss=r_boss, max_axis_deviation_deg=deviation,
                ring=ring_built, filled=hull is not None, contact_only=web_levels is not None,
                web_clearance=float(clearance) if web_levels is not None else None,
                protrusion=[float((pin_tops[i] if pin_tops else z_top) - h) for i, h in enumerate(local[:, 2])],
                guide_height_range=(float(z_top - np.nanmax(heights)), float(z_top - np.nanmin(heights))))
    return out, info


def _seat_heights(bone_obb, origin, e_u, e_v, e_w, lo, hi, step, gap):
    """Highest bone surface (+gap) along e_v over the (e_w, e_u) lattice lo .. hi with `step` - a
    z-buffer of densely sampled bone triangles (vectorised; replaces one ray per lattice node).
    Every sample counts for the lattice node it lies within step/2 of, so the height of a node is the
    highest bone in its cell (conservative). NaN where no bone lies below."""
    sa = lo[0] + step * np.arange(int(np.ceil((hi[0] - lo[0]) / step)) + 1)
    sc = lo[2] + step * np.arange(int(np.ceil((hi[2] - lo[2]) / step)) + 1)
    coarse = np.full((len(sc), len(sa)), -np.inf)
    poly = bone_obb.GetDataSet()
    if poly is None or poly.GetNumberOfPoints() == 0:
        return np.full(coarse.shape, np.nan)
    P = vtk_to_numpy(poly.GetPoints().GetData()).astype(float) - origin
    L = np.stack([P @ e_u, P @ e_v, P @ e_w], axis=1)
    polys = poly.GetPolys()
    off = vtk_to_numpy(polys.GetOffsetsArray()).astype(np.int64)
    if len(off) < 2 or not np.all(np.diff(off) == 3):
        tf = vtk.vtkTriangleFilter()
        tf.SetInputData(poly)
        tf.Update()
        polys = tf.GetOutput().GetPolys()
    T = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64).reshape(-1, 3)
    m = 0.5 * step + 0.01
    inb = ((L[:, 0] >= lo[0] - m) & (L[:, 0] <= hi[0] + m) & (L[:, 2] >= lo[2] - m) & (L[:, 2] <= hi[2] + m)
           & (L[:, 1] >= lo[1] - 25.0) & (L[:, 1] <= hi[1] + 5.0))
    T = T[inb[T].any(axis=1)]
    if len(T) == 0:
        return np.full(coarse.shape, np.nan)
    A, Bv, Cv = L[T[:, 0]], L[T[:, 1]], L[T[:, 2]]
    edge = np.max([np.linalg.norm((Bv - A)[:, [0, 2]], axis=1), np.linalg.norm((Cv - Bv)[:, [0, 2]], axis=1),
                   np.linalg.norm((A - Cv)[:, [0, 2]], axis=1)], axis=0)
    # subdivide each triangle so that the samples lie <= step/2 apart in the (u, w) plane
    k_tri = np.clip(np.ceil(edge / (0.5 * step)).astype(int), 1, 40)
    for k in np.unique(k_tri):
        sel = k_tri == k
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        ok = i + j <= k
        w1, w2 = (i[ok] / k)[None, :, None], (j[ok] / k)[None, :, None]
        a, b, c = A[sel][:, None, :], Bv[sel][:, None, :], Cv[sel][:, None, :]
        S = (a + w1 * (b - a) + w2 * (c - a)).reshape(-1, 3)
        ia = np.rint((S[:, 0] - lo[0]) / step).astype(int)
        ic = np.rint((S[:, 2] - lo[2]) / step).astype(int)
        good = ((ia >= 0) & (ia < len(sa)) & (ic >= 0) & (ic < len(sc))
                & (S[:, 1] >= lo[1] - 25.0) & (S[:, 1] <= hi[1] + 5.0))
        np.maximum.at(coarse, (ic[good], ia[good]), S[good, 1])
    coarse[np.isinf(coarse)] = np.nan
    return coarse + float(gap)


BLOCK_FRAME_ARRAY = "DoggyboneBlockFrame"


def saw_block_pair(slot_a, slot_b, bone_obb, gap, voxel=0.12, seat_step=0.15, block=False):
    """ONE saw block for both cuts (closing wedge): the outer faces of the two blocks are extended until
    they close each other (no crossing plates, no mirrored remnants), the space between the blocks is
    filled, the underside follows the bone (+gap). Both kerfs run through the whole block (open at the
    sides, from the bone to the top); only at the top of both ends a bridge stays across each kerf -
    the break-off points that hold the block together until it is broken apart after sawing.
    Signed distance field of planes -> iso-surface (closed, flat faces)."""
    from vtk.util.numpy_support import numpy_to_vtk as _n2v
    from scipy.ndimage import distance_transform_edt
    half = 0.5 * SLOT_WIDTH
    # block guide: no slots - the whole wedge between the two cutting faces is open, the blade runs
    # along each face (the kerf lies in the wedge); thicker walls behind the faces
    outer = half + (BLOCK_WALL if block else SLOT_WALL)
    slots = (slot_a, slot_b)
    frames = []
    for slot in slots:
        C, u, v, n = (np.asarray(slot[k], float) for k in ("centre", "u", "v", "normal"))
        side = float(slot.get("side") or 1.0)
        # break-off bridges: small, only at the top of both ENDS of the block (see below)
        hb = BREAK_BRIDGE_HEIGHT
        frames.append((C, u, v, n, side, float(slot["ua"]), float(slot["ub"]), float(slot["u0"]),
                       float(slot["u1"]), float(slot["v_top"]), hb))
    ua_ = frames[0][1]
    ub_ = frames[1][1] * (1.0 if float(np.dot(frames[0][1], frames[1][1])) >= 0 else -1.0)
    e_v = unit(frames[0][2] + frames[1][2])
    e_u = unit(ua_ + ub_ - float(np.dot(ua_ + ub_, e_v)) * e_v)
    e_w = np.cross(e_u, e_v)
    corners = []
    for slot in slots:
        low = float(np.min(slot["bottom"])) - 1.0
        for uu in (float(slot["ua"]), float(slot["ub"])):
            for vv in (low, float(slot["v_top"])):
                for nv in (-outer, outer):
                    corners.append(_frame_points(slot, np.array([uu]), np.array([vv]), np.array([nv]))[0])
    corners = np.array(corners)
    origin = corners.mean(axis=0)
    local = (corners - origin) @ np.stack([e_u, e_v, e_w]).T
    # common length limits along e_u (flat, parallel end faces for the whole block)
    a_lo, a_hi = float(local[:, 0].min()), float(local[:, 0].max())
    lo = local.min(axis=0) - 3.0
    hi = local.max(axis=0) + 3.0
    na, nb, nc = (int(np.ceil((hi[k] - lo[k]) / voxel)) + 1 for k in range(3))
    As = lo[0] + voxel * np.arange(na)
    Bs = lo[1] + voxel * np.arange(nb)
    Cs = lo[2] + voxel * np.arange(nc)
    CC, AA = np.meshgrid(Cs, As, indexing="ij")               # (nc, na) columns
    # underside: bone surface seen from above (+gap), sampled on a coarser grid and interpolated
    from scipy.ndimage import map_coordinates
    coarse = _seat_heights(bone_obb, origin, e_u, e_v, e_w, lo, hi, seat_step, gap)
    if np.all(np.isnan(coarse)):
        coarse[:] = lo[1]
    else:                                                     # beyond the bone: continue the nearest seat
        missing = np.isnan(coarse)
        if missing.any():
            _d, (ic, ia) = distance_transform_edt(missing, return_indices=True)
            coarse = coarse[ic, ia]
    # conservative on slopes: highest bone within ~0.3 mm, then interpolate
    from scipy.ndimage import maximum_filter
    coarse = maximum_filter(coarse, size=2 * max(1, int(round(0.3 / seat_step))) + 1, mode="nearest")
    seat = map_coordinates(coarse, [(CC - lo[2]) / seat_step, (AA - lo[0]) / seat_step], order=1, mode="nearest")
    # even ends: in each end zone (corner bridge + 1 mm) both halves end at ONE flat level - the highest
    # seat in that zone - no strips of one half reaching further down the bone flank than the other
    w_lo_b, w_hi_b = float(local[:, 2].min()), float(local[:, 2].max())
    in_w = (CC >= w_lo_b) & (CC <= w_hi_b)
    for end in (AA <= a_lo + BRIDGE_WIDTH + 1.0, AA >= a_hi - BRIDGE_WIDTH - 1.0):
        zone = end & in_w
        if zone.any():
            level = float(seat[zone].max())
            seat = np.where(end, np.maximum(seat, level), seat)
    sdf = np.empty((nc, nb, na), dtype=np.float32)
    for jb, b in enumerate(Bs):
        X = origin + AA[..., None] * e_u + b * e_v + CC[..., None] * e_w       # (nc, na, 3)
        f_outer = np.full(AA.shape, -np.inf)
        f_u = np.full(AA.shape, np.inf)
        f_top = np.full(AA.shape, -np.inf)
        kerf = np.full(AA.shape, np.inf)
        plates = np.full(AA.shape, np.inf)
        normals_dist = []
        for C, u, v, n, side, ua, ub, u0, u1, top, hb in frames:
            rel = X - C
            s_n = rel @ n
            s_v = rel @ v
            f_outer = np.maximum(f_outer, -side * s_n - outer)            # not beyond the kept-side face
            # the OUTER wall of this cut (behind its kerf, away from the wedge) on its own - kept even
            # where the two cuts cross inside the block and the wedge between them flips over
            plates = np.minimum(plates, np.maximum(-side * s_n - outer, side * s_n))
            f_top = np.maximum(f_top, s_v - top)
            normals_dist.append(s_n)
        f_outer = np.minimum(f_outer, plates)
        # hinge stop (apply_hinge_stop): a kerf ends before the hinge - beyond its end the block is solid
        hinge_terms = []
        for slot in slots:
            if "hinge_end" in slot:
                sign, u_end = slot["hinge_end"]
                s_u = (X - np.asarray(slot["centre"], float)) @ np.asarray(slot["u"], float)
                hinge_terms.append(sign * (s_u - u_end))
            else:
                hinge_terms.append(np.full(AA.shape, -np.inf))
        # kerfs through the WHOLE block (open at the sides, from the bone to the top) except a SMALL
        # break-off bridge at each end of each kerf: BREAK_BRIDGE_LENGTH along the kerf and
        # BREAK_BRIDGE_HEIGHT below the ACTUAL top of the block (the top of the whole block, which may be
        # set by the other slot) - the halves hang together only at these four corners
        at_end = np.minimum(AA - (a_lo + BREAK_BRIDGE_LENGTH), (a_hi - BREAK_BRIDGE_LENGTH) - AA)    # < 0 at an end
        bridge = np.maximum(at_end, -f_top - BREAK_BRIDGE_HEIGHT)                                  # <= 0 in a bridge
        if block:
            # open window = the wedge between both cutting faces (side * s_n > -half for both slots)
            window = np.full(AA.shape, -np.inf)
            for (C, u, v, n, side, ua, ub, u0, u1, top, hb), s_n, h_term in zip(frames, normals_dist, hinge_terms):
                window = np.maximum(np.maximum(window, -side * s_n - half), h_term)
            kerf = np.maximum(window, -bridge)
        else:
            for s_n, h_term in zip(normals_dist, hinge_terms):
                kerf = np.minimum(kerf, np.maximum(np.maximum(np.abs(s_n) - half, -bridge), h_term))
        f_u = np.maximum(a_lo - AA, AA - a_hi)                           # common block length
        solid = np.maximum(np.maximum(f_outer, f_u), np.maximum(f_top, seat - b))
        sdf[:, jb, :] = np.maximum(solid, -kerf)
    # close the surface at the grid border (a solid truncated there gets a flat face, never a hole)
    sdf[0, :, :] = np.maximum(sdf[0, :, :], voxel)
    sdf[-1, :, :] = np.maximum(sdf[-1, :, :], voxel)
    sdf[:, 0, :] = np.maximum(sdf[:, 0, :], voxel)
    sdf[:, -1, :] = np.maximum(sdf[:, -1, :], voxel)
    sdf[:, :, 0] = np.maximum(sdf[:, :, 0], voxel)
    sdf[:, :, -1] = np.maximum(sdf[:, :, -1], voxel)
    image = vtk.vtkImageData()
    image.SetDimensions(na, nb, nc)
    image.SetOrigin(lo[0], lo[1], lo[2])
    image.SetSpacing(voxel, voxel, voxel)
    image.GetPointData().SetScalars(_n2v(sdf.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
    iso = vtk.vtkFlyingEdges3D()
    iso.SetInputData(image)
    iso.SetValue(0, 0.0)
    iso.ComputeNormalsOff()
    iso.Update()
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        matrix.SetElement(r, 0, float(e_u[r]))
        matrix.SetElement(r, 1, float(e_v[r]))
        matrix.SetElement(r, 2, float(e_w[r]))
        matrix.SetElement(r, 3, float(origin[r]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    to_world = vtk.vtkTransformPolyDataFilter()
    to_world.SetInputConnection(iso.GetOutputPort())
    to_world.SetTransform(transform)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(to_world.GetOutputPort())
    single = vtk.vtkPolyDataConnectivityFilter()               # one piece: drop degenerate specks where
    single.SetInputConnection(clean.GetOutputPort())           # the kerfs cross at the wedge tip
    single.SetExtractionModeToLargestRegion()
    compact = vtk.vtkCleanPolyData()
    compact.SetInputConnection(single.GetOutputPort())
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputConnection(compact.GetOutputPort())
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.SplittingOff()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    # block frame for the guide body: its end faces (along e_u) trim the body flush (see sdf_guide_body)
    frame = vtk.vtkDoubleArray()
    frame.SetName(BLOCK_FRAME_ARRAY)
    for value in np.concatenate([origin, e_u, e_w]):
        frame.InsertNextValue(float(value))
    out.GetFieldData().AddArray(frame)
    return out


# ---------------------------------------------------------------------------------------------
# Closing wedge: hinge stop and break-off strut at the hinge side of the saw block
# ---------------------------------------------------------------------------------------------

def kerf_hinge_u(slot, other):
    """Where the OTHER cutting plane meets this slot, as this slot's u coordinate (along the kerf),
    evaluated over the depth of the block; the value nearest to the slot centre (most restrictive)
    is returned, or None if the planes are (nearly) parallel."""
    C, u, v = (np.asarray(slot[k], float) for k in ("centre", "u", "v"))
    o2, n2 = np.asarray(other["origin"], float), unit(np.asarray(other["normal"], float))
    k = float(n2 @ u)
    if abs(k) < 1e-6:
        return None
    bottom = np.asarray(slot.get("bottom", [0.0]), float)
    depths = np.linspace(float(np.min(bottom)), float(slot["v_top"]), 7)
    values = [-float(n2 @ (C + d * v - o2)) / k for d in depths]
    side = 1.0 if np.mean(values) >= 0.0 else -1.0
    return min(values) if side > 0 else max(values)


def apply_hinge_stop(slots, stop):
    """Closing wedge with the hinge (where the two cutting planes meet) inside the block's length:
    each kerf ends `stop` mm before the hinge; beyond that the block stays solid, so the frame is
    closed at the hinge side and the blade cannot cut through the hinge. Changes the slot dicts
    (u0/u1, ua/ub, length, 'hinge_end') and returns a short note ('' if nothing was changed)."""
    good = [s for s in slots if "error" not in s]
    if len(good) != 2 or stop <= 0.0:
        return ""
    notes = []
    for index, slot in enumerate(good):
        u_h = kerf_hinge_u(slot, good[1 - index])
        if u_h is None or not (float(slot["ua"]) - 2.0 < u_h < float(slot["ub"]) + 2.0):
            continue
        if u_h > 0.0:
            end = min(float(slot["u1"]), u_h - stop)
            if end <= float(slot["u0"]) + 2.0:
                continue
            slot["u1"], slot["ub"], slot["hinge_end"] = end, end + BRIDGE_WIDTH, (1.0, end)
        else:
            end = max(float(slot["u0"]), u_h + stop)
            if end >= float(slot["u1"]) - 2.0:
                continue
            slot["u0"], slot["ua"], slot["hinge_end"] = end, end - BRIDGE_WIDTH, (-1.0, end)
        slot["length"] = float(slot["u1"]) - float(slot["u0"])
        notes.append(f"{slot.get('label', 'cut')} ends {stop:.1f} mm before the hinge")
    return ("Hinge stop: " + ", ".join(notes) + " - frame closed there, the hinge cannot be cut through.") if notes else ""


def hinge_strut(slots, block=None, bar_radius=1.3, neck_radius=0.55, neck_height=2.0, sink=0.5):
    """Break-off strut across the closing wedge AT THE HINGE: a round bar from the proximal to the
    distal outer wall of the saw block, standing on the block's real top surface on two thin necks
    (break them with pliers once the guide is pinned). With `block` (the saw_block_pair mesh) the
    feet are found by casting rays down onto the block, so they always sit on it; if the walls do
    not reach the hinge position the strut moves towards the slot centre until both feet sit.
    Returns a closed polydata (three overlapping closed tubes) or None."""
    good = [s for s in slots if "error" not in s]
    if len(good) != 2:
        return None
    frames = []
    for index, slot in enumerate(good):
        C, u, v, n = (np.asarray(slot[k], float) for k in ("centre", "u", "v", "normal"))
        u_h = kerf_hinge_u(slot, good[1 - index])
        if u_h is None:                                       # parallel cuts: at the u1 end
            u_h = float(slot["u1"])
        side = float(slot.get("side") or 1.0)
        frames.append((C, u, v, n, side, u_h, float(slot["v_top"])))
    up = unit(frames[0][2] + frames[1][2])
    mid_wall = 0.5 * SLOT_WIDTH + 0.5 * SLOT_WALL                # middle of each outer wall
    obb = None
    if block is not None and block.GetNumberOfPoints() > 0:
        obb = vtk.vtkOBBTree()
        obb.SetDataSet(block)
        obb.BuildLocator()

    def top_hit(point):
        """Highest point of the block straight below / above `point` (None without a hit)."""
        if obb is None:
            return point
        hits = vtk.vtkPoints()
        obb.IntersectWithLine([float(x) for x in point + 30.0 * up], [float(x) for x in point - 30.0 * up], hits, None)
        if hits.GetNumberOfPoints() == 0:
            return None
        found = [np.array(hits.GetPoint(i)) for i in range(hits.GetNumberOfPoints())]
        return max(found, key=lambda q: float(q @ up))

    feet = None
    for step in np.arange(0.0, 12.01, 0.5):                     # from the hinge towards the slot centre
        candidate = []
        for C, u, v, n, side, u_h, v_top in frames:
            u_at = u_h - np.sign(u_h) * step
            guess = C + u_at * u + v_top * v - side * mid_wall * n
            hit = top_hit(guess)
            if hit is None or abs(float((hit - guess) @ up)) > 3.0:     # no wall here (e.g. over a kerf)
                break
            candidate.append(hit)
        if len(candidate) == 2:
            feet = candidate
            break
    if feet is None:
        return None
    top_level = max(float(f @ up) for f in feet)                 # bar level: above the higher foot
    tops = [f + (top_level - float(f @ up) + neck_height + bar_radius) * up for f in feet]
    append = vtk.vtkAppendPolyData()
    for a, b, r in ((feet[0] - sink * up, tops[0], neck_radius), (feet[1] - sink * up, tops[1], neck_radius),
                    (tops[0], tops[1], bar_radius)):
        line = vtk.vtkLineSource()
        line.SetPoint1(*[float(x) for x in a])
        line.SetPoint2(*[float(x) for x in b])
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(float(r))
        tube.SetNumberOfSides(20)
        tube.CappingOn()
        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(tube.GetOutputPort())
        weld = vtk.vtkCleanPolyData()                    # caps share the end rings -> closed shell
        weld.SetInputConnection(tri.GetOutputPort())
        weld.SetTolerance(1e-6)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(weld.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        part = vtk.vtkPolyData()
        part.DeepCopy(normals.GetOutput())
        append.AddInputData(part)
    append.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    return out


# ---------------------------------------------------------------------------------------------
# Raised label on a guide (patient number, bone/side, arrow towards proximal)
# ---------------------------------------------------------------------------------------------

def _glyph(char):
    """2D glyph (x to the right, y up, baseline at 0), height ~1 for capitals."""
    if char == ">":                                  # proximal arrow: solid triangle
        points = vtk.vtkPoints()
        for x, y in ((0.0, 0.0), (0.0, 1.0), (0.9, 0.5)):
            points.InsertNextPoint(x, y, 0.0)
        tri = vtk.vtkCellArray()
        tri.InsertNextCell(3)
        for k in range(3):
            tri.InsertCellPoint(k)
        glyph = vtk.vtkPolyData()
        glyph.SetPoints(points)
        glyph.SetPolys(tri)
        return glyph
    text = vtk.vtkVectorText()
    text.SetText(char)
    text.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(text.GetOutputPort())
    tri.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(tri.GetOutput())
    return out


def emboss_label(body_obb, text, centre, along, up, height=3.0, raised=0.6, max_length=None):
    """Raised letters on a guide surface: `text` runs along `along` (reading direction), letters
    stand on the surface found below them (ray cast along -up onto body_obb) and rise `raised` mm
    above its highest point; each letter is anchored separately so it follows a curved top.
    Returns (closed polydata or None, used letter height)."""
    up = unit(np.asarray(up, float))
    along = np.asarray(along, float)
    along = unit(along - float(np.dot(along, up)) * up)
    side = np.cross(up, along)                        # glyph 'up' direction -> readable from above
    centre = np.asarray(centre, float)
    glyphs = []
    cursor = 0.0
    for char in text:
        if char == " ":
            cursor += 0.55
            continue
        glyph = _glyph(char)
        if glyph.GetNumberOfPoints() == 0:
            continue
        b = glyph.GetBounds()
        glyphs.append((glyph, cursor - b[0]))
        cursor += (b[1] - b[0]) + 0.18
    if not glyphs:
        return None, 0.0
    total = cursor - 0.18
    scale = float(height)
    if max_length is not None and total * scale > max_length:
        scale = max(1.2, float(max_length) / total)
    hit_points = vtk.vtkPoints()
    hit_cells = vtk.vtkIdList()

    def surface_height(point):
        hit_points.Reset()
        hit_cells.Reset()
        body_obb.IntersectWithLine((point + 40.0 * up).tolist(), (point - 40.0 * up).tolist(), hit_points, hit_cells)
        if hit_points.GetNumberOfPoints() == 0:
            return None
        return max(float(np.dot(np.asarray(hit_points.GetPoint(k)) - centre, up)) for k in range(hit_points.GetNumberOfPoints()))

    append = vtk.vtkAppendPolyData()
    x0 = -0.5 * total * scale
    for glyph, offset in glyphs:
        b = glyph.GetBounds()
        heights = []
        for gx in (b[0], 0.5 * (b[0] + b[1]), b[1]):
            for gy in (b[2], 0.5 * (b[2] + b[3]), b[3]):
                lx = x0 + (offset + gx) * scale
                ly = (gy - 0.5) * scale
                h = surface_height(centre + lx * along + ly * side)
                if h is not None:
                    heights.append(h)
        if len(heights) < 5:                          # letter would hang over an edge -> skip it
            continue
        bottom, top = min(heights) - 0.3, max(heights) + float(raised)
        extrude = vtk.vtkLinearExtrusionFilter()
        extrude.SetInputData(glyph)
        extrude.SetExtrusionTypeToVectorExtrusion()
        extrude.SetVector(0.0, 0.0, 1.0)
        extrude.SetScaleFactor(top - bottom)
        extrude.CappingOn()
        extrude.Update()
        matrix = vtk.vtkMatrix4x4()
        for r in range(3):
            matrix.SetElement(r, 0, float(along[r] * scale))
            matrix.SetElement(r, 1, float(side[r] * scale))
            matrix.SetElement(r, 2, float(up[r]))
            origin = centre + (x0 + offset * scale) * along - 0.5 * scale * side + bottom * up
            matrix.SetElement(r, 3, float(origin[r]))
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        placed = vtk.vtkTransformPolyDataFilter()
        placed.SetInputConnection(extrude.GetOutputPort())
        placed.SetTransform(transform)
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(placed.GetOutputPort())
        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputConnection(clean.GetOutputPort())
        nrm.ConsistencyOn()
        nrm.AutoOrientNormalsOn()
        nrm.SplittingOff()
        nrm.Update()
        letter = vtk.vtkPolyData()
        letter.DeepCopy(nrm.GetOutput())
        append.AddInputData(letter)
    if append.GetNumberOfInputConnections(0) == 0:
        return None, scale
    append.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    return out, scale


def label_units(text):
    """Length of `text` in glyph units (letter height = 1), as laid out by emboss_label."""
    cursor = 0.0
    for char in text:
        if char == " ":
            cursor += 0.55
            continue
        glyph = _glyph(char)
        if glyph.GetNumberOfPoints() == 0:
            continue
        b = glyph.GetBounds()
        cursor += (b[1] - b[0]) + 0.18
    return max(cursor - 0.18, 0.0)


def find_label_spot(guide, up, directions, units, heights=(3.0, 2.5, 2.0, 1.6, 1.3), res=0.25,
                    flat_tol=0.25, margin=0.6):
    """Free, flat spot on the TOP of a finished guide (all parts) for a label `units` letter-heights
    long: a height map seen from `up` marks cells whose 3x3 neighbourhood is flat; a rectangle
    (label + margin) must lie completely on such cells at (nearly) one height. Tries the largest
    letter height first and the given reading directions. Returns (centre, along, height) or None."""
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy.ndimage import maximum_filter, minimum_filter, distance_transform_edt
    up = unit(np.asarray(up, float))
    e1, e2 = _perp_frame(up)
    P = vtk_to_numpy(guide.GetPoints().GetData()).astype(float)
    o = P.mean(axis=0)
    rel = P - o
    a_lo, a_hi = float((rel @ e1).min()) - 1.0, float((rel @ e1).max()) + 1.0
    b_lo, b_hi = float((rel @ e2).min()) - 1.0, float((rel @ e2).max()) + 1.0
    z_hi = float((rel @ up).max()) + 5.0
    z_lo = float((rel @ up).min()) - 5.0
    As = np.arange(a_lo, a_hi, res)
    Bs = np.arange(b_lo, b_hi, res)
    # height map from the surface itself (fast): subdivide long edges to <= res, then keep the highest
    # vertex per grid cell (no ray casting - that took minutes on large guides)
    sub = vtk.vtkAdaptiveSubdivisionFilter()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(guide)
    sub.SetInputConnection(tri.GetOutputPort())
    sub.SetMaximumEdgeLength(res)
    sub.SetMaximumNumberOfPasses(12)
    sub.Update()
    Q = vtk_to_numpy(sub.GetOutput().GetPoints().GetData()).astype(float) - o
    qa = np.floor(((Q @ e1) - a_lo) / res).astype(int)
    qb = np.floor(((Q @ e2) - b_lo) / res).astype(int)
    qz = Q @ up
    ok = (qa >= 0) & (qa < len(As)) & (qb >= 0) & (qb < len(Bs))
    H = np.full((len(Bs), len(As)), -np.inf)
    np.maximum.at(H, (qb[ok], qa[ok]), qz[ok])
    H[np.isinf(H)] = np.nan
    valid = ~np.isnan(H)
    if valid.sum() < 50:
        return None
    Hf = np.where(valid, H, -1e6)
    flat = valid & (maximum_filter(Hf, size=3) - minimum_filter(np.where(valid, H, 1e6), size=3) < flat_tol)
    clearance = distance_transform_edt(flat) * res            # distance to the nearest non-flat cell
    order = np.argsort(-clearance.ravel())
    for height in heights:
        half_len = 0.5 * units * height + margin
        half_wid = 0.5 * height + margin
        if clearance.max() < half_wid:
            continue
        for direction in directions:
            d = np.asarray(direction, float)
            d = d - float(np.dot(d, up)) * up
            if np.linalg.norm(d) < 1e-6:
                continue
            d = unit(d)
            da, db = float(d @ e1), float(d @ e2)
            sa, sb = -db, da                                   # across the reading direction
            ts = np.arange(-half_len, half_len + 1e-9, res)
            ss = np.arange(-half_wid, half_wid + 1e-9, res)
            TT, SS = np.meshgrid(ts, ss)
            offs_a = (TT * da + SS * sa).ravel()
            offs_b = (TT * db + SS * sb).ravel()
            for flat_index in order[:400]:
                jb, ja = divmod(int(flat_index), len(As))
                if clearance[jb, ja] < half_wid:
                    break
                ia = np.rint((As[ja] + offs_a - a_lo) / res).astype(int)
                ib = np.rint((Bs[jb] + offs_b - b_lo) / res).astype(int)
                if ia.min() < 0 or ib.min() < 0 or ia.max() >= len(As) or ib.max() >= len(Bs):
                    continue
                if not np.all(flat[ib, ia]):
                    continue
                heights_under = H[ib, ia]
                if float(heights_under.max() - heights_under.min()) > 2.0 * flat_tol:
                    continue
                centre = o + As[ja] * e1 + Bs[jb] * e2 + float(np.median(heights_under)) * up
                return centre, d, float(height)
    return None


def _box(centre, e_along, e_side, e_up, length, width, height):
    """Closed box polydata centred at `centre` with edges along the three unit vectors."""
    cube = vtk.vtkCubeSource()
    cube.SetXLength(1.0)
    cube.SetYLength(1.0)
    cube.SetZLength(1.0)
    cube.Update()
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        matrix.SetElement(r, 0, float(e_along[r] * length))
        matrix.SetElement(r, 1, float(e_side[r] * width))
        matrix.SetElement(r, 2, float(e_up[r] * height))
        matrix.SetElement(r, 3, float(centre[r]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    placed = vtk.vtkTransformPolyDataFilter()
    placed.SetInputConnection(cube.GetOutputPort())
    placed.SetTransform(transform)
    clean = vtk.vtkCleanPolyData()                    # merge the cube's per-face corner points
    clean.SetInputConnection(placed.GetOutputPort())
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(clean.GetOutputPort())
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputConnection(tri.GetOutputPort())
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.SplittingOff()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    return out


def label_tag(guide, up, proximal, lines, letter=2.5, thickness=1.6, gap=1.5, neck_width=3.0,
              neck_thickness=1.2, margin=1.0, bone_obb=None):
    """Small break-off label tag: a flat plate that sticks out of the PROXIMAL end of the guide at its
    top, joined only by a thin neck (break it off with pliers after printing / before use). The plate
    carries raised text (`lines`, one or two) and an arrow that points proximally.
    Returns (polydata or None, info dict)."""
    from vtk.util.numpy_support import vtk_to_numpy
    up = unit(np.asarray(up, float))
    along = np.asarray(proximal, float)
    along = along - float(np.dot(along, up)) * up
    if np.linalg.norm(along) < 1e-6:
        along = _perp_frame(up)[0]
    along = unit(along)
    side = np.cross(up, along)
    lines = [line for line in lines if line]
    if not lines:
        return None, {"error": "no text"}
    # size of the plate from the text (the arrow goes at the end of the last line)
    widths = [label_units(line) + (1.4 if k == len(lines) - 1 else 0.0) for k, line in enumerate(lines)]
    length = max(widths) * letter + 2.0 * margin
    width = len(lines) * letter + (len(lines) - 1) * 1.0 + 2.0 * margin
    # anchor: the most proximal point of the guide's top region
    P = vtk_to_numpy(guide.GetPoints().GetData()).astype(float)
    hu = P @ up
    top_region = P[hu >= float(hu.max()) - 3.0]
    anchor = top_region[int(np.argmax(top_region @ along))]
    h_top = float(anchor @ up)
    base_point = anchor - float(anchor @ up) * up       # anchor without its height component
    start = float(anchor @ along) + gap                 # plate begins `gap` mm beyond the guide
    plate_centre = (base_point - float(base_point @ along) * along + (start + 0.5 * length) * along
                    + (h_top - 0.5 * thickness) * up)
    plate = _box(plate_centre, along, side, up, length, width, thickness)
    # thin neck from 1.5 mm inside the guide to 0.5 mm inside the plate, at mid thickness
    n0, n1 = float(anchor @ along) - 1.5, start + 0.5
    neck_centre = (base_point - float(base_point @ along) * along + 0.5 * (n0 + n1) * along
                   + (h_top - 0.5 * thickness) * up)
    neck = _box(neck_centre, along, side, up, n1 - n0, neck_width, neck_thickness)
    # raised text on the plate
    obb = vtk.vtkOBBTree()
    obb.SetDataSet(plate)
    obb.BuildLocator()
    top_centre = plate_centre + 0.5 * thickness * up
    parts = [plate, neck]
    pitch = letter + 1.0
    for k, line in enumerate(lines):
        offset = (0.5 * (len(lines) - 1) - k) * pitch       # first line on the +side half
        line_units = label_units(line)
        line_len = widths[k] * letter
        text_centre = top_centre + offset * side - 0.5 * (line_len - line_units * letter) * along
        text, _h = emboss_label(obb, line, text_centre, along, up, height=letter, raised=0.6)
        if text is not None:
            parts.append(text)
        if k == len(lines) - 1:
            arrow_centre = text_centre + (0.5 * line_units + 0.9) * letter * along
            arrow, _h = emboss_label(obb, ">", arrow_centre, along, up, height=letter, raised=0.6)
            if arrow is not None:
                parts.append(arrow)
    append = vtk.vtkAppendPolyData()
    for part in parts:
        append.AddInputData(part)
    append.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    info = {"length": length, "width": width, "thickness": thickness, "letter": letter}
    # clearance of the plate above the bone (a few rays straight down from its underside)
    if bone_obb is not None:
        points, cells = vtk.vtkPoints(), vtk.vtkIdList()
        clearances = []
        under = plate_centre - 0.5 * thickness * up
        for a in (-0.5, 0.0, 0.5):
            for b in (-0.5, 0.0, 0.5):
                q = under + a * length * along + b * width * side
                points.Reset()
                cells.Reset()
                bone_obb.IntersectWithLine((q + 60.0 * up).tolist(), (q - 60.0 * up).tolist(), points, cells)
                if points.GetNumberOfPoints():
                    heights = [float((np.asarray(points.GetPoint(i)) - q) @ up) for i in range(points.GetNumberOfPoints())]
                    clearances.append(-max(heights))     # > 0: bone below the plate, < 0: bone in the plate
        if clearances:
            info["bone_clearance"] = float(min(clearances))
    return out, info


def _stencil_mask(polydata, origin, spacing, dims):
    """Voxel mask (k, j, i) of a CLOSED surface on the given grid."""
    image = vtk.vtkImageData()
    image.SetOrigin(*[float(x) for x in origin])
    image.SetSpacing(spacing, spacing, spacing)
    image.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(1)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(polydata)
    stencil.SetOutputOrigin(image.GetOrigin())
    stencil.SetOutputSpacing(image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()
    to_image = vtk.vtkImageStencil()
    to_image.SetInputData(image)
    to_image.SetStencilConnection(stencil.GetOutputPort())
    to_image.ReverseStencilOff()
    to_image.SetBackgroundValue(0)
    to_image.Update()
    return vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars()).reshape(
        int(dims[2]), int(dims[1]), int(dims[0])) > 0


def _sample_labelmap(labelmap, label_value, origin, spacing, dims):
    """Bone mask (k, j, i) on the guide grid, sampled (nearest voxel) from the segmentation labelmap -
    exact, no surface artefacts."""
    m = vtk.vtkMatrix4x4()
    labelmap.GetImageToWorldMatrix(m)
    M = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])
    Minv = np.linalg.inv(M)
    ext = labelmap.GetExtent()
    ldims = (ext[5] - ext[4] + 1, ext[3] - ext[2] + 1, ext[1] - ext[0] + 1)
    full = vtk_to_numpy(labelmap.GetPointData().GetScalars()).reshape(ldims)
    zs = origin[2] + spacing * np.arange(dims[2])
    ys = origin[1] + spacing * np.arange(dims[1])
    xs = origin[0] + spacing * np.arange(dims[0])
    out = np.zeros((dims[2], dims[1], dims[0]), dtype=bool)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    for k, z in enumerate(zs):                       # slice by slice: modest memory
        ras = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z), np.ones(X.size)])
        ijk = np.rint((Minv @ ras)[:3]).astype(int)
        i, j, kk = ijk[0] - ext[0], ijk[1] - ext[2], ijk[2] - ext[4]
        ok = (i >= 0) & (i < ldims[2]) & (j >= 0) & (j < ldims[1]) & (kk >= 0) & (kk < ldims[0])
        values = np.zeros(X.size, dtype=bool)
        values[ok] = full[kk[ok], j[ok], i[ok]] == label_value
        out[k] = values.reshape(X.shape)
    return out


def clean_guide(guide, bone, gap, voxel=0.15, min_island=20.0, smooth=True, bone_labelmap=None,
                label_value=1, fill_axis=None):
    """Rebuild a guide as ONE clean, closed, manifold solid:
    all parts (block, footprint body, sleeves, label - also overlapping ones) are united in a voxel grid,
    the bone - as a SOLID (medullary canal and pores filled), grown by the fit gap - is cut away, so the
    whole contact surface becomes an even offset of the bone surface; loose islands below `min_island`
    mm^3 are dropped; the surface is re-created (no single triangles, no broken edges).
    Returns (polydata, info)."""
    import scipy.ndimage as ndi
    from vtk.util.numpy_support import numpy_to_vtk as _n2v
    if guide is None or guide.GetNumberOfPoints() == 0:
        raise ValueError("clean_guide: the guide is empty")
    b = guide.GetBounds()
    pad = 2.0 + float(gap)
    origin = np.array([b[0] - pad, b[2] - pad, b[4] - pad])
    dims = np.ceil((np.array([b[1], b[3], b[5]]) + pad - origin) / voxel).astype(int) + 1
    # 1) union of all parts (each part voxelised on its own: overlapping shells would cancel out)
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(guide)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    n_parts = conn.GetNumberOfExtractedRegions()
    solid = np.zeros((dims[2], dims[1], dims[0]), dtype=bool)
    pick = vtk.vtkPolyDataConnectivityFilter()
    pick.SetInputData(guide)
    pick.SetExtractionModeToSpecifiedRegions()
    for region in range(n_parts):
        pick.InitializeSpecifiedRegionList()
        pick.AddSpecifiedRegion(region)
        pick.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(pick.GetOutput())
        clean.Update()
        if clean.GetOutput().GetNumberOfCells() < 4:
            continue
        solid |= _stencil_mask(clean.GetOutput(), origin, voxel, dims)
    # 2) the bone as a solid in the same grid, grown by the fit gap, cut away
    removed_bone = 0
    if bone is not None and bone.GetNumberOfPoints():
        box = vtk.vtkBox()
        box.SetBounds(origin[0] - 5, origin[0] + dims[0] * voxel + 5, origin[1] - 5, origin[1] + dims[1] * voxel + 5,
                      origin[2] - 5, origin[2] + dims[2] * voxel + 5)
        if bone_labelmap is not None:
            bone_mask = _sample_labelmap(bone_labelmap, label_value, origin, voxel, dims)
        else:
            bone_mask = _stencil_mask(bone, origin, voxel, dims)
        bone_mask = ndi.binary_closing(bone_mask, iterations=2) | bone_mask
        # pores / medullary canal: filled slice by slice ACROSS THE SHAFT only (slices along other
        # directions could close pockets between femur and patella / fabellae) and in 3D
        axes = [int(np.argmax(np.abs(np.asarray(fill_axis, float))))] if fill_axis is not None else [2]
        for axis_xyz in axes:
            axis = 2 - axis_xyz                     # grid (k, j, i) <-> (z, y, x)
            for index in range(bone_mask.shape[axis]):
                sl = [slice(None)] * 3
                sl[axis] = index
                face = bone_mask[tuple(sl)]
                if face.any():
                    bone_mask[tuple(sl)] = ndi.binary_fill_holes(face)
        bone_mask = ndi.binary_fill_holes(bone_mask)
        if float(gap) > 0:
            grown = ndi.distance_transform_edt(~bone_mask) * voxel <= float(gap)
        else:
            grown = bone_mask
        removed_bone = int((solid & grown).sum())
        solid &= ~grown
    # 3) drop loose islands (single voxels / slivers)
    labels, count = ndi.label(solid)
    dropped = 0
    if count > 1:
        sizes = ndi.sum(solid, labels, index=np.arange(1, count + 1)) * voxel ** 3
        keep = np.zeros(count + 1, dtype=bool)
        keep[1:] = sizes >= float(min_island)
        if not keep.any():
            keep[1 + int(np.argmax(sizes))] = True
        dropped = int((~keep[1:]).sum())
        solid = keep[labels]
    # 4) new surface: slightly blurred mask -> iso-surface (closed, manifold, no stair steps)
    field = ndi.gaussian_filter(solid.astype(np.float32), 0.6) if smooth else solid.astype(np.float32)
    image = vtk.vtkImageData()
    image.SetOrigin(*[float(x) for x in origin])
    image.SetSpacing(voxel, voxel, voxel)
    image.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    image.GetPointData().SetScalars(_n2v(field.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
    iso = vtk.vtkFlyingEdges3D()
    iso.SetInputData(image)
    iso.SetValue(0, 0.5)
    iso.ComputeNormalsOff()
    iso.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(iso.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputConnection(clean.GetOutputPort())
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.SplittingOff()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    # iso-surface specks (a few triangles around single blurred voxels): keep only real bodies
    regions = vtk.vtkPolyDataConnectivityFilter()
    regions.SetInputData(out)
    regions.SetExtractionModeToAllRegions()
    regions.Update()
    sizes = vtk_to_numpy(regions.GetRegionSizes())
    if len(sizes) == 0:
        raise ValueError("clean_guide: nothing left of the guide (it lies completely in the bone + fit gap)")
    keep_ids = [int(r) for r in np.where(sizes >= max(500, 0.01 * sizes.max()))[0]]
    if len(keep_ids) < len(sizes):
        dropped += len(sizes) - len(keep_ids)
        pick = vtk.vtkPolyDataConnectivityFilter()
        pick.SetInputData(out)
        pick.SetExtractionModeToSpecifiedRegions()
        for r in keep_ids:
            pick.AddSpecifiedRegion(r)
        clean2 = vtk.vtkCleanPolyData()
        clean2.SetInputConnection(pick.GetOutputPort())
        clean2.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(clean2.GetOutput())
    if open_edge_count(out) > 0:                  # close the odd tiny gap (merged coincident points)
        fill = vtk.vtkFillHolesFilter()
        fill.SetInputData(out)
        fill.SetHoleSize(2.0)
        fill.Update()
        nrm2 = vtk.vtkPolyDataNormals()
        nrm2.SetInputConnection(fill.GetOutputPort())
        nrm2.ConsistencyOn()
        nrm2.AutoOrientNormalsOn()
        nrm2.SplittingOff()
        nrm2.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(nrm2.GetOutput())
    regions = vtk.vtkPolyDataConnectivityFilter()
    regions.SetInputData(out)
    regions.SetExtractionModeToAllRegions()
    regions.Update()
    info = dict(parts_before=int(n_parts), bodies_after=int(regions.GetNumberOfExtractedRegions()),
                islands_dropped=dropped, bone_voxels_removed=removed_bone,
                open_edges=open_edge_count(out), volume_mm3=float(solid.sum()) * voxel ** 3)
    return out, info


# ---------------------------------------------------------------------------------------------
# Volume guide body: a shell built from the DISTANCE FIELD of the bone around the painted contact
# region - grows away from the bone in every direction (several sides of the bone at once), the
# inner surface follows the bone exactly at the fit gap, closed by construction (one iso-surface).
# ---------------------------------------------------------------------------------------------

def _poly_to_local(poly, R):
    m = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            m.SetElement(r, c, float(R[r, c]))
    tf = vtk.vtkTransform()
    tf.SetMatrix(m)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputData(poly)
    f.SetTransform(tf)
    f.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(f.GetOutput())
    return out


def _rasterize_closed(poly_local, origin, voxel, dims):
    """bool[x, y, z]: voxels inside the closed surface poly_local (local frame)."""
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly_local)
    stencil.SetOutputOrigin(*[float(v) for v in origin])
    stencil.SetOutputSpacing(voxel, voxel, voxel)
    stencil.SetOutputWholeExtent(0, int(dims[0]) - 1, 0, int(dims[1]) - 1, 0, int(dims[2]) - 1)
    stencil.Update()
    conv = vtk.vtkImageStencilToImage()
    conv.SetInputConnection(stencil.GetOutputPort())
    conv.SetInsideValue(1)
    conv.SetOutsideValue(0)
    conv.SetOutputScalarTypeToUnsignedChar()
    conv.Update()
    arr = vtk_to_numpy(conv.GetOutput().GetPointData().GetScalars())
    return arr.reshape(int(dims[2]), int(dims[1]), int(dims[0])).transpose(2, 1, 0) > 0


def _smin(a, b, k):
    """Smooth minimum (rounded union) with blend radius k."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _smax(a, b, k):
    """Smooth maximum (rounded intersection) with blend radius k."""
    return -_smin(-a, -b, k)


def cut_bridges(slots, radius=1.75, reach_into=6.0, leg=7.0):
    """Four round bridges across the cut(s): at both ends of the kerf (just outside it) and at two
    heights (close to the bone and near the top of the saw block). They run perpendicular to the
    cutting plane(s) and reach `reach_into` mm into the guide on both sides, so the two halves of the
    guide are held at four points and cannot twist or bend at the cut. Each bridge ends in two legs
    that run back over the guide on both sides (U-shape), so it is anchored in the body even where the
    painted region does not reach the end of the kerf. Returns capsule segments [(P0, P1), ...] (world)."""
    slots = [s for s in (slots or []) if "error" not in s]
    if not slots:
        return []
    s0 = slots[0]
    C0, n0, u0v, v0 = (np.asarray(s0[k], float) for k in ("centre", "normal", "u", "v"))
    us, ts = [], []
    for s in slots:
        C = np.asarray(s["centre"], float)
        u = np.asarray(s["u"], float)
        for uu in (float(s["u0"]), float(s["u1"])):
            us.append(float((C + uu * u - C0) @ u0v))
        ts.append(float((C - C0) @ n0))
    u_lo, u_hi = min(us) - 0.5 - radius - 0.2, max(us) + 0.5 + radius + 0.2
    half = 0.5 * SLOT_WIDTH + SLOT_WALL
    t_lo, t_hi = min(ts) - half - reach_into, max(ts) + half + reach_into
    bars = []
    for u_end in (u_lo, u_hi):
        # low: just above the block bottom (bone + gap) at this end; high: below the block top
        lows, tops = [], []
        for s in slots:
            C = np.asarray(s["centre"], float)
            uu = float((C0 + u_end * u0v - C) @ np.asarray(s["u"], float))
            ubins = np.asarray(s["ubins"], float)
            bottom = np.asarray(s["bottom"], float)
            # beyond the bone edge the block bottom is only a depth limit - take the bone height at the
            # nearest bin that really lies over bone (the edge of the bone at this end)
            on_bone = bottom > float(s["v_top"]) - BLOCK_MAX_DEPTH + 0.01
            if on_bone.any():
                k = int(np.argmin(np.where(on_bone, np.abs(ubins - uu), np.inf)))
                lows.append(float(bottom[k]) + float((C - C0) @ v0))
            tops.append(float(s["v_top"]) + float((C - C0) @ v0))
        if not lows:
            lows = [min(tops) - 8.0]
        v_low = max(lows) + radius + 0.3
        v_high = max(min(tops) - radius - 0.3, v_low + 2.0 * radius + 0.5)
        inward = -np.sign(u_end - 0.5 * (u_lo + u_hi)) * leg     # towards the middle of the cut
        for vv in (v_low, v_high):
            base = C0 + u_end * u0v + vv * v0
            a, b = base + t_lo * n0, base + t_hi * n0
            bars.append((a, b))                                   # across the cut, outside the kerf
            bars.append((a, a + inward * u0v))                    # legs into the guide on both sides
            bars.append((b, b + inward * u0v))
    return bars


def _remove_thin_features(F, r, voxel):
    """Morphological opening of the field F (< 0 inside, ~ signed distance in mm) with a ball of
    radius r: parts thinner than 2 r vanish, everything else keeps its shape (convex edges get at
    most an r rounding). Sub-voxel: erosion = F + r; the dilation back is measured from the eroded
    boundary voxels including their own field value. Returns (F_new, removed volume mm3)."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    inside = F < 0
    if not inside.any():
        return F, 0.0
    # the field is only roughly a distance (smooth unions, the smoothed rim region ...): near the
    # surface it is normalised by its gradient (first-order distance), so erosion/dilation are in mm
    layer_idx = np.flatnonzero((F < voxel) & (F > -(r + 3.0 * voxel)))
    li = np.stack(np.unravel_index(layer_idx, F.shape), axis=1)
    grad2 = np.zeros(len(layer_idx), np.float32)
    for ax in range(3):
        hi_ = li.copy()
        lo_ = li.copy()
        hi_[:, ax] = np.minimum(hi_[:, ax] + 1, F.shape[ax] - 1)
        lo_[:, ax] = np.maximum(lo_[:, ax] - 1, 0)
        # one-sided slopes, the steeper one: also correct ON a ridge (thin wall: the central
        # difference would be zero there)
        f0 = F[tuple(li.T)]
        slope = np.maximum(np.abs(F[tuple(hi_.T)] - f0), np.abs(f0 - F[tuple(lo_.T)])) / voxel
        grad2 += slope ** 2
    gmag = np.ones(F.shape, np.float32)
    gmag.ravel()[layer_idx] = np.clip(np.sqrt(grad2), 0.2, 1.0)   # > 1 only at kinks (carved faces): keep F
    Fn = F / gmag
    Fe = Fn + np.float32(r)
    E = Fe < 0
    cand = inside & ~E
    if not cand.any():
        return F, 0.0
    cand_idx = np.flatnonzero(cand)
    eb = E & ~ndi.binary_erosion(E)
    eb_idx = np.flatnonzero(eb)
    G = np.full(len(cand_idx), np.float32(voxel))           # no eroded core nearby: removed
    if len(eb_idx):
        Pe = np.stack(np.unravel_index(eb_idx, F.shape), axis=1).astype(np.float32) * voxel
        Pc = np.stack(np.unravel_index(cand_idx, F.shape), axis=1).astype(np.float32) * voxel
        k = min(8, len(eb_idx))
        dist, idx = cKDTree(Pe).query(Pc, k=k, distance_upper_bound=r + 2.5 * voxel, workers=-1)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        ok = idx < len(eb_idx)
        fe = np.where(ok, Fe.ravel()[eb_idx[np.minimum(idx, len(eb_idx) - 1)]], 0.0)
        val = np.where(ok, dist + fe, np.inf).min(axis=1) - r
        G = np.where(np.isfinite(val), val, voxel).astype(np.float32)
    old = F.ravel()[cand_idx]
    fn_c = Fn.ravel()[cand_idx]
    # the reconstruction from voxel samples overestimates by a few 1/100 mm on any face - such small
    # differences are not thin features: the face keeps its exact field (fit surface stays exact)
    new = np.where(G > fn_c + 0.08, G, fn_c) * gmag.ravel()[cand_idx]
    removed = float(np.count_nonzero((old < 0) & (new >= 0))) * voxel ** 3
    F = F.copy()
    F.ravel()[cand_idx] = new
    return F, removed


def _smooth_keep_fit_faces(surface, bone_samples, gap, pins, slots, block, R, half, iterations=25):
    """Windowed-sinc smoothing of the extracted guide surface (local frame): removes voxel steps
    and ripples on the outside, rims and sleeve edges without shrinking the part. Vertices of the
    FITTING faces - the bone contact (within the fit gap), the pin bores and the kerf faces - keep
    their exact original position."""
    from scipy.spatial import cKDTree
    V = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
    protect = np.zeros(len(V), dtype=bool)
    d_bone = cKDTree(bone_samples).query(V, distance_upper_bound=float(gap) + 1.0, workers=-1)[0]
    protect |= d_bone <= float(gap) + 0.3
    for pin in pins:
        if "error" in pin:
            continue
        a = R @ unit(np.asarray(pin["axis"], float))
        E = R @ np.asarray(pin["entry"], float)
        r = V - E
        along = r @ a
        radial = np.linalg.norm(r - np.outer(along, a), axis=1)
        protect |= radial <= float(pin["r_in"]) + 0.25
    for s in slots:
        C = R @ np.asarray(s["centre"], float)
        n = R @ np.asarray(s["normal"], float)
        protect |= np.abs((V - C) @ n) <= half + 0.25
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(int(iterations))
    smoother.SetPassBand(0.05)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    smoothed = vtk.vtkPolyData()
    smoothed.DeepCopy(smoother.GetOutput())
    if smoothed.GetNumberOfPoints() != len(V):
        return surface                                    # (never expected) keep the unsmoothed one
    W = vtk_to_numpy(smoothed.GetPoints().GetData()).astype(float).copy()
    W[protect] = V[protect]
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(W), deep=True))
    smoothed.SetPoints(points)
    return smoothed


# --- bone fields of the SDF guide body: cached between builds --------------------------------------
# The expensive part of a guide build (bone stencil, pit closing, distance transforms, exact band
# distances, smoothed outer distance) depends only on the bone, the guide side and gap / wall - not on
# the painted region, the pins, the saw block or the kerfs. It is computed once on a slightly larger
# lattice block and reused by every later build whose grid lies inside it (pins moved, saw box
# dragged, region brushed a little).
_BONE_FIELDS = []                 # [(key, lattice index of the first voxel, shape, fields)] - newest first
_BONE_FIELDS_MAX = 2              # saw guide + reduction guide
_BONE_FIELD_PAD_MM = 3.0


def _poly_fingerprint(poly):
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    return (int(poly.GetNumberOfPoints()), int(poly.GetNumberOfCells()),
            round(float(pts.sum()), 3), round(float((pts * pts).sum()), 1))


def _bone_fields_cached(bone, R, voxel, i_lo, shape, gap, reach, rim_round, outer_smooth, pit_close):
    """-> (fields, offset of the requested block inside the cached block, from_cache)."""
    key = (_poly_fingerprint(bone), tuple(np.round(R, 6).ravel().tolist()), round(float(voxel), 4),
           round(float(gap), 4), round(float(reach), 4), round(float(rim_round), 4),
           round(float(outer_smooth or 0.0), 4), round(float(pit_close or 0.0), 4))
    i_lo = np.asarray(i_lo, int)
    shp = np.asarray(shape, int)
    for k, c_lo, c_shape, fields in _BONE_FIELDS:
        if k == key and np.all(i_lo >= c_lo) and np.all(i_lo + shp <= c_lo + c_shape):
            return fields, i_lo - c_lo, True
    _BONE_FIELDS[:] = [e for e in _BONE_FIELDS if e[0] != key]          # free the memory first
    del _BONE_FIELDS[max(0, _BONE_FIELDS_MAX - 1):]
    pad = int(round(_BONE_FIELD_PAD_MM / voxel))
    c_lo, c_shape = i_lo - pad, shp + 2 * pad
    if np.prod(c_shape.astype(float)) > 45e6:
        c_lo, c_shape = i_lo, shp
    fields = _bone_fields(bone, R, voxel, c_lo, c_shape, gap, reach, rim_round, outer_smooth, pit_close)
    if fields is None:
        return None, None, False
    _BONE_FIELDS.insert(0, (key, c_lo, c_shape, fields))
    return fields, i_lo - c_lo, False


def _bone_fields(bone, R, voxel, i_lo, dims, gap, reach, rim_round, outer_smooth, pit_close):
    """Selection-independent fields of the bone on the lattice block i_lo .. i_lo + dims - 1 (local
    frame R, voxel mm): filled bone mask, signed bone distance db (exact in the band around the
    shell), band voxels with exact / smoothed distances, dense surface samples + normals."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    dims = np.asarray(dims, int)
    origin = np.asarray(i_lo, float) * voxel
    lo, hi = origin, origin + voxel * (dims - 1)
    shape = tuple(int(v) for v in dims)
    gx, gy, gz = ((origin[a] + voxel * np.arange(shape[a])).astype(np.float32) for a in range(3))
    bone_local = _poly_to_local(bone, R)
    box = vtk.vtkBox()
    box.SetBounds(lo[0] - 4, hi[0] + 4, lo[1] - 4, hi[1] + 4, lo[2] - 4, hi[2] + 4)
    crop = vtk.vtkExtractPolyDataGeometry()
    crop.SetInputData(bone_local)
    crop.SetImplicitFunction(box)
    crop.ExtractInsideOn()
    crop.ExtractBoundaryCellsOn()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(crop.GetOutputPort())
    tri.Update()
    bone_crop = vtk.vtkPolyData()
    bone_crop.DeepCopy(tri.GetOutput())
    if bone_crop.GetNumberOfPoints() == 0:
        return None
    bone_mask = _rasterize_closed(bone_local, origin, voxel, dims)
    # the medullary canal leaves the grid at its faces - cap the faces first, then fill cavities
    for ax in range(3):
        for end in (0, -1):
            sl = [slice(None)] * 3
            sl[ax] = end
            bone_mask[tuple(sl)] = ndi.binary_fill_holes(bone_mask[tuple(sl)])
    bone_mask = ndi.binary_fill_holes(bone_mask)
    # pits / pores of the bone surface: closed with a ball of radius pit_close
    pit_mask = None
    pit_db = None
    if pit_close and pit_close > 0:
        rk = max(1, int(round(float(pit_close) / voxel)))
        # closing with a ball by two distance transforms (much faster than binary dilation /
        # erosion with an 11^3 ball)
        padded = np.pad(bone_mask, rk + 1)
        dilated = ndi.distance_transform_edt(~padded) <= rk
        closed = (ndi.distance_transform_edt(dilated) > rk)[
            rk + 1:-(rk + 1), rk + 1:-(rk + 1), rk + 1:-(rk + 1)]
        del dilated, padded
        closed |= bone_mask
        pit_mask = closed & ~bone_mask
        # only REAL pits: the closing also fills the voxel steps of every oblique surface - those are
        # at most ~1 voxel deep; a pit counts when it reaches >= 0.5 mm below the bridged surface
        if pit_mask.any():
            depth_c = ndi.distance_transform_edt(closed) * voxel
            lab, n_lab = ndi.label(pit_mask)
            if n_lab:
                deep = ndi.maximum(depth_c, lab, index=np.arange(1, n_lab + 1))
                keep_lab = np.flatnonzero(np.asarray(deep) >= 0.5) + 1
                pit_mask = np.isin(lab, keep_lab) if len(keep_lab) else np.zeros_like(pit_mask)
                closed = bone_mask | pit_mask
            del depth_c, lab
        if pit_mask.any():
            near = ndi.distance_transform_edt(~pit_mask) * voxel <= float(gap) + float(pit_close) + 1.0
            out_c = np.maximum(ndi.distance_transform_edt(~closed) * voxel - 0.5 * voxel, 0.0)
            pit_db = np.full(shape, np.inf, dtype=np.float32)
            pit_db[near] = out_c[near]
            pit_db[pit_mask] = -voxel
            del near, out_c
        else:
            pit_mask = None
        del closed
    outside = ndi.distance_transform_edt(~bone_mask) * voxel
    depth = ndi.distance_transform_edt(bone_mask) * voxel
    db = np.where(bone_mask, -depth, outside).astype(np.float32)          # approx. signed distance
    del depth
    band = (~bone_mask) & (outside >= max(0.0, gap - 0.8)) & (outside <= reach + rim_round + 0.8)
    del outside
    band_idx = np.flatnonzero(band)
    del band
    ii, jj, kk = np.unravel_index(band_idx, shape)
    Q = np.stack([gx[ii], gy[jj], gz[kk]], axis=1).astype(float)
    # distance to the bone surface: densely resampled surface (<= 0.2 mm edges) + its normals; the
    # distance to the tangent plane of the nearest sample (exact on smooth bone, no voxel steps)
    dense = vtk.vtkAdaptiveSubdivisionFilter()
    dense.SetInputData(bone_crop)
    dense.SetMaximumEdgeLength(0.2)
    dense.SetMaximumNumberOfPasses(12)
    nrm_b = vtk.vtkPolyDataNormals()
    nrm_b.SetInputConnection(dense.GetOutputPort())
    nrm_b.ComputePointNormalsOn()
    nrm_b.SplittingOff()
    nrm_b.ConsistencyOn()
    nrm_b.AutoOrientNormalsOff()
    nrm_b.Update()
    D = vtk_to_numpy(nrm_b.GetOutput().GetPoints().GetData()).astype(float)
    Dn = vtk_to_numpy(nrm_b.GetOutput().GetPointData().GetNormals()).astype(float)
    cap_d = reach + rim_round + 1.5
    dist_v, near = cKDTree(D).query(Q, distance_upper_bound=cap_d, workers=-1)
    found = near < len(D)
    plane = np.full(len(Q), cap_d)
    plane[found] = np.abs(np.einsum("ij,ij->i", Q[found] - D[near[found]], Dn[near[found]]))
    dist_v = np.where(found, dist_v, cap_d)
    db_band = np.maximum(plane, dist_v - 0.15).astype(np.float32)
    del Q, plane, dist_v, near, found
    pits_closed = 0.0
    if pit_mask is not None:
        # over a pit the distance to the bridged surface; elsewhere (difference < 0.15 mm: only the
        # voxel quantisation of the bridged distance) the exact distance stays
        pit_band = pit_db.ravel()[band_idx]
        db_band = np.where(pit_band < db_band - 0.15, pit_band, db_band).astype(np.float32)
        db = np.where(pit_db < db - 0.15, pit_db, db).astype(np.float32)
        pits_closed = float(pit_mask.sum()) * voxel ** 3
        del pit_db, pit_band
    db.ravel()[band_idx] = db_band
    # outer surface: from a smoothed distance (the nearest-sample distance ripples a little far from
    # the bone; the bone surface itself has small steps) - the inner fit surface keeps db_band
    if outer_smooth and outer_smooth > 0:
        db_outer = ndi.gaussian_filter(db, float(outer_smooth) / voxel).ravel()[band_idx].astype(np.float32)
    else:
        db_outer = db_band
    return dict(bone_mask=bone_mask, db=db, band_ijk=np.stack([ii, jj, kk], axis=1).astype(np.int32),
                db_band=db_band, db_outer=db_outer, D=D, Dn=Dn,
                B=vtk_to_numpy(bone_crop.GetPoints().GetData()).astype(float), pits_closed=pits_closed)


def sdf_guide_body(bone, mesh, selected, direction, gap, wall, slots=None, pins=None, block=False,
                   parts=None, bridges=True, bridge_radius=1.75, voxel=None, rim_round=1.0, fillet=1.0,
                   trim_undercut=True, min_feature=0.3, outer_smooth=0.75, pit_close=1.0, sleeves=False,
                   smooth_surface=True):
    """Guide body from the distance field of the bone.
    bone: closed bone surface (whole bone or the fragments); mesh + selected: the painted contact
    region (bool per vertex of the bone-surface ROI mesh); direction: the guide side (insertion
    direction, bone -> guide).
    - shell: every point whose distance to the bone is between `gap` and `gap + wall` and whose
      nearest bone point lies in the painted region - grows away from the bone in all directions,
      the inner surface is the bone offset exactly by the fit gap (distances evaluated on the mesh,
      not on voxels); hollow spaces inside the bone (medullary canal ...) are ignored.
    - rim: rounded (rim_round mm) where the painted region ends.
    - undercuts (trim_undercut): parts of the shell lying under the bone as seen from the guide side
      would block placing the guide - they are cut away.
    - bridges: four round bars across the cut (see cut_bridges), fused with the shell.
    - parts (saw block, sleeves): fused with the shell by rounded fillets (fillet mm).
    - kerfs (or the open wedge of a block guide) and pin bores are kept exactly free.
    - outer_smooth: the outer surface follows a smoothed bone distance (Gaussian, mm) - no ripples
      from the bone surface / the voxel grid; the inner fit surface stays exact.
    - pit_close: pits / pores in the bone surface narrower than 2 x this (mm) are bridged - the
      inner surface does not grow pegs into them (they would block seating and break off).
    - sleeves: the pin sleeves are built IN the field (exact cylinders r_out up to pin['top']): their
      bottom keeps the same fit gap to the bone as the body (normal distance), they are fused with a
      fillet, and every bore is cut through everything - a sleeve standing close to another pin can
      never reach into its bore.
    - voxel: grid spacing (mm); None = 0.25 mm, coarser only when the region is very large.
    - smooth_surface: the extracted surface is smoothed (no voxel steps / ripples on the outside,
      rims and sleeve edges); the fitting faces - bone contact, bores, kerfs - keep their exact
      positions.
    - min_feature: everything thinner than this (mm) - thin edges, slivers, feathered rims - is
      removed (morphological opening of the field; material is only removed, never added).
    Returns (polydata or None, info)."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    t_start = time.time()
    _tm = {}
    d = unit(np.asarray(direction, float))
    e1, e2 = _perp_frame(d)
    R = np.stack([e1, e2, d])                                  # world -> local rows
    P = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    sel = np.asarray(selected, bool)
    if sel.sum() < 3:
        return None, {"error": "the footprint is empty"}
    reach = float(gap) + float(wall)
    S_local = P[sel] @ R.T
    parts = [p for p in (parts or []) if p is not None and p.GetNumberOfPoints() > 0]
    parts_local = [_poly_to_local(p, R) for p in parts]
    bars = [(R @ a, R @ b) for a, b in (cut_bridges(slots, bridge_radius) if bridges else [])]
    extent_pts = [S_local] + [vtk_to_numpy(p.GetPoints().GetData()).astype(float) for p in parts_local]
    extent_pts += [np.array(bar) for bar in bars]
    if sleeves:
        extent_pts += [np.array([R @ np.asarray(p_["top"], float), R @ np.asarray(p_["entry"], float)])
                       for p_ in (pins or []) if "error" not in p_]
    allp = np.vstack(extent_pts)
    margin = reach + rim_round + bridge_radius + 2.0
    lo = allp.min(axis=0) - margin
    hi = allp.max(axis=0) + margin
    if voxel is None:
        # fine grid for a clean surface; coarser only for very large guides (memory / time)
        voxel = 0.25
        while np.prod(np.ceil((hi - lo) / voxel) + 1) > 25e6 and voxel < 0.35:
            voxel += 0.05
    voxel = float(voxel)
    # the grid sits on a fixed lattice (multiples of voxel in the local frame): a later build whose
    # box lies inside an earlier one reuses its bone fields on exactly the same voxels
    i_lo = np.floor(lo / voxel).astype(int)
    origin = i_lo * voxel
    dims = np.ceil((hi - origin) / voxel).astype(int) + 1
    if np.prod(dims) > 45e6:
        return None, {"error": f"region too large for the voxel grid ({dims.tolist()}) - make the footprint smaller"}
    shape = tuple(int(v) for v in dims)
    # float32 grid coordinates: every field below stays float32 (half the memory of float64)
    gx, gy, gz = ((origin[a] + voxel * np.arange(shape[a])).astype(np.float32) for a in range(3))

    def sub_box(points, margin_mm):
        """Index slices of the grid box around `points` (local mm) widened by margin_mm - analytic
        parts (bridges, sleeves) are evaluated only there, not on the whole grid."""
        pts = np.atleast_2d(np.asarray(points, float))
        lo_i = np.floor((pts.min(axis=0) - margin_mm - origin) / voxel).astype(int)
        hi_i = np.ceil((pts.max(axis=0) + margin_mm - origin) / voxel).astype(int) + 1
        lo_i = np.clip(lo_i, 0, np.array(shape))
        hi_i = np.clip(hi_i, 0, np.array(shape))
        if np.any(hi_i <= lo_i):
            return None
        return tuple(slice(int(l), int(h)) for l, h in zip(lo_i, hi_i))
    # --- bone: filled solid (cavities closed) + exact distance in the band around the shell -----
    bf, off, from_cache = _bone_fields_cached(bone, R, voxel, i_lo, shape, gap, reach, rim_round,
                                              outer_smooth, pit_close)
    if bf is None:
        return None, {"error": "no bone surface around the footprint"}
    _tm['bone_fields'] = round(time.time() - t_start, 1)
    _tm['cached'] = from_cache
    sl = tuple(slice(int(o), int(o) + s) for o, s in zip(off, shape))
    bone_mask = bf["bone_mask"][sl]
    db = bf["db"][sl]
    ijk = bf["band_ijk"] - off.astype(np.int32)
    inside = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
    ii, jj, kk = ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]
    Q = np.stack([gx[ii], gy[jj], gz[kk]], axis=1).astype(float)
    # only the band around the painted region is needed
    tree_s = cKDTree(S_local)
    dS = tree_s.query(Q, distance_upper_bound=reach + rim_round + 3.0, workers=-1)[0]
    keep = np.isfinite(dS)
    band_idx = np.ravel_multi_index((ii[keep], jj[keep], kk[keep]), shape)
    Q, dS = Q[keep], dS[keep]
    db_band = bf["db_band"][inside][keep]
    db_outer = bf["db_outer"][inside][keep]
    D, B = bf["D"], bf["B"]
    pits_closed = bf["pits_closed"]
    del ijk, inside, ii, jj, kk, keep
    t_dist = time.time() - t_start
    # --- painted region: is the nearest bone point painted? ------------------------------------
    spacing = float(np.median(tree_s.query(S_local, k=2)[0][:, 1])) if len(S_local) > 2 else 0.5
    near_sel = tree_s.query(B, distance_upper_bound=2.0 * spacing + 0.3, workers=-1)[0] < np.inf
    U = np.vstack([B[~near_sel], P[~sel] @ R.T])
    tree_u = cKDTree(U) if len(U) else None
    dU = tree_u.query(Q, distance_upper_bound=reach + rim_round + 6.0, workers=-1)[0] if tree_u is not None \
        else np.full(len(Q), np.inf)
    dU = np.minimum(dU, reach + rim_round + 6.0)
    f_reg = (0.5 * (dS - dU)).astype(np.float32)                         # < 0 inside the region
    # the painted rim follows the triangles - smooth the region field (0.5 mm) for an even rim
    reg3 = np.full(shape, reach, dtype=np.float32)
    reg3.ravel()[band_idx] = f_reg
    reg3 = ndi.gaussian_filter(reg3, 0.5 / voxel)
    f_reg = reg3.ravel()[band_idx]
    del reg3
    # --- shell -----------------------------------------------------------------------------------
    f_inner = gap - db_band
    f_outer = db_outer - reach
    shell_band = np.maximum(f_inner, _smax(f_outer, f_reg, rim_round))
    F = np.ones(shape, dtype=np.float32)
    F.ravel()[band_idx] = shell_band
    del Q, f_inner, f_outer, f_reg, shell_band
    # --- bridges across the cut ----------------------------------------------------------------
    X = gx[:, None, None]
    Y = gy[None, :, None]
    Z = gz[None, None, :]
    keep_across = np.zeros(shape, dtype=bool)        # bridges / saw block: allowed to span the cut plane
    for a, b in bars:
        # outside this box the capsule is farther than radius + fillet + 1.5 mm: smin leaves F unchanged
        box = sub_box([a, b], bridge_radius + fillet + 1.5)
        if box is None:
            continue
        a32, ba = a.astype(np.float32), (b - a).astype(np.float32)
        bb = float(ba @ ba)
        px, py, pz = X[box[0]] - a32[0], Y[:, box[1]] - a32[1], Z[:, :, box[2]] - a32[2]
        h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / np.float32(bb), 0.0, 1.0)
        cap = np.sqrt((px - ba[0] * h) ** 2 + (py - ba[1] * h) ** 2 + (pz - ba[2] * h) ** 2) - np.float32(bridge_radius)
        F[box] = _smin(F[box], cap, fillet)
        keep_across[box] |= cap < 0.3
    # --- undercuts: nothing below the bone as seen from the guide side ------------------------
    trimmed = 0.0
    if trim_undercut:
        has = bone_mask.any(axis=2)
        top_k = shape[2] - 1 - np.argmax(bone_mask[:, :, ::-1], axis=2)
        top_z = np.where(has, gz[top_k], -1e9).astype(np.float32)
        f_uc = top_z[:, :, None] - Z.astype(np.float32)
        trimmed = float(((F < 0) & (f_uc > 0)).sum()) * voxel ** 3
        F = np.maximum(F, f_uc)
        del f_uc
    # --- saw block and sleeves fused with fillets -----------------------------------------------
    if parts_local:
        pmask = np.zeros(shape, dtype=bool)
        for p in parts_local:
            pmask |= _rasterize_closed(p, origin, voxel, dims)
        # the exact part meshes are part of the guide as well: their voxel copy stays 0.3 mm INSIDE
        # them (otherwise its voxel steps poke through the exact faces as a rough skin)
        psdf = (ndi.distance_transform_edt(~pmask) - ndi.distance_transform_edt(pmask)) * voxel + 0.3
        F = _smin(F, psdf.astype(np.float32), fillet)
        keep_across |= pmask
        del psdf, pmask
        # saw block ends: everything within the thickness of the block is trimmed flush with its two
        # end faces - the body following the bone does not stick out unevenly beyond them
        for p_world, p_loc in zip(parts, parts_local):
            fr = p_world.GetFieldData().GetArray(BLOCK_FRAME_ARRAY) if p_world.GetFieldData() else None
            if fr is None or fr.GetNumberOfTuples() < 9:
                continue
            vals = np.array([fr.GetValue(k) for k in range(9)], float)
            o_l, eu_l, ew_l = R @ vals[0:3], R @ vals[3:6], R @ vals[6:9]
            PL = vtk_to_numpy(p_loc.GetPoints().GetData()).astype(float) - o_l
            a_lo_, a_hi_ = float((PL @ eu_l).min()), float((PL @ eu_l).max())
            w_lo_, w_hi_ = float((PL @ ew_l).min()), float((PL @ ew_l).max())
            o32, eu32, ew32 = o_l.astype(np.float32), eu_l.astype(np.float32), ew_l.astype(np.float32)
            aa = (X - o32[0]) * eu32[0] + (Y - o32[1]) * eu32[1] + (Z - o32[2]) * eu32[2]
            ww = (X - o32[0]) * ew32[0] + (Y - o32[1]) * ew32[1] + (Z - o32[2]) * ew32[2]
            beyond = np.maximum(np.float32(a_lo_) - aa, aa - np.float32(a_hi_))
            in_slab = np.maximum(np.float32(w_lo_) - ww, ww - np.float32(w_hi_)) <= 0.0
            F = np.where(in_slab, np.maximum(F, beyond), F).astype(np.float32)
            del aa, ww, beyond, in_slab
    # --- pin sleeves as exact cylinders (bottom trimmed below by the fit gap) ------------------------
    n_sleeves = 0
    if sleeves:
        for pin in (pins or []):
            if "error" in pin:
                continue
            a = R @ unit(np.asarray(pin["axis"], float))
            E = R @ np.asarray(pin["entry"], float)
            t_top = float((R @ np.asarray(pin["top"], float) - E) @ a)
            r_out = float(pin["r_out"])
            box = sub_box([E - 3.0 * a, E + t_top * a], r_out + fillet + 1.5)
            if box is None:
                continue
            a32, E32 = a.astype(np.float32), E.astype(np.float32)
            px, py, pz = X[box[0]] - E32[0], Y[:, box[1]] - E32[1], Z[:, :, box[2]] - E32[2]
            along = px * a32[0] + py * a32[1] + pz * a32[2]
            radial = np.sqrt(np.maximum(px ** 2 + py ** 2 + pz ** 2 - along ** 2, 0.0))
            f_s = np.maximum(np.maximum(radial - np.float32(r_out), along - np.float32(t_top)), np.float32(-3.0) - along)
            F[box] = _smin(F[box], f_s, fillet)
            n_sleeves += 1
            del px, py, pz, along, radial, f_s
    # --- keep the bone + fit gap, the kerfs / open wedge and the pin bores free --------------------
    F = np.maximum(F, gap - db)
    good = [s for s in (slots or []) if "error" not in s]

    def plane_dist(C, v):
        C = (R @ np.asarray(C, float)).astype(np.float32)
        v = (R @ np.asarray(v, float)).astype(np.float32)
        return (X - C[0]) * v[0] + (Y - C[1]) * v[1] + (Z - C[2]) * v[2]
    half = 0.5 * SLOT_WIDTH
    if block and len(good) == 2:
        s_free = None
        for s in good:
            side = float(s.get("side") or 1.0)
            term = -half - side * plane_dist(s["centre"], s["normal"])
            s_free = term if s_free is None else np.maximum(s_free, term)
        uu = plane_dist(good[0]["centre"], good[0]["u"])
        s_free = np.maximum(s_free, np.maximum(float(good[0]["u0"]) - 0.5 - uu, uu - float(good[0]["u1"]) - 0.5))
        F = np.maximum(F, -s_free)
    else:
        for s in good:
            uu = plane_dist(s["centre"], s["u"])
            across = np.abs(plane_dist(s["centre"], s["normal"])) - half
            s_free = np.maximum(across, np.maximum(float(s["u0"]) - 0.5 - uu, uu - float(s["u1"]) - 0.5))
            F = np.maximum(F, -s_free)
            # beyond the slot ends the kerf plane is kept clean as well (no shell remnants hugging the
            # bone flank in the cut) - only the saw block (its end caps) and the bridges span the cut
            F = np.where(keep_across, F, np.maximum(F, -across)).astype(np.float32)
    for pin in (pins or []):
        if "error" in pin:
            continue
        a = (R @ unit(np.asarray(pin["axis"], float))).astype(np.float32)
        E = (R @ np.asarray(pin["entry"], float)).astype(np.float32)
        px, py, pz = X - E[0], Y - E[1], Z - E[2]
        along = px * a[0] + py * a[1] + pz * a[2]
        radial = np.sqrt(np.maximum(px ** 2 + py ** 2 + pz ** 2 - along ** 2, 0.0))
        # exact bore (pin + PIN_CLEARANCE); + 0.01 mm: the polygonised bore runs ~0.01 mm inside the field
        F = np.maximum(F, np.float32(float(pin["r_in"]) + 0.01) - radial)
        del px, py, pz, along, radial
    F = F.astype(np.float32, copy=False)
    thin_removed = 0.0
    if min_feature and float(min_feature) > 0:
        F, thin_removed = _remove_thin_features(F, 0.5 * float(min_feature), voxel)
        _tm['thin'] = round(time.time() - t_start, 1)
    F[0, :, :] = F[-1, :, :] = 1.0
    F[:, 0, :] = F[:, -1, :] = 1.0
    F[:, :, 0] = F[:, :, -1] = 1.0
    volume = float((F < 0).sum()) * voxel ** 3
    if volume <= 0.0:
        return None, {"error": "no guide material left (footprint too small, or everything lies in the "
                               "bone / the kerfs / the bores)"}
    # --- surface ---------------------------------------------------------------------------------
    image = vtk.vtkImageData()
    image.SetDimensions(*shape)
    image.SetOrigin(*[float(v) for v in origin])
    image.SetSpacing(voxel, voxel, voxel)
    image.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray((-F).transpose(2, 1, 0)).ravel(),
                                                 deep=True, array_type=vtk.VTK_FLOAT))
    iso = vtk.vtkFlyingEdges3D()
    iso.SetInputData(image)
    iso.SetValue(0, 0.0)
    iso.ComputeNormalsOff()
    iso.Update()
    surface = iso.GetOutput()
    if smooth_surface and surface.GetNumberOfPoints() > 0:
        surface = _smooth_keep_fit_faces(surface, D, gap, pins or [], good, block, R, half)
    out = _poly_to_local(surface, R.T)                                # back to world
    regions = vtk.vtkPolyDataConnectivityFilter()
    regions.SetInputData(out)
    regions.SetExtractionModeToAllRegions()
    regions.Update()
    n_regions = int(regions.GetNumberOfExtractedRegions())
    # keep every piece of real size (a guide that falls apart must be SEEN, not silently halved);
    # crumbs (< 3 % of the largest piece) are dropped
    sizes = vtk_to_numpy(regions.GetRegionSizes()).astype(float) if n_regions else np.zeros(0)
    keep_ids = [int(i) for i in np.flatnonzero(sizes >= 0.03 * sizes.max())] if n_regions else []
    if not keep_ids:
        return None, {"error": "the guide surface could not be extracted"}
    pick = vtk.vtkPolyDataConnectivityFilter()
    pick.SetInputData(out)
    pick.SetExtractionModeToSpecifiedRegions()
    for i in keep_ids:
        pick.AddSpecifiedRegion(i)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(pick.GetOutputPort())
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputConnection(clean.GetOutputPort())
    nrm.ConsistencyOn()
    nrm.AutoOrientNormalsOn()
    nrm.SplittingOff()
    nrm.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(nrm.GetOutput())
    result.GetPointData().SetScalars(None)
    return result, dict(volume_mm3=volume, islands_dropped=max(0, n_regions - len(keep_ids)),
                        pieces=len(keep_ids), bridges=len(bars) // 3,
                        undercut_trimmed_mm3=trimmed, thin_removed_mm3=thin_removed, pits_closed_mm3=pits_closed, sleeves=n_sleeves, open_edges=open_edge_count(result), grid=list(shape),
                        t_distance=round(t_dist, 1), timing=_tm, t_total=round(time.time() - t_start, 1))
