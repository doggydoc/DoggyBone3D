# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""bone_split - separate the bones of ONE segment (femur and tibia touching in the knee) with the
landmarks as seeds (numpy / scipy / SimpleITK, no Slicer / Qt).

split_watershed (used when the CT is at hand): marker watershed on the inverted CT inside the bone
mask - the bones are flooded from their landmarks along the bright bone, the dim joint space (the
partial-volume bridges between femur and tibia) is reached last and becomes the border. Pieces of a
bone that hold none of its landmarks (patella, fabellae, debris) are dropped.

split (fallback without CT): the bone mask is eroded step by step (0, 0.4, 0.7 ... mm) until the seed landmarks of the bones lie in
different connected pieces; those pieces are grown back competitively inside the original mask, so
the thin partial-volume bridges of the joint are split and every bone keeps its full surface. Pieces
without a seed (patella, fabellae, debris) are dropped; thin parts that vanished in the erosion (e.g.
the fibula) are given to the bone they are attached to."""

import numpy as np

try:
    import scipy.ndimage as ndi
except ImportError:                                        # Slicer without scipy
    ndi = None

EROSION_STEPS_MM = (0.0, 0.4, 0.7, 1.0, 1.4, 1.8, 2.3, 3.0)
SNAP_MM = 8.0


def _snap(mask, ijk, spacing_kji, radius_mm=SNAP_MM):
    """Nearest voxel of mask to ijk (k, j, i order) within radius_mm, or None."""
    centre = np.round(np.asarray(ijk, float)).astype(int)
    r = np.ceil(radius_mm / spacing_kji).astype(int)
    lo = np.maximum(centre - r, 0)
    hi = np.minimum(centre + r + 1, mask.shape)
    if np.any(hi <= lo):
        return None
    sub = np.argwhere(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    if len(sub) == 0:
        return None
    d = (((sub + lo) - np.asarray(ijk, float)) * spacing_kji) ** 2
    best = int(np.argmin(d.sum(1)))
    if d[best].sum() > radius_mm ** 2:
        return None
    return tuple(sub[best] + lo)


def split(mask, spacing_kji, seeds):
    """mask: bool array (k, j, i); spacing_kji: mm per voxel; seeds: {bone: [(k, j, i) float, ...]}.
    Returns (labels uint8 with 1..n in the order of seeds, info) - info["separated"] False when the bones
    could not be told apart (then every bone gets the pieces of its seeds, shared pieces included)."""
    if ndi is None:
        raise RuntimeError("scipy is not available in this Slicer")
    names = [name for name, pts in seeds.items() if pts]
    spacing_kji = np.asarray(spacing_kji, float)
    nz = np.argwhere(mask)
    if len(nz) == 0 or not names:
        return np.zeros(mask.shape, np.uint8), {"separated": False, "erosion_mm": None, "bones": names}
    pad = 2
    lo = np.maximum(nz.min(0) - pad, 0)
    hi = np.minimum(nz.max(0) + pad + 1, mask.shape)
    crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    local = {name: [np.asarray(p, float) - lo for p in seeds[name]] for name in names}
    distance = None
    chosen = None
    for step in EROSION_STEPS_MM:
        if step == 0:
            core = crop
        else:
            if distance is None:
                distance = ndi.distance_transform_edt(crop, sampling=spacing_kji)
            core = distance > step
        pieces, _count = ndi.label(core)
        ids = {}
        for name in names:
            found = set()
            for p in local[name]:
                v = _snap(core, p, spacing_kji)
                if v is not None and pieces[v]:
                    found.add(int(pieces[v]))
            ids[name] = found
        if any(not found for found in ids.values()):
            break                                           # eroded too far: a seed lost its bone
        overlap = any(ids[a] & ids[b] for i, a in enumerate(names) for b in names[i + 1:])
        chosen = (step, pieces, ids, not overlap)
        if not overlap:
            break
    if chosen is None:
        return np.zeros(mask.shape, np.uint8), {"separated": False, "erosion_mm": None, "bones": names}
    step, pieces, ids, separated = chosen
    labels = np.zeros(crop.shape, np.uint8)
    for index, name in enumerate(names, start=1):
        labels[np.isin(pieces, list(ids[name])) & (labels == 0)] = index
    if step > 0:
        # competitive growth back into the mask: one voxel layer per round for every bone at once
        rounds = int(np.ceil(step / spacing_kji.min())) + 2
        for _ in range(rounds):
            free = crop & (labels == 0)
            if not free.any():
                break
            grown = labels.copy()
            for index in range(1, len(names) + 1):
                front = ndi.binary_dilation(labels == index) & free & (grown == 0)
                grown[front] = index
            if np.array_equal(grown, labels):
                break
            labels = grown
        # what is still free (thin parts that vanished in the erosion, e.g. the fibula): each
        # connected rest goes to the bone it touches most
        rest, count = ndi.label(crop & (labels == 0))
        for piece, box in enumerate(ndi.find_objects(rest) if count else [], start=1):
            if box is None:
                continue
            box = tuple(slice(max(b.start - 1, 0), b.stop + 1) for b in box)
            where = rest[box] == piece
            ring = ndi.binary_dilation(where) & ~where
            touching = labels[box][ring]
            touching = touching[touching > 0]
            if len(touching):
                labels[box][where] = np.bincount(touching).argmax()
    full = np.zeros(mask.shape, np.uint8)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = labels
    return full, {"separated": separated, "erosion_mm": step, "bones": names}


def _keep_seeded(labels, markers):
    """Per label only the connected pieces that hold one of its markers."""
    for index in np.unique(markers[markers > 0]):
        pieces, _count = ndi.label(labels == index)
        keep = np.unique(pieces[(markers == index) & (pieces > 0)])
        labels[(labels == index) & ~np.isin(pieces, keep)] = 0
    return labels


def split_watershed(mask, intensity, spacing_kji, seeds, seed_radius_mm=1.5):
    """mask: bool (k, j, i); intensity: CT values on the same grid; seeds: {bone: [(k, j, i), ...]}.
    Returns (labels uint8 1..n in the order of the bones WITH seeds, info)."""
    import SimpleITK as sitk
    if ndi is None:
        raise RuntimeError("scipy is not available in this Slicer")
    spacing_kji = np.asarray(spacing_kji, float)
    names = [name for name, pts in seeds.items() if pts]
    nz = np.argwhere(mask)
    if len(nz) == 0 or not names:
        return np.zeros(mask.shape, np.uint8), {"separated": False, "bones": names, "missing": list(seeds)}
    lo = np.maximum(nz.min(0) - 2, 0)
    hi = np.minimum(nz.max(0) + 3, mask.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    crop = mask[box]
    values = np.asarray(intensity[box], np.float32)
    top = float(values[crop].max())
    relief = np.where(crop, top - values, top - float(values.min()) + 1.0).astype(np.float32)
    markers = np.zeros(crop.shape, np.uint32)
    radius = np.ceil(seed_radius_mm / spacing_kji).astype(int)
    found = {}
    for index, name in enumerate(names, start=1):
        found[name] = 0
        for p in seeds[name]:
            v = _snap(crop, np.asarray(p, float) - lo, spacing_kji)
            if v is None:
                continue
            ball = tuple(slice(max(c - r, 0), c + r + 1) for c, r in zip(v, radius))
            markers[ball][crop[ball] & (markers[ball] == 0)] = index
            found[name] += 1
    present = [name for name in names if found[name]]
    if not present:
        return np.zeros(mask.shape, np.uint8), {"separated": False, "bones": names, "missing": list(seeds)}
    image = sitk.GetImageFromArray(relief)
    image.SetSpacing(tuple(float(x) for x in spacing_kji[::-1]))
    marker_image = sitk.GetImageFromArray(markers)
    marker_image.CopyInformation(image)
    flooded = sitk.GetArrayFromImage(sitk.MorphologicalWatershedFromMarkers(
        image, marker_image, markWatershedLine=False, fullyConnected=False))
    labels = _keep_seeded(np.where(crop, flooded, 0).astype(np.uint8), markers)
    full = np.zeros(mask.shape, np.uint8)
    full[box] = labels
    missing = [name for name in seeds if not found.get(name)]
    return full, {"separated": len(present) > 1, "bones": names, "missing": missing}
