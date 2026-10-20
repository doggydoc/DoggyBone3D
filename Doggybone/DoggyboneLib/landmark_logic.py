# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

import numpy as np
import slicer
import vtk

# Current bone-density threshold (HU) used by snap_to_nearest_bone_voxel below. Kept in sync
# with the module's own "Lower threshold" slider (see set_bone_threshold, called from
# Doggybone.py) so the snap always matches whatever the user has actually configured
# for segmentation, rather than a fixed guess that could drift out of sync.
_bone_threshold_hu = 300.0

def set_bone_threshold(value):
    global _bone_threshold_hu
    _bone_threshold_hu = float(value)

def get_bone_threshold():
    return _bone_threshold_hu

# The specific CT volume node the bone segmentation was actually generated from. Set by
# Doggybone.py right after "Apply" succeeds. Snapping prefers this over re-scanning
# the scene for scalar volumes, since a scene can easily end up with more than one volume loaded
# (from DICOM import, cropping, segmentation helper volumes, etc.) - re-scanning and requiring
# there be exactly one would silently do nothing the moment that stops being true.
_bone_volume_node = None

def set_bone_volume(volume_node):
    global _bone_volume_node
    _bone_volume_node = volume_node

def get_bone_volume():
    return _bone_volume_node


def snap_to_nearest_bone_voxel(position_ras, in_plane_radius_mm=100.0, out_of_plane_radius_mm=4.0, lower_threshold=None, margin_hu=0.0):
    '''
    Given a RAS position (e.g. just placed by clicking on the rendered bone surface), searches a
    neighborhood in the underlying CT volume for the nearest voxel at or above
    lower_threshold + margin_hu and returns its RAS center.

    lower_threshold defaults to get_bone_threshold(), which is kept in sync with the module's own
    "Landmark snap threshold" control - a value independent of the segmentation's own Lower
    threshold, so a strict bone-only snap doesn't require sacrificing segmentation completeness
    (raising the segmentation threshold to keep partial-volume/soft-tissue voxels out can also
    exclude legitimate lower-density bone, leaving gaps in the mesh). margin_hu is an optional
    extra cushion on top of that, off by default since the snap threshold is already meant to be
    set to whatever value works for a given scan.

    The search is anisotropic rather than a plain cube/sphere: wide within the axial slice plane
    (in_plane_radius_mm, along the volume's I/J axes) so it can actually reach from a click on the
    skin surface across to the bone, but narrow along the slice-stacking axis
    (out_of_plane_radius_mm, the volume's K axis - the direction axial slices are stacked in) so
    it can't jump to bone at a meaningfully different height/level, which a plain isotropic search
    with a large-enough radius to reach skin-to-bone would risk doing. Note that a large
    in_plane_radius_mm can still reach an unintended structure (a different bone, or the far
    cortex of the same bone) if that happens to be closer than the intended target - the search
    always picks whichever qualifying voxel is nearest, with no notion of "which bone" beyond
    that.

    Uses get_bone_volume() if set (the volume "Apply" actually segmented); otherwise falls back
    to scanning the scene, but only proceeds if there's unambiguously exactly one scalar volume.
    Falls back to the original position unchanged if no volume can be identified, or no
    qualifying voxel is found within the search neighborhood.
    '''
    if lower_threshold is None:
        lower_threshold = get_bone_threshold()
    lower_threshold = lower_threshold + margin_hu

    volume_node = get_bone_volume()
    if volume_node is None:
        volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if len(volume_nodes) != 1:
            return position_ras
        volume_node = volume_nodes[0]

    # a transformed CT: world -> the volume's own (local) RAS first
    parent = volume_node.GetParentTransformNode()
    local_ras = [float(position_ras[0]), float(position_ras[1]), float(position_ras[2])]
    if parent is not None:
        from_world = vtk.vtkGeneralTransform()
        parent.GetTransformFromWorld(from_world)
        local_ras = list(from_world.TransformPoint(local_ras))
    ras_to_ijk = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(ras_to_ijk)
    ijk_h = [0.0, 0.0, 0.0, 0.0]
    ras_to_ijk.MultiplyPoint(local_ras + [1.0], ijk_h)
    i0, j0, k0 = int(round(ijk_h[0])), int(round(ijk_h[1])), int(round(ijk_h[2]))

    spacing = volume_node.GetSpacing()  # (x, y, z) mm per voxel
    ri = max(1, int(round(in_plane_radius_mm / max(spacing[0], 1e-3))))
    rj = max(1, int(round(in_plane_radius_mm / max(spacing[1], 1e-3))))
    rk = max(1, int(round(out_of_plane_radius_mm / max(spacing[2], 1e-3))))

    array = slicer.util.arrayFromVolume(volume_node)  # indexed [k, j, i]
    nk, nj, ni = array.shape

    # Vectorized with numpy instead of a nested Python loop, so a large in-plane radius (which
    # can mean many thousands of voxels to check) stays fast.
    k_lo, k_hi = max(0, k0 - rk), min(nk, k0 + rk + 1)
    j_lo, j_hi = max(0, j0 - rj), min(nj, j0 + rj + 1)
    i_lo, i_hi = max(0, i0 - ri), min(ni, i0 + ri + 1)

    sub = array[k_lo:k_hi, j_lo:j_hi, i_lo:i_hi]
    mask = sub >= lower_threshold
    if not np.any(mask):
        return position_ras

    kk, jj, ii = np.nonzero(mask)
    kk = kk + k_lo
    jj = jj + j_lo
    ii = ii + i_lo
    dk = kk - k0
    dj = jj - j0
    di = ii - i0
    # distance in mm, not in voxel steps (slices are usually thicker than the in-plane pixels)
    dist2 = (di * spacing[0]) ** 2 + (dj * spacing[1]) ** 2 + (dk * spacing[2]) ** 2
    best = int(np.argmin(dist2))
    best_ijk = (int(ii[best]), int(jj[best]), int(kk[best]))

    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    ras_h = [0.0, 0.0, 0.0, 0.0]
    ijk_to_ras.MultiplyPoint([float(best_ijk[0]), float(best_ijk[1]), float(best_ijk[2]), 1.0], ras_h)
    result = [ras_h[0], ras_h[1], ras_h[2]]
    if parent is not None:
        to_world = vtk.vtkGeneralTransform()
        parent.GetTransformToWorld(to_world)
        result = list(to_world.TransformPoint(result))
    return (result[0], result[1], result[2])


class SimpleLandmark:
    def __init__(self, name, description="", image_path=""):
        self.name = name
        self.placed = False
        self.change_callbacks = []
        self.description = description
        self.image_path = image_path

        # Internal members
        self._markups_node = None
        self._id = None
        self._private_observers = []

    def set_markups_node_id(self, id):
        self._markups_node = slicer.mrmlScene.GetNodeByID(id)

        # Set automatic glyph style
        self._markups_node.GetDisplayNode().SetGlyphType(self._markups_node.GetDisplayNode().ThickCross2D)
        self._markups_node.GetDisplayNode().SetGlyphScale(2.5)

    def add_change_callback(self, callback):
        self.change_callbacks.append(callback)

    def show(self):
        if self._id is not None:
            self._markups_node.SetNthControlPointVisibility(self._id, True)

    def hide(self):
        self.stop_interaction()
        if self._id is not None:
            self._markups_node.SetNthControlPointVisibility(self._id, False)

    def start_interaction(self):
        if self.placed:
            self._markups_node.SetNthControlPointLocked(self._id, False)
            self._markups_node.SetNthControlPointSelected(self._id, True)
            self.center_in_slices()
            self._private_observers.append(self._markups_node.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, 
                                                                         lambda caller, event: self._changed_callback(caller)))
        else:
            # place into THIS landmark's list (left / right): the active list may still be the other
            # side's after a reload or 'Delete all' - the point would land there with a default name
            # ('DoggyboneFiducialsRight-1') and belong to no landmark
            selection = slicer.app.applicationLogic().GetSelectionNode()
            if selection is not None and self._markups_node is not None:
                selection.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
                selection.SetActivePlaceNodeID(self._markups_node.GetID())
            slicer.modules.markups.logic().StartPlaceMode(0)
            self._private_observers.append(self._markups_node.AddObserver(slicer.vtkMRMLMarkupsNode.PointAddedEvent, 
                                                                         lambda caller, event: self._added_callback()))
            self._private_observers.append(self._markups_node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, 
                                                                         lambda caller, event: self._defined_callback()))

    def stop_interaction(self):
        if self.placed:
            self._markups_node.SetNthControlPointLocked(self._id, True)
            self._markups_node.SetNthControlPointSelected(self._id, False)
        
        # Turn off placement mode and remove observers
        for observer in self._private_observers:
            self._markups_node.RemoveObserver(observer)

        interaction_node = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interaction_node.SwitchToViewTransformMode()
        interaction_node.SetPlaceModePersistence(0)                                                       

    def center_in_slices(self, excluded_slice_nodes=None):
        position_RAS = [0.0, 0.0, 0.0]
        self._markups_node.GetNthControlPointPosition(self._id, position_RAS)

        if excluded_slice_nodes is None:
            slicer.vtkMRMLSliceNode.JumpAllSlices(slicer.mrmlScene, *position_RAS, slicer.vtkMRMLSliceNode.CenteredJumpSlice)
        else:
            slice_nodes = slicer.util.getNodesByClass("vtkMRMLSliceNode")
            for slice_node in slice_nodes:
                if slice_node.GetName() not in excluded_slice_nodes:
                    slice_node.SetJumpModeToCentered()
                    slice_node.JumpSlice(*position_RAS)

    def define(self, x, y, z):
        self._markups_node.AddControlPoint([x, y, z], self.name)
        self._id = self._markups_node.GetNumberOfControlPoints() - 1
        self.placed = True
        self._changed_callback()

    def delete(self):
        '''
        Un-places this landmark so it can be placed again. Note: all landmarks on a side share
        one markups node, and this landmark's _id is a raw index into it, so the underlying
        fiducial point is hidden and locked rather than physically removed from the node -
        removing it would silently shift the _id of every other already-placed landmark with a
        higher index, corrupting their positions.
        '''
        if self.placed and self._id is not None:
            self.stop_interaction()
            self._markups_node.SetNthControlPointVisibility(self._id, False)
            self._markups_node.SetNthControlPointLocked(self._id, True)
            # Mark it as deleted so a module reload does not re-link it (see
            # DoggyboneWidget._relink_existing_landmarks).
            self._markups_node.SetNthControlPointLabel(self._id, f"{self.name} (deleted)")
            self._id = None
            self.placed = False
            for cb in self.change_callbacks:
                cb()

    def get_position(self):
        if self._id is None or not self.placed:
            return None
        xyz_buffer = [0.0, 0.0, 0.0]
        self._markups_node.GetNthControlPointPosition(self._id, xyz_buffer)
        return np.array(xyz_buffer)

    def _added_callback(self):
        '''
        Called when a point is added (but not necessarily placed). Takes care of giving the right label to the newly added point
        '''
        id = self._markups_node.GetNumberOfControlPoints() - 1 # id is only temporary until the point is defined
        self._markups_node.SetNthControlPointLabel(id, self.name)
        
    def _defined_callback(self):
        '''
        Called when a point is placed/defined. 
        '''
        self._id = self._markups_node.GetNumberOfControlPoints()-1
        self.placed = True

        # Snap the newly-placed point onto actual bone-density CT voxels rather than leaving it
        # wherever it landed on the rendered (possibly smoothed) segmentation surface, which can
        # sit on partial-volume/soft-tissue-influenced voxels near the true bone edge. Only
        # applied here, on the very first placement - not on every subsequent reposition/drag
        # (see _changed_callback), since cortical (outer shell) bone is dense but the interior
        # (cancellous/trabecular bone, exactly where sphere-center landmarks are meant to go) is
        # naturally much softer. Re-snapping on every drag would fight the user whenever they try
        # to move a point deeper into that legitimate but lower-density interior.
        raw_position = self.get_position()
        if raw_position is not None:
            snapped = snap_to_nearest_bone_voxel(raw_position)
            if tuple(snapped) != tuple(raw_position):
                self._markups_node.SetNthControlPointPosition(self._id, snapped[0], snapped[1], snapped[2])

        self.stop_interaction()
        self._changed_callback()
        self.start_interaction()

    def _changed_callback(self, caller=None):
        '''
        Called when a point is changed (including defined). Triggers updates to all dependent measurements
        '''
        if caller is not None:
            calling_node = caller.GetAttribute("Markups.MovingInSliceView")
        else:
            calling_node = None
        self.center_in_slices([calling_node])

        for cb in self.change_callbacks:
            cb()


class SphereLandmark(SimpleLandmark):
    '''
    A landmark placed at a sphere's center, with a radius the user adjusts by hand (e.g. via a
    spinbox in the GUI) rather than one computed from several points. Used where visually
    matching a sphere against the bone surface is more practical than an automatic best-fit,
    for example locating a femoral condyle.
    '''
    def __init__(self, name, description="", image_path="", default_radius=8.0):
        super().__init__(name, description, image_path)
        self.radius = default_radius

    def set_radius(self, radius):
        self.radius = radius
        # Stored on the markups node so the radius survives a module reload (re-linking).
        if self._markups_node is not None:
            self._markups_node.SetAttribute(f"Doggybone.radius.{self.name}", str(float(radius)))
        if not getattr(self, "_handle_updating", False):
            self._place_radius_handle()          # radius typed in the spinbox: the handle follows
        for cb in self.change_callbacks:
            cb()


    # --- radius handle: an orange point ON the sphere - drag it in any slice (MPR) or 3D view ---------
    RADIUS_HANDLE = "Doggybone sphere radius"
    SPHERE_PREVIEW = "Doggybone sphere preview"

    def _update_sphere_preview(self):
        """The sphere of THIS landmark while it is edited - shown right away (before the measurement is
        complete) as a circle in all three slice views and a transparent sphere in 3D."""
        if not self.placed or self._id is None or getattr(self, "_handle_tag", None) is None:
            return
        node = slicer.mrmlScene.GetFirstNodeByName(self.SPHERE_PREVIEW)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.SPHERE_PREVIEW)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(1.0, 0.55, 0.1)
            display.SetOpacity(0.25)
            display.SetVisibility2D(True)
            display.SetSliceIntersectionThickness(2)
            node.SetSelectable(False)
        source = vtk.vtkSphereSource()
        c = self.get_position()
        source.SetCenter(float(c[0]), float(c[1]), float(c[2]))
        source.SetRadius(max(float(self.radius), 0.1))
        source.SetThetaResolution(48)
        source.SetPhiResolution(48)
        source.Update()
        node.SetAndObservePolyData(source.GetOutput())
        node.GetDisplayNode().SetVisibility(True)

    def _remove_sphere_preview(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.SPHERE_PREVIEW)
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    def _handle_node(self, create=False):
        node = slicer.mrmlScene.GetFirstNodeByName(self.RADIUS_HANDLE)
        if node is None and create:
            # a new markups node may become the ACTIVE place list - keep the landmarks' list active
            selection = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
            active = selection.GetActivePlaceNodeID() if selection is not None else None
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.RADIUS_HANDLE)
            if selection is not None:
                selection.SetActivePlaceNodeID(active or (self._markups_node.GetID() if self._markups_node else None))
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetSelectedColor(1.0, 0.55, 0.1)
            display.SetColor(1.0, 0.55, 0.1)
            display.SetGlyphScale(2.5)
            display.SetTextScale(2.0)
            display.SetSnapMode(slicer.vtkMRMLMarkupsDisplayNode.SnapModeUnconstrained)
            node.SetMarkupLabelFormat("")
        return node

    def _remove_handle_observer(self):
        node = self._handle_node()
        tag = getattr(self, "_handle_tag", None)
        if node is not None and tag is not None:
            node.RemoveObserver(tag)
        self._handle_tag = None

    def _handle_direction(self):
        direction = getattr(self, "_handle_dir", None)
        if direction is None:
            try:            # first time: to the right in the axial (red) slice
                red = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode()
                m = red.GetSliceToRAS()
                direction = np.array([m.GetElement(0, 0), m.GetElement(1, 0), m.GetElement(2, 0)], float)
            except Exception:
                direction = np.array([1.0, 0.0, 0.0])
        n = np.linalg.norm(direction)
        return direction / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _place_radius_handle(self):
        """Put the handle on the sphere surface (only while this sphere is being edited)."""
        node = self._handle_node()
        if node is None or getattr(self, "_handle_tag", None) is None or not self.placed or self._id is None:
            return
        self._update_sphere_preview()
        p = self.get_position() + float(self.radius) * self._handle_direction()
        self._handle_updating = True
        try:
            if node.GetNumberOfControlPoints() == 0:
                node.AddControlPointWorld(vtk.vtkVector3d(*[float(v) for v in p]))
                node.SetNthControlPointLabel(0, "r")
            else:
                node.SetNthControlPointPositionWorld(0, *[float(v) for v in p])
        finally:
            self._handle_updating = False

    def _show_radius_handle(self):
        if not self.placed or self._id is None:
            return
        node = self._handle_node(create=True)
        self._remove_handle_observer()
        node.RemoveAllControlPoints()
        node.SetLocked(False)
        node.GetDisplayNode().SetVisibility(True)
        self._handle_tag = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                                            lambda caller, event: self._on_handle_moved())
        self._place_radius_handle()
        # all three slice views through the centre: the circle is seen in each (largest cross-section)
        try:
            self.center_in_slices()
        except Exception:
            pass

    def _hide_radius_handle(self):
        self._remove_sphere_preview()
        self._remove_handle_observer()
        node = self._handle_node()
        if node is not None:
            node.RemoveAllControlPoints()

    def _on_handle_moved(self):
        if getattr(self, "_handle_updating", False) or not self.placed:
            return
        node = self._handle_node()
        if node is None or node.GetNumberOfControlPoints() == 0:
            return
        p = np.array(node.GetNthControlPointPositionWorld(0), float)
        c = self.get_position()
        r = float(np.linalg.norm(p - c))
        if r < 0.5:
            return
        self._handle_dir = (p - c) / r
        self._handle_updating = True
        try:
            self.set_radius(r)
        finally:
            self._handle_updating = False
        self._update_sphere_preview()

    def start_interaction(self):
        super().start_interaction()
        if self.placed:
            self._show_radius_handle()

    def stop_interaction(self):
        self._hide_radius_handle()
        super().stop_interaction()

    def _defined_callback(self):
        super()._defined_callback()
        self._show_radius_handle()

    def _changed_callback(self, caller=None):
        super()._changed_callback(caller)
        self._place_radius_handle()            # the centre moved: the handle moves with it

    def delete(self):
        self._hide_radius_handle()
        super().delete()



# --- diaphysis centres: area centroid of the bone cross-section ---------------------------------------

# Function returning the bone surface (vtkPolyData, world coordinates) the centroids are taken from -
# set by the module widget (its bone segmentation). None or a function returning None: no centring.
_bone_surface_provider = None


def set_bone_surface_provider(provider):
    global _bone_surface_provider
    _bone_surface_provider = provider


_follow_suspended = 0


class follow_suspended:
    """with follow_suspended(): ... - percentage points do NOT follow the head / condyles meanwhile
    (loading a scene, importing or transferring landmarks keeps the stored positions)."""
    def __enter__(self):
        global _follow_suspended
        _follow_suspended += 1

    def __exit__(self, *exc):
        global _follow_suspended
        _follow_suspended -= 1
        return False


def get_bone_surface():
    try:
        return _bone_surface_provider() if _bone_surface_provider is not None else None
    except Exception:
        return None


class CentroidLandmark(SimpleLandmark):
    '''
    Midpoint of a diaphysis, set to the AREA CENTROID of the bone cross-section: the click only gives
    the height along the bone; the section is taken perpendicular to the local bone axis (refined from
    the centroids of two neighbouring sections), the filled outer contour - medullary canal included -
    of the bone that was picked gives the centre (see DoggyboneLib.section_centre).
    axis_from: ((proximal landmark names), (distal landmark names)) - a first guess of the bone axis
               from landmarks that are already placed (optional, the local shape is used otherwise).
    fraction:  for points defined by a percentage of the bone length (from the proximal end), e.g.
               0.33 - placed automatically from the axis landmarks ('Place at ...%').
    '''
    def __init__(self, name, description="", image_path="", axis_from=None, fraction=None):
        super().__init__(name, description, image_path)
        self.axis_from = axis_from
        self.fraction = fraction
        self.auto_centre = True
        self.note = ""

    # positions of other landmarks of the same side (they share this markups node)
    def _sibling(self, name):
        node = self._markups_node
        if node is None:
            return None
        for i in range(node.GetNumberOfControlPoints()):
            if node.GetNthControlPointLabel(i) == name and node.GetNthControlPointVisibility(i) \
                    and node.GetNthControlPointPositionStatus(i) == slicer.vtkMRMLMarkupsNode.PositionDefined:
                p = [0.0, 0.0, 0.0]
                node.GetNthControlPointPosition(i, p)
                return np.array(p, float)
        return None

    def _sibling_radius(self, name, default=None):
        node = self._markups_node
        value = node.GetAttribute(f"Doggybone.radius.{name}") if node is not None else None
        try:
            return float(value) if value else default
        except ValueError:
            return default

    def _ends(self):
        """(proximal point, distal point, [proximal names], [distal names]) from axis_from, or None."""
        if not self.axis_from:
            return None
        ends = []
        for names in self.axis_from:
            points = [self._sibling(n) for n in names]
            if any(p is None for p in points):
                return None
            ends.append(np.mean(points, axis=0))
        if float(np.linalg.norm(ends[1] - ends[0])) < 5.0:
            return None
        return ends[0], ends[1], list(self.axis_from[0]), list(self.axis_from[1])

    def axis_hint(self):
        ends = self._ends()
        if ends is None:
            return None
        d = ends[1] - ends[0]
        return d / np.linalg.norm(d)

    def centred(self, position):
        """Area centroid of the cross-section at the height of `position`, or None."""
        from DoggyboneLib import section_centre
        surface = get_bone_surface()
        if surface is None or surface.GetNumberOfPoints() == 0:
            self.note = "No bone segmentation - the point stays where it was set (not centred)."
            return None
        centre, axis, area = section_centre.diaphysis_centre(surface, position, axis_hint=self.axis_hint())
        if centre is None:
            self.note = "No bone cross-section found at this point - not centred."
            return None
        moved = float(np.linalg.norm(np.asarray(centre) - np.asarray(position, float)))
        self.note = (f"Centred in the cross-section ({area:.0f} mm\N{SUPERSCRIPT TWO}, perpendicular to the "
                     f"local bone axis): moved {moved:.1f} mm.")
        return np.asarray(centre, float)

    def _move_to(self, p):
        self._markups_node.SetNthControlPointPosition(self._id, float(p[0]), float(p[1]), float(p[2]))

    def _defined_callback(self):
        # like SimpleLandmark, but the new point goes to the section centroid instead of the nearest
        # dense bone voxel (the centre of a diaphysis lies in the medullary canal - not dense bone)
        self._id = self._markups_node.GetNumberOfControlPoints() - 1
        self.placed = True
        raw = self.get_position()
        if self.auto_centre and raw is not None and self.fraction is not None:
            found = self.length_point()
            if found is not None:
                raw = found[0]                    # percentage point: the click only says which bone
        centre = self.centred(raw) if (self.auto_centre and raw is not None) else None
        if self.auto_centre and self.fraction is not None and self._ends() is None:
            self.note += self._pending_note()
        if centre is not None:
            self._move_to(centre)
        elif raw is not None and not self.auto_centre:
            self.note = ""
            snapped = snap_to_nearest_bone_voxel(raw)
            if tuple(snapped) != tuple(raw):
                self._move_to(snapped)
        self.stop_interaction()
        self._changed_callback()
        self.start_interaction()

    def recentre(self):
        """Centre the placed point again at its current height (e.g. after dragging it)."""
        if not self.placed or self._id is None:
            return False
        centre = self.centred(self.get_position())
        if centre is None:
            self._changed_callback()
            return False
        self._move_to(centre)
        self._changed_callback()
        return True

    def length_point(self):
        """Point at `fraction` of the bone length from the proximal end, on the line between the
        axis landmarks. The ends of the bone are the sphere SURFACES: proximal end = head centre minus
        its radius, distal end = condyle midpoint plus their mean radius (sphere landmarks)."""
        ends = self._ends()
        if ends is None or self.fraction is None:
            return None
        prox, dist, prox_names, dist_names = ends
        a = (dist - prox) / np.linalg.norm(dist - prox)
        r_prox = np.mean([self._sibling_radius(n, 0.0) for n in prox_names])
        r_dist = np.mean([self._sibling_radius(n, 0.0) for n in dist_names])
        top = prox - r_prox * a
        bottom = dist + r_dist * a
        return top + float(self.fraction) * (bottom - top), float(np.linalg.norm(bottom - top))

    def _pending_note(self):
        names = " and ".join(", ".join(group) for group in (self.axis_from or ()))
        return (f" Moves to {100.0 * float(self.fraction):.0f} % of the bone length by itself as soon as "
                f"{names} are placed.")

    def follow_ends(self):
        """Called when one of the axis landmarks (head / condyles, their spheres) changes: a centred
        percentage point moves to its height again - quietly, without taking over the interaction or
        jumping the slice views."""
        if not (self.placed and self._id is not None and self.auto_centre and self.fraction is not None):
            return
        if _follow_suspended:
            return
        found = self.length_point()
        if found is None:
            return
        point, length = found
        centre = self.centred(point)
        target = centre if centre is not None else point
        if float(np.linalg.norm(np.asarray(target) - self.get_position())) < 0.05:
            return
        self._move_to(target)
        self.note = (f"At {100.0 * float(self.fraction):.0f} % of the bone length ({length:.1f} mm), "
                     f"follows the femur head / condyles. " + self.note)
        for cb in self.change_callbacks:
            cb()

    def place_at_fraction(self):
        """Places (or moves) the point at its percentage of the bone length, centred in the section.
        Returns a message for the user."""
        found = self.length_point()
        if found is None:
            names = " / ".join(", ".join(group) for group in (self.axis_from or ()))
            return f"Place {names} first - they define the bone length."
        point, length = found
        centre = self.centred(point)
        target = centre if centre is not None else point
        self.stop_interaction()
        if self.placed and self._id is not None:
            self._move_to(target)
            self._changed_callback()
        else:
            self.define(*[float(v) for v in target])
        self.start_interaction()
        return (f"Placed at {100.0 * float(self.fraction):.0f} % of the bone length ({length:.1f} mm, from the "
                f"proximal end of the head to the distal end of the condyles). " + self.note)
