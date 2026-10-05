# Measurement methods shared with Doggybone 3D (Matthias Frank, Kleintierklinik Frank & Partner, Freiburg)
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License.

import numpy as np
import os.path as osp
import slicer
import vtk

# Current bone-density threshold (HU) used by snap_to_nearest_bone_voxel below. Kept in sync
# with the module's own "Lower threshold" slider (see set_bone_threshold, called from
# BoneAngleMeterModule.py) so the snap always matches whatever the user has actually configured
# for segmentation, rather than a fixed guess that could drift out of sync.
_bone_threshold_hu = 300.0

def set_bone_threshold(value):
    global _bone_threshold_hu
    _bone_threshold_hu = float(value)

def get_bone_threshold():
    return _bone_threshold_hu

# The specific CT volume node the bone segmentation was actually generated from. Set by
# BoneAngleMeterModule.py right after "Apply" succeeds. Snapping prefers this over re-scanning
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
            self._markups_node.SetAttribute(f"BoneAngleMeter.radius.{self.name}", str(float(radius)))
        if not getattr(self, "_handle_updating", False):
            self._place_radius_handle()          # radius typed in the spinbox: the handle follows
        for cb in self.change_callbacks:
            cb()


    # --- radius handle: an orange point ON the sphere - drag it in any slice (MPR) or 3D view ---------
    RADIUS_HANDLE = "BoneAngleMeter sphere radius"
    SPHERE_PREVIEW = "BoneAngleMeter sphere preview"

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

