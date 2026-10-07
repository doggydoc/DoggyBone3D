# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""SegmentationMixin: Bone segmentation clean-up: lasso, remove bone, fill cavities, small islands, undo.

Part of DoggyboneWidget (see Doggybone.py). Methods were moved here unchanged; they rely on
attributes created in DoggyboneWidget.setup() and on methods of the other mixins.
"""

from DoggyboneUI.common import *


class SegmentationMixin:

    # ------------------------------------------------------------------
    # Segmentation clean-up: lasso (Scissors, erase outside) + small islands
    # ------------------------------------------------------------------
    def _ensure_cleanup_editor(self):
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
            self.segmentation_status.setText("No segmentation - run 'Apply' first.")
            return None
        volume_node = landmark_logic.get_bone_volume()
        if volume_node is None:
            volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if len(volumes) != 1:
                self.segmentation_status.setText("Could not identify the CT volume - run 'Apply' first.")
                return None
            volume_node = volumes[0]
            landmark_logic.set_bone_volume(volume_node)
        editor = self._cleanup_editor_widget
        if editor is None:
            editor = slicer.qMRMLSegmentEditorWidget()
            editor.setMRMLScene(slicer.mrmlScene)
            editor.setMaximumNumberOfUndoStates(0)   # one undo only: Doggybone's Delete (baseline)
            editor_node = slicer.mrmlScene.GetFirstNodeByName("Doggybone segment editor")
            if editor_node is None:
                editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "Doggybone segment editor")
            editor.setMRMLSegmentEditorNode(editor_node)
            self._cleanup_editor_widget = editor
        if editor.segmentationNode() != segmentation_node:
            editor.setSegmentationNode(segmentation_node)
        if hasattr(editor, "setSourceVolumeNode"):
            editor.setSourceVolumeNode(volume_node)
        else:
            editor.setMasterVolumeNode(volume_node)
        editor.setCurrentSegmentID(segmentation_node.GetSegmentation().GetNthSegmentID(0))
        return editor

    def onEraseInsideToggled(self, checked):
        self._bone_lasso_toggled(checked, "inside")

    def onLassoToggled(self, checked):
        self._bone_lasso_toggled(checked, "outside")

    def _bone_lasso_toggled(self, checked, mode):
        '''Own lasso on the bone segmentation (erase inside / outside, full depth). The embedded Segment
        Editor's Scissors did not receive mouse events in this module, so the lasso is handled
        like the guide tools: own view observers, orange trace, one stroke per click.'''
        if not checked:
            other = self.lasso_button if mode == "inside" else getattr(self, "erase_inside_button", None)
            if other is None or not other.checked:
                self._stop_lasso()   # landmark placement stays paused until a landmark is clicked
            return
        # only one of the two lasso buttons is on
        other = self.lasso_button if mode == "inside" else getattr(self, "erase_inside_button", None)
        if other is not None and other.checked:
            was = other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(was)
        self._bone_lasso_mode = mode
        own = self.lasso_button if mode == "outside" else self.erase_inside_button
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
            self.segmentation_status.setText("No segmentation - run 'Apply' first.")
            was_blocked = own.blockSignals(True)
            own.setChecked(False)
            own.blockSignals(was_blocked)
            return
        self._stop_guide_edit()
        self._pause_landmark_placement()  # no landmark placement while drawing
        self._hide_preview_for_guide_edit(bone_lasso=True)  # cut on the visible ORIGINAL bone (any guide mode)
        self._lasso_hidden_preview = list(getattr(self, "_lasso_hidden_preview", []) or []) + \
            list(getattr(self, "_guide_hidden_preview", []))
        self._guide_hidden_preview = []
        self._guide_active_tool = "bone_lasso"
        self._install_guide_observers()
        what = ("the part to REMOVE (orange trace). On release everything INSIDE is removed"
                if mode == "inside" else
                "the region to KEEP (orange trace). On release everything OUTSIDE is removed")
        self.segmentation_status.setText(
            f"Lasso active: in the 3D view hold the left mouse button and draw around {what} through the "
            "full depth and the lasso switches off - rotate, then click the button again for another stroke.")

    def _pause_landmark_placement(self):
        '''Stop placing / dragging the current landmark (landmarks stay visible) and give the mouse
        back to the view. Placement resumes when a landmark is clicked in the list, or via
        Next / Previous.'''
        panel = self._current_panel()
        measurement_widget = panel.measurement_stack.currentWidget()
        if measurement_widget is not None and getattr(measurement_widget, "enabled", False):
            current = measurement_widget.landmark_stack.currentWidget()
            if current is not None:
                current.landmark.stop_interaction()
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
        interaction_node.SetPlaceModePersistence(0)

    def _stop_lasso(self):
        buttons = [b for b in (getattr(self, "lasso_button", None), getattr(self, "erase_inside_button", None))
                   if b is not None]
        was_active = any(b.checked for b in buttons)
        if getattr(self, "_guide_active_tool", None) == "bone_lasso":
            self._remove_guide_observers()
            self._guide_active_tool = None
            # the lasso set the bone fully opaque (and hid things) - back to the user's view
            if hasattr(self, "_restore_guide_hidden"):
                self._restore_guide_hidden()
        for button in buttons:
            if button.checked:
                was_blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(was_blocked)
        for node_id in getattr(self, "_lasso_hidden_preview", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(True)
        self._lasso_hidden_preview = []
        if was_active and getattr(self, "segmentation_status", None) is not None:
            self.segmentation_status.setText("Lasso off.")

    # --- remove a whole bone (patella, fabella ...) by clicking it in 3D ------------------------------
    def onRemoveBoneToggled(self, checked):
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        interactor = view.interactor()
        tag = getattr(self, "_remove_bone_tag", None)
        if tag is not None:
            interactor.RemoveObserver(tag)
            self._remove_bone_tag = None
        if not checked:
            self.segmentation_status.setText("Bone removal off.")
            return
        self._stop_lasso()
        self._remove_bone_tag = interactor.AddObserver("LeftButtonPressEvent", self._on_remove_bone_click, 2.0)
        self.segmentation_status.setText(
            "Click the bone to remove (patella, fabella, tibia ...) in the 3D view - one bone per click, "
            "then the tool switches off. Drag on the background to rotate.")

    def _on_remove_bone_click(self, caller, event):
        if caller.GetShiftKey() or caller.GetControlKey() or caller.GetAltKey():
            return
        x, y = caller.GetEventPosition()
        bone = self._get_bone_polydata()
        if bone is None:
            return
        renderer = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderers().GetFirstRenderer()
        ends = []
        for z in (0.0, 1.0):
            renderer.SetDisplayPoint(float(x), float(y), z)
            renderer.DisplayToWorld()
            wp = renderer.GetWorldPoint()
            ends.append(np.array([wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]]))
        key = (id(bone), bone.GetMTime())
        if getattr(self, "_remove_bone_locator_key", None) != key:
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(bone)
            locator.BuildLocator()
            self._remove_bone_locator, self._remove_bone_locator_key = locator, key
        t, sub_id = vtk.reference(0.0), vtk.reference(0)
        hit, pcoords = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        if not self._remove_bone_locator.IntersectWithLine(ends[0].tolist(), ends[1].tolist(), 0.0, t, hit, pcoords, sub_id):
            return                                    # background: rotate as usual
        command = caller.GetCommand(self._remove_bone_tag)
        if command is not None:
            command.SetAbortFlag(1)
        ray = ends[1] - ends[0]
        ray = ray / np.linalg.norm(ray)
        qt.QTimer.singleShot(0, lambda h=np.asarray(hit, float), r=ray: self._remove_bone_once(h, r))

    def _remove_bone_once(self, hit, ray):
        # one bone per click: the tool switches itself off again
        self.remove_bone_button.setChecked(False)
        self._remove_bone_at(hit, ray)

    def _remove_bone_at(self, hit, ray):
        """Remove the clicked bone (patella, fabella, tibia ...) from the bone segmentation.
        Works locally around the click: the bone is eroded step by step until the clicked piece comes
        loose from its neighbour (the piece no longer reaches the edge of the work box), then every bone
        voxel is given back to the piece or its neighbour by a watershed on the distance map - the cut
        runs through the narrowest point of the contact (joint space / partial-volume bridge)."""
        from vtk.util.numpy_support import vtk_to_numpy
        import scipy.ndimage as ndi
        self._remove_bone_last_click = (np.asarray(hit, float).tolist(), np.asarray(ray, float).tolist())
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None:
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            image = self._bone_labelmap_copy(segmentation_node)
            ext = image.GetExtent()
            dims = (ext[5] - ext[4] + 1, ext[3] - ext[2] + 1, ext[1] - ext[0] + 1)
            arr = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims)
            m = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(m)
            M = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])
            Minv = np.linalg.inv(M)
            spacing = np.array([np.linalg.norm(M[:3, c]) for c in range(3)])      # i, j, k
            mask = arr > 0
            # the voxel under the click: a little way INTO the bone along the view ray
            seed = None
            for depth in np.arange(0.0, 4.01, 0.25):
                p = (Minv @ np.append(hit + depth * ray, 1.0))[:3]
                i, j, k = (int(round(v)) for v in p)
                kk, jj, ii = k - ext[4], j - ext[2], i - ext[0]
                if 0 <= kk < dims[0] and 0 <= jj < dims[1] and 0 <= ii < dims[2] and mask[kk, jj, ii]:
                    seed = (kk, jj, ii)
                    break
            if seed is None:
                self.segmentation_status.setText("No bone voxel under the click - try again on the bone surface.")
                return
            # 1) the whole connected bone the click belongs to - already separate? then remove it as a whole
            labels_all, _ = ndi.label(mask)
            whole = labels_all == labels_all[seed]
            sizes_all = np.bincount(labels_all.ravel())[1:]
            if whole.sum() < 0.5 * float(sizes_all.max()):
                remove = whole
                lo = (0, 0, 0)
                note = "separate piece"
            else:
                # 2) attached to the big bone: work in a box around the click
                radius_mm = 40.0
                rad = [int(np.ceil(radius_mm / spacing[2 - d])) for d in range(3)]      # k, j, i
                lo = [max(0, seed[d] - rad[d]) for d in range(3)]
                hi = [min(dims[d], seed[d] + rad[d] + 1) for d in range(3)]
                sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] & whole[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                local_seed = tuple(seed[d] - lo[d] for d in range(3))
                edt = ndi.distance_transform_edt(sub, sampling=tuple(spacing[::-1]))
                border = np.zeros(sub.shape, dtype=bool)
                for d in range(3):
                    sl = [slice(None)] * 3
                    if lo[d] > 0:
                        sl[d] = 0
                        border[tuple(sl)] = True
                    if hi[d] < dims[d]:
                        sl[d] = -1
                        border[tuple(sl)] = True
                piece_core = None
                used_depth = 0.0
                for depth in np.arange(0.4, 1.61, 0.2):            # erosion depth in mm (deeper would cut off the femoral head)
                    core = edt > depth
                    labels, count = ndi.label(core)
                    if count == 0:
                        break
                    seed_label = int(labels[local_seed])
                    if seed_label == 0:            # the click voxel eroded away: nearest core voxel
                        window = labels[tuple(slice(max(0, c - 8), c + 9) for c in local_seed)]
                        found = window[window > 0]
                        if len(found) == 0:
                            break
                        seed_label = int(np.bincount(found).argmax())
                    candidate = labels == seed_label
                    if not (candidate & border).any():
                        piece_core, used_depth = candidate, float(depth)
                        other_core = core & ~candidate
                        break
                if piece_core is None:
                    self.segmentation_status.setText(
                        "That bone could not be separated from its neighbour here (it is the femur, or the contact "
                        "is too broad). Click the middle of the patella / fabella, or cut the bridge with the lasso.")
                    return
                # 3) watershed: every bone voxel goes to the nearer core, cut along the narrowest contact
                markers = np.zeros(sub.shape, dtype=np.int16)
                markers[other_core] = 2
                markers[piece_core] = 1
                markers[~sub] = 3                  # background is its own basin
                cost = (np.clip(edt.max() - edt, 0, None) / max(edt.max(), 1e-6) * 65000).astype(np.uint16)
                basins = ndi.watershed_ift(cost, markers)
                remove_local = (basins == 1) & sub
                remove = np.zeros(mask.shape, dtype=bool)
                remove[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = remove_local
                note = f"cut free at {used_depth:.1f} mm"
            removed = int(remove.sum())
            if removed == 0:
                self.segmentation_status.setText("Nothing to remove there.")
                return
            self._bone_push_undo(segmentation_node)
            arr[remove] = 0
            image.GetPointData().GetScalars().Modified()
            image.Modified()
            self._bone_set_labelmap(segmentation_node, image)
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._invalidate_guide_surfaces()
        self.segmentation_status.setText(
            f"Bone removed ({removed * float(np.prod(spacing)):.0f} mm\N{SUPERSCRIPT THREE}, {note}). "
            "For the next bone press 'Remove bone (click)' again. 'Delete' undoes all clean-up edits.")

    def onFillCavities(self):
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        import scipy.ndimage as ndi
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
            self.segmentation_status.setText("No segmentation - run 'Apply' first.")
            return
        self._stop_lasso()
        self._bone_push_undo(segmentation_node)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            image = self._bone_labelmap_copy(segmentation_node)
            ext = image.GetExtent()
            dims = (ext[5] - ext[4] + 1, ext[3] - ext[2] + 1, ext[1] - ext[0] + 1)
            scalars = vtk_to_numpy(image.GetPointData().GetScalars())
            arr = scalars.reshape(dims)
            mask = arr > 0
            label = int(arr[mask].max()) if mask.any() else 1
            before = int(mask.sum())
            mask = ndi.binary_closing(mask, iterations=1) | mask          # seal hairline cortical gaps
            # slices across the bone: along the image axis in which the bone is longest
            coords = np.nonzero(mask)
            extents = [int(c.max() - c.min()) if len(c) else 0 for c in coords]
            m = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(m)
            spacing = [np.linalg.norm([m.GetElement(r, c) for r in range(3)]) for c in (2, 1, 0)]   # (k, j, i)
            axis = int(np.argmax([e * s for e, s in zip(extents, spacing)]))
            for index in range(mask.shape[axis]):
                sl = [slice(None)] * 3
                sl[axis] = index
                face = mask[tuple(sl)]
                if face.any():
                    mask[tuple(sl)] = ndi.binary_fill_holes(face)
            mask = ndi.binary_fill_holes(mask)
            arr[...] = np.where(mask, label, 0).astype(arr.dtype)
            image.GetPointData().GetScalars().Modified()
            image.Modified()
            self._bone_set_labelmap(segmentation_node, image)
        finally:
            qt.QApplication.restoreOverrideCursor()
        added = int(mask.sum()) - before
        # the prepared guide surfaces were made from the old bone - rebuild them on the next paint
        self._invalidate_guide_surfaces()
        self.segmentation_status.setText(
            f"Cavities filled: {added} voxels added - the bone is solid, only its outer surface remains. "
            "Paint the guide footprint again ('Delete' undoes the fill).")

    def _invalidate_guide_surfaces(self):
        """The bone changed: the prepared guide bone surfaces (both bones, both guide modes) were made
        from the old bone - removed, they are rebuilt from the new bone on the next paint."""
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            if (node.GetName() or "").startswith("Doggybone guide bone surface"):
                slicer.mrmlScene.RemoveNode(node)

    def onRemoveSmallIslands(self):
        editor = self._ensure_cleanup_editor()
        if editor is None:
            return
        self._stop_lasso()
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        self._bone_push_undo(segmentation_node)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            editor.setActiveEffectByName("Islands")
            effect = editor.activeEffect()
            effect.setParameter("Operation", "REMOVE_SMALL_ISLANDS")
            effect.setParameter("MinimumSize", str(int(self.island_min_size.value)))
            effect.self().onApply()
            editor.setActiveEffect(None)
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._invalidate_guide_surfaces()
        self.segmentation_status.setText(
            f"Islands smaller than {int(self.island_min_size.value)} voxels removed.")

    def onDeleteSegmentationEdits(self):
        '''Discard every clean-up edit: restore the segmentation as it was before the first
        lasso / islands step (since the last 'Apply').'''
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        baseline = getattr(self, "_bone_baseline", None)
        if segmentation_node is None or baseline is None:
            self.segmentation_status.setText("No clean-up edits to delete.")
            return
        self._stop_lasso()
        self._bone_set_labelmap(segmentation_node, baseline)
        self._bone_baseline = None
        self._invalidate_guide_surfaces()
        self.segmentation_status.setText("Clean-up edits deleted - segmentation restored. Start again with the lasso.")

    # --- clean-up baseline + lasso cut for the bone segmentation --------------
    def onTakeOverEditorSegmentation(self):
        """Doggybone bone segmentation = union of the VISIBLE segments of a segmentation made in the
        Segment Editor (the one open there, otherwise the first other segmentation in the scene)."""
        own_name = self.SEGMENTATION_NODE_NAME
        source, volume = None, None
        try:
            editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            node = editor.segmentationNode()
            if node is not None and node.GetName() != own_name:
                source = node
            volume = editor.sourceVolumeNode() if hasattr(editor, "sourceVolumeNode") else editor.masterVolumeNode()
        except Exception:
            pass
        if source is None:
            others = [n for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode") if n.GetName() != own_name]
            source = others[0] if others else None
        if source is None:
            self.segmentation_status.setText("No Segment Editor segmentation found.")
            return
        display = source.GetDisplayNode()
        segmentation = source.GetSegmentation()
        ids = vtk.vtkStringArray()
        names = []
        for i in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(i)
            if display is None or display.GetSegmentVisibility(segment_id):
                ids.InsertNextValue(segment_id)
                names.append(segmentation.GetSegment(segment_id).GetName())
        if ids.GetNumberOfValues() == 0:
            self.segmentation_status.setText(f"'{source.GetName()}': no visible segment - show the segment(s) to use.")
            return
        if hasattr(self, "_stop_guide_edit"):
            self._stop_guide_edit()
        self._stop_lasso()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        labelmap, target = None, None
        try:
            labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Doggybone take-over (temp)")
            logic = slicer.modules.segmentations.logic()
            if volume is not None:
                logic.ExportSegmentsToLabelmapNode(source, ids, labelmap, volume)
            else:
                logic.ExportSegmentsToLabelmapNode(source, ids, labelmap)
            array = slicer.util.arrayFromVolume(labelmap)
            if array is None or not np.any(array):
                # nothing to take over - the existing Doggybone segmentation stays untouched
                self.segmentation_status.setText("The visible segments are empty - nothing taken over.")
                return
            array[:] = (array > 0).astype(array.dtype)            # all visible segments -> one bone
            slicer.util.arrayFromVolumeModified(labelmap)
            # build the new segmentation FIRST; the old one is replaced only when the new one is valid
            target = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", own_name + " (new)")
            target.CreateDefaultDisplayNodes()
            target.SetReferenceImageGeometryParameterFromVolumeNode(volume if volume is not None else labelmap)
            logic.ImportLabelmapToSegmentationNode(labelmap, target)
            if target.GetSegmentation().GetNumberOfSegments() == 0:
                slicer.mrmlScene.RemoveNode(target)
                target = None
                self.segmentation_status.setText("The visible segments are empty - nothing taken over.")
                return
            old = slicer.mrmlScene.GetFirstNodeByName(own_name)
            if old is not None:
                slicer.mrmlScene.RemoveNode(old)
            target.SetName(own_name)
            target.GetSegmentation().GetNthSegment(0).SetName("Bones")
            smoothing = getattr(self, "smoothing", None)
            if smoothing is not None:
                target.GetSegmentation().SetConversionParameter("Smoothing factor", f"{smoothing.value}")
            target.CreateClosedSurfaceRepresentation()
            if volume is not None:
                landmark_logic.set_bone_volume(volume)
            if display is not None:
                display.SetVisibility(False)                       # the editor's copy stays, hidden
            self._bone_baseline = None
            self._invalidate_guide_surfaces()
            if hasattr(self, "_set_bone_opacity") and hasattr(self, "opacity_slider"):
                self._set_bone_opacity(self.opacity_slider.value)
        finally:
            if labelmap is not None and slicer.mrmlScene.IsNodePresent(labelmap):
                slicer.mrmlScene.RemoveNode(labelmap)
            qt.QApplication.restoreOverrideCursor()
        self.segmentation_status.setText(
            f"Bone segmentation taken over from '{source.GetName()}': " + ", ".join(names)
            + ". Hidden segments were left out. The editor's segmentation is hidden (unchanged).")

    def _bone_labelmap_copy(self, segmentation_node):
        segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)
        image = slicer.vtkOrientedImageData()
        segmentation_node.GetBinaryLabelmapRepresentation(segment_id, image)
        copy = slicer.vtkOrientedImageData()
        copy.DeepCopy(image)
        return copy

    def _bone_set_labelmap(self, segmentation_node, image):
        segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)
        slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
            image, segmentation_node, segment_id, slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE)

    def _bone_push_undo(self, segmentation_node):
        '''Before the FIRST clean-up edit keep a copy of the segmentation ('Delete' restores it).'''
        if segmentation_node is None or getattr(self, "_bone_baseline", None) is not None:
            return
        self._bone_baseline = self._bone_labelmap_copy(segmentation_node)

    def _bone_lasso_apply(self, polygon):
        '''Erase outside (keep only bone voxels whose projection lies inside the screen polygon) or
        erase inside (remove them) - through the full depth.'''
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None:
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._bone_push_undo(segmentation_node)
            image = self._bone_labelmap_copy(segmentation_node)
            extent = image.GetExtent()
            dims = (extent[5] - extent[4] + 1, extent[3] - extent[2] + 1, extent[1] - extent[0] + 1)
            voxels = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims)
            kk, jj, ii = np.nonzero(voxels)
            if len(kk) == 0:
                return
            matrix = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(matrix)
            M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
            renderer = self._guide_renderer()
            render_window = self._guide_view.renderWindow()
            width, height = render_window.GetSize()
            camera = renderer.GetActiveCamera()
            pm = camera.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)
            P = np.array([[pm.GetElement(r, c) for c in range(4)] for r in range(4)])
            A = (P @ M).astype(np.float32)          # voxel index -> clip coordinates
            ijk = np.stack([ii + extent[0], jj + extent[2], kk + extent[4]], axis=1).astype(np.float32)
            clip = ijk @ A[:, :3].T + A[:, 3]
            xs = (clip[:, 0] / clip[:, 3] + 1.0) * 0.5 * width
            ys = (clip[:, 1] / clip[:, 3] + 1.0) * 0.5 * height
            poly = np.asarray(polygon, dtype=np.float32)
            keep = np.zeros(len(xs), dtype=bool)
            box = (xs >= poly[:, 0].min()) & (xs <= poly[:, 0].max()) & (ys >= poly[:, 1].min()) & (ys <= poly[:, 1].max())
            candidates = np.nonzero(box)[0]
            if len(candidates):
                keep[candidates] = guide_logic.points_in_polygon(np.c_[xs[candidates], ys[candidates]], poly)
            inside_mode = getattr(self, "_bone_lasso_mode", "outside") == "inside"
            drop = keep if inside_mode else ~keep          # keep = inside the lasso
            removed = int(drop.sum())
            voxels[kk[drop], jj[drop], ii[drop]] = 0
            image.GetPointData().GetScalars().Modified()
            image.Modified()
            self._bone_set_labelmap(segmentation_node, image)
            self._invalidate_guide_surfaces()
            spacing = float(abs(np.linalg.det(M[:3, :3])))
            self.segmentation_status.setText(
                f"Erase {'inside' if inside_mode else 'outside'}: {removed * spacing:.0f} mm\N{SUPERSCRIPT THREE} "
                f"removed. 'Delete' restores the segmentation from before the clean-up.")
        finally:
            qt.QApplication.restoreOverrideCursor()
