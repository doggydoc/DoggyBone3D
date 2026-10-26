# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""GuideEditMixin: Guide footprint editing: suggestion, brush / lasso on the bone surface, undo, visibility while editing.

Part of DoggyboneWidget (see Doggybone.py). Methods were moved here unchanged; they rely on
attributes created in DoggyboneWidget.setup() and on methods of the other mixins.
"""

from DoggyboneUI.common import *


class GuideEditMixin:

    # ------------------------------------------------------------------
    # Guide planning - step 1: footprint on the bone SURFACE (mesh), editing, extrusion
    # ------------------------------------------------------------------
    # Two guides share the whole painting/extrusion workflow: the SAW guide (on the bone as it is)
    # and the REDUCTION guide (on the fragments in the corrected position). Every guide node name
    # below depends on the active mode, so the tools work on the selected guide's own nodes.
    def _bone_tag(self):
        """Guide objects are kept per bone: the femur keeps the plain names, the tibia gets ' tibia'."""
        box = getattr(self, "osteotomy_bone", None)
        return " tibia" if box is not None and box.currentText == "Tibia" else ""

    def _mode_name(self, base):
        base = base + self._bone_tag()
        return base if getattr(self, "_guide_mode", "saw") == "saw" else base + " (reduction)"

    def _guide_mode_is_reduction(self):
        return getattr(self, "_guide_mode", "saw") == "reduction"

    GUIDE_MESH_NAME = property(lambda self: self._mode_name("Doggybone guide bone surface"))
    GUIDE_FOOTPRINT_NAME = property(lambda self: self._mode_name("Doggybone guide footprint"))
    GUIDE_BODY_NAME = property(lambda self: self._mode_name("Doggybone guide body"))
    GUIDE_SEGMENTATION_NAME = "Doggybone guide"   # earlier voxel-based version (hidden if present)

    def _guide_direction_frame(self, side, bone):
        '''(d, shaft, across, error): d = unit vector to the chosen guide side, shaft = proximal->
        distal, across = d x shaft. Cranial from the limb's handedness: with shaft proximal->distal
        and ml medial->lateral, cranial = shaft x ml for a LEFT and ml x shaft for a RIGHT limb.'''
        shaft, ml, _, error = self._get_anatomical_frame(side, bone)
        if error:
            return None, None, None, error
        shaft = guide_logic.unit(shaft)
        ml = np.asarray(ml, dtype=float)
        ml = guide_logic.unit(ml - float(np.dot(ml, shaft)) * shaft)
        cranial = guide_logic.unit(np.cross(shaft, ml) if side == "left" else np.cross(ml, shaft))
        d = {"Cranial": cranial, "Caudal": -cranial, "Lateral": ml, "Medial": -ml}[self.guide_side_selector.currentText]
        return d, shaft, guide_logic.unit(np.cross(d, shaft)), None

    def _guide_mesh_node(self):
        return slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_MESH_NAME)

    def _guide_external(self, mesh, recompute=False):
        '''Per-vertex mask of surface facing the OUTSIDE of the bone (no pores, cancellous bone
        or medullary canal). Computed from the bone segmentation once per prepared surface and
        stored on the mesh; computed on demand for surfaces prepared before it existed.'''
        if not recompute and mesh.GetPointData().GetArray("External") is not None:
            return guide_logic.get_selection(mesh, name="External")
        if self._guide_mode_is_reduction():
            # moved fragment no longer matches the labelmap; its ROI is the outer surface already
            external = np.ones(mesh.GetNumberOfPoints(), dtype=bool)
            guide_logic.set_selection(mesh, external, name="External")
            return external
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
            return np.ones(mesh.GetNumberOfPoints(), dtype=bool)
        segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)
        segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
        labelmap = segment.GetRepresentation(
            slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
        if labelmap is None:
            return np.ones(mesh.GetNumberOfPoints(), dtype=bool)
        external = guide_logic.external_mask(mesh, labelmap, segment.GetLabelValue())
        guide_logic.set_selection(mesh, external, name="External")
        return external

    @staticmethod
    def _guide_outer(mesh):
        '''Outer-surface mask of the guide side (all True for meshes made before it existed).'''
        if mesh.GetPointData().GetArray("Outer") is None:
            return np.ones(mesh.GetNumberOfPoints(), dtype=bool)
        return guide_logic.get_selection(mesh, name="Outer")

    def onSuggestGuideFootprint(self, empty=False):
        '''Prepare the bone surface around the osteotomy and suggest a footprint on the chosen
        side. empty=True prepares the surface only (nothing coloured) - used when a brush / lasso
        tool is switched on before any footprint exists.'''
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        self._stop_guide_edit()
        plane_nodes = [slicer.mrmlScene.GetFirstNodeByName(name) for name in
                       (self._plane_node_name(side, bone), self._distal_plane_node_name(side, bone))]
        plane_nodes = [node for node in plane_nodes if node is not None]
        if not plane_nodes:
            self.guide_status.setText("Insert the cutting plane(s) in Osteotomy Planning first.")
            return
        d, shaft, across, error = self._guide_direction_frame(side, bone)
        if error:
            self.guide_status.setText(error)
            return
        bone_polydata = self._guide_source_polydata()
        if bone_polydata is None or bone_polydata.GetNumberOfPoints() == 0:
            self.guide_status.setText(
                "No corrected fragments - run the correction preview first." if self._guide_mode_is_reduction()
                else "No bone surface - run 'Apply' in 3D Segmentation first.")
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            origins = [np.asarray(self._get_plane_origin_normal(node)[0], dtype=float) for node in plane_nodes]
            origins = self._guide_mode_plane_origins(plane_nodes, origins)
            centre = np.mean(origins, axis=0)
            t_values = [float(np.dot(o - centre, shaft)) for o in origins]
            margin = float(self.guide_footprint_margin.value)
            half_width = 0.5 * float(self.guide_footprint_width.value)
            t_min, t_max = min(t_values) - margin, max(t_values) + margin
            mesh = self._extract_guide_roi(bone_polydata, centre, shaft, across, d,
                                           (t_min - 15.0, t_max + 15.0), half_width + 20.0, 45.0)
            if mesh.GetNumberOfPoints() == 0:
                self.guide_status.setText("No bone surface found around the osteotomy.")
                return
            # Re-centre across the bone (cross-section centroid at the level of the cuts); the
            # diaphyseal shaft axis can lie outside a curved metaphysis.
            P, _ = guide_logic.points_and_normals(mesh)
            rel = P - centre
            slab = np.abs(rel @ shaft - 0.5 * (t_min + t_max)) < 3.0
            if np.any(slab):
                shift = P[slab].mean(axis=0) - centre
                centre = centre + shift - float(np.dot(shift, shaft)) * shaft
            selected = guide_logic.orient_and_suggest(mesh, centre, shaft, across, d, t_min, t_max, half_width)
            outer = guide_logic.outer_mask(mesh, centre, shaft, across, d)
            external = self._guide_external(mesh, recompute=True)
            outer = outer & external
            selected = selected & outer
            if empty:
                selected = np.zeros_like(selected)
            guide_logic.set_selection(mesh, selected)
            guide_logic.set_selection(mesh, outer, name="Outer")

            mesh_node = self._guide_mesh_node()
            if mesh_node is None:
                mesh_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_MESH_NAME)
                mesh_node.CreateDefaultDisplayNodes()
            mesh_node.SetAndObservePolyData(mesh)
            mesh_node.GetDisplayNode().SetVisibility(False)   # computation only; you see the bone
            mesh_node.SetAttribute("Doggybone.CorrectionSignature", self._correction_signature())
            self._set_guide_body(None)   # an old body does not match the new footprint
            old = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_SEGMENTATION_NAME)
            if old is not None and old.GetDisplayNode() is not None:
                old.GetDisplayNode().SetVisibility(False)
            area = self._update_guide_footprint_display()
            pin_note = "" if self._guide_mode_is_reduction() else self._show_saw_pin_preview()
            if empty:
                self.guide_status.setText(
                    "Bone surface around the osteotomy prepared - paint the footprint with Brush + / "
                    "Lasso + (rotate the view to reach every side)." + (f"\n{pin_note}" if pin_note else ""))
                return
            self.guide_status.setText(
                f"Footprint suggested on the {self.guide_side_selector.currentText.lower()} side of the "
                f"{side} {bone.lower()}: {int(selected.sum())} surface points, ~{area:.0f} mm\N{SUPERSCRIPT TWO}. "
                f"Refine with brush / lasso, then 'Extrude'." + (f"\n{pin_note}" if pin_note else ""))
        finally:
            qt.QApplication.restoreOverrideCursor()

    def _update_guide_footprint_display(self):
        mesh_node = self._guide_mesh_node()
        if mesh_node is None:
            return 0.0
        mesh = mesh_node.GetPolyData()
        patch = guide_logic.patch_from_selection(mesh, guide_logic.get_selection(mesh) & self._guide_external(mesh), lift=0.08)
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_FOOTPRINT_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_FOOTPRINT_NAME)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(1.0, 0.5, 0.05)
            node.GetDisplayNode().SetScalarVisibility(False)
        node.SetAndObservePolyData(patch)
        node.GetDisplayNode().SetVisibility(True)
        if patch.GetNumberOfCells() == 0:
            return 0.0
        mass = vtk.vtkMassProperties()
        mass.SetInputData(patch)
        mass.Update()
        return float(mass.GetSurfaceArea())

    # --- editing tools -------------------------------------------------
    def _correction_signature(self):
        """Fingerprint of the current correction (incl. fine shift) - the reduction surface must be
        rebuilt when it changes, since the moving fragment is somewhere else then."""
        if not self._guide_mode_is_reduction():
            return ""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        transform = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[2])
        if transform is None:
            return "none"
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)
        return ";".join(f"{matrix.GetElement(r, c):.3f}" for r in range(3) for c in range(4))

    def _ensure_current_reduction_surface(self):
        """Reduction guide: rebuild the prepared surface if the correction changed since it was made.
        Returns True if it was rebuilt (the old footprint no longer fits and is discarded)."""
        if not self._guide_mode_is_reduction():
            return False
        node = self._guide_mesh_node()
        if node is None or node.GetAttribute("Doggybone.CorrectionSignature") == self._correction_signature():
            return False
        self.onSuggestGuideFootprint(empty=True)
        self.guide_status.setText("The correction changed since the reduction surface was prepared - it was "
                                  "rebuilt on the fragments in the CURRENT corrected position (old footprint "
                                  "discarded). Paint again with Brush + / Lasso +.")
        return True

    def onGuideToolToggled(self, key, checked):
        if checked and self._ensure_current_reduction_surface():
            button = self._guide_tool_buttons[key]
            was_blocked = button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(was_blocked)
        if not checked:
            if getattr(self, "_guide_active_tool", None) == key:
                self._stop_guide_edit()   # placement stays paused until a landmark is clicked
                if key == "pins" and getattr(self, "_pins_dirty", False):
                    self._pins_dirty = False
                    self.onExtrudeGuideBody()          # marked pins -> sleeves and bores
            return
        if self._guide_mesh_node() is None:
            # No surface yet (e.g. new scene): prepare it now instead of refusing the tool.
            self.onSuggestGuideFootprint(empty=True)
        if self._guide_mesh_node() is None:
            message = getattr(self.guide_status, "full_text", self.guide_status.text)   # why the surface could not be prepared
            button = self._guide_tool_buttons[key]
            was_blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(was_blocked)
            slicer.util.warningDisplay(f"Brush / lasso cannot start: {message}", windowTitle="DoggyBone3D")
            return
        if key == "pins" and not self._guide_mode_is_reduction():
            button = self._guide_tool_buttons[key]
            was_blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(was_blocked)
            self.guide_status.setText("Pins are set AFTER the reduction: switch to 'Reduction guide' and click "
                                      "on the corrected bone - the saw guide gets its (pre-angled) sleeves from them.")
            return
        for other, button in self._guide_tool_buttons.items():
            if other != key and button.checked:
                was_blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(was_blocked)
        previous = getattr(self, "_guide_active_tool", None)
        if previous in (None, "bone_lasso"):
            self._stop_lasso()
            self._pause_landmark_placement()   # no landmark placement while editing
            self._hide_preview_for_guide_edit()
        self._guide_active_tool = key
        self._install_guide_observers()
        texts = {"brush_add": "Brush +: hold the left mouse button and move over the bone surface to colour it.",
                 "brush_remove": "Brush -: hold the left mouse button and move over the coloured area to remove it.",
                 "pins": "Pins: click on the guide to mark where a pin should go (click a mark again to remove "
                        "it). Not in the wedge and not in the saw blocks. 'Finish pins' (or Extrude) builds them.",
                 "lasso_add": "Lasso +: draw around the area to colour (only the visible surface is affected).",
                 "lasso_remove": "Lasso -: draw around the area to uncolour (only the visible surface is affected)."}
        hint = (" Drag on the background to rotate." if key.startswith("brush") or key == "pins"
                else " The lasso switches off after each stroke (click it again for the next one).")
        self.guide_status.setText(texts[key] + hint + " Shift+drag pans, right button / wheel zooms. "
                                  "Click the tool again to stop.")

    def _hide_preview_for_guide_edit(self, bone_lasso=False):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        self._guide_hidden_preview = []
        self._guide_locked_planes = []
        # painting: the bone surface fully OPAQUE (only the outer surface visible, no spongiosa /
        # medullary canal shining through); the user's opacity comes back when the tool is switched off
        already = {entry[0] for entry in getattr(self, "_guide_saved_opacity", [])}
        self._guide_saved_opacity = list(getattr(self, "_guide_saved_opacity", []))
        for name in (self.SEGMENTATION_NODE_NAME, self._preview_node_names(side, bone)[3]):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            display = node.GetDisplayNode() if node is not None else None
            if display is None:
                continue
            if node.GetID() in already:                 # saved before (tool switched): keep the user's value
                display.SetOpacity3D(1.0) if hasattr(display, "GetOpacity3D") else display.SetOpacity(1.0)
                continue
            if hasattr(display, "GetOpacity3D"):
                self._guide_saved_opacity.append((node.GetID(), "3d", float(display.GetOpacity3D())))
                display.SetOpacity3D(1.0)
            else:
                self._guide_saved_opacity.append((node.GetID(), "model", float(display.GetOpacity())))
                display.SetOpacity(1.0)
        # Painting the guide footprint: the cutting planes STAY visible for orientation (the lasso's
        # visibility test renders the bone surface alone, so they do not hide anything) - only locked,
        # so a stroke never grabs a plane handle. The bone lasso (segmentation) and the X/Y gizmo: hidden.
        plane_names = (self._plane_node_name(side, bone), self._distal_plane_node_name(side, bone))
        for name in plane_names + (self._fine_shift_node_names(side, bone)[1],):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None or node.GetDisplayNode() is None or not node.GetDisplayNode().GetVisibility():
                continue
            if name in plane_names and not bone_lasso:
                display = node.GetDisplayNode()
                self._guide_locked_planes.append((node.GetID(), bool(node.GetLocked()),
                                                  bool(display.GetHandlesInteractive())))
                node.SetLocked(True)
                display.SetHandlesInteractive(False)
                continue
            node.GetDisplayNode().SetVisibility(False)
            self._guide_hidden_preview.append(node.GetID())
        for name in self._preview_node_names(side, bone):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.IsA("vtkMRMLModelNode") and node.GetDisplayNode() is not None \
                    and node.GetDisplayNode().GetVisibility():
                node.GetDisplayNode().SetVisibility(False)
                self._guide_hidden_preview.append(node.GetID())
        # the measurement axes (shaft axis ...) run through the osteotomy and look like material in the
        # cut while painting - hidden during editing, shown again afterwards
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            name = node.GetName() or ""
            display = node.GetDisplayNode()
            if name.startswith(f"Doggybone {side} ") and name.endswith(" axis") and display is not None \
                    and display.GetVisibility():
                display.SetVisibility(False)
                self._guide_hidden_preview.append(node.GetID())
        bone_segmentation = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if self._guide_mode_is_reduction() and not bone_lasso:
            # paint on what the reduction guide sits on: the fragments in the corrected position
            if bone_segmentation is not None and bone_segmentation.GetDisplayNode() is not None \
                    and bone_segmentation.GetDisplayNode().GetVisibility():
                bone_segmentation.GetDisplayNode().SetVisibility(False)
                self._guide_hidden_preview.append(bone_segmentation.GetID())
            combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
            if combined is not None and combined.GetDisplayNode() is not None:
                combined.GetDisplayNode().SetVisibility(True)
                if combined.GetID() in self._guide_hidden_preview:
                    self._guide_hidden_preview.remove(combined.GetID())
                self._show_only_guide_surface(combined)
            return
        if bone_segmentation is not None and bone_segmentation.GetDisplayNode() is not None:
            bone_segmentation.GetDisplayNode().SetVisibility(True)      # may be off after a guide-mode switch
            bone_segmentation.GetDisplayNode().SetVisibility3D(True)
        if not bone_lasso:
            self._show_only_guide_surface(bone_segmentation)

    def _show_only_guide_surface(self, bone_node):
        """While painting: the femur surface around the osteotomy alone - patella, fabellae, tibia are hidden,
        so the footprint can be painted where they lie close to it (the tools only ever paint this surface)."""
        mesh_node = self._guide_mesh_node()
        if mesh_node is None or mesh_node.GetDisplayNode() is None or bone_node is None \
                or bone_node.GetDisplayNode() is None:
            return
        if bone_node.GetDisplayNode().GetVisibility():
            bone_node.GetDisplayNode().SetVisibility(False)
            self._guide_hidden_preview.append(bone_node.GetID())
        display = mesh_node.GetDisplayNode()
        if not display.GetVisibility():
            display.SetColor(0.93, 0.87, 0.6)
            display.SetOpacity(1.0)
            display.SetVisibility(True)
            self._guide_shown_surface = mesh_node.GetID()

    def _install_guide_observers(self):
        self._remove_guide_observers()
        view = main_3d_widget().threeDView()          # the VISIBLE 3D view
        interactor = view.interactor()
        self._guide_view = view
        self._guide_tags = {}
        for event in ("LeftButtonPressEvent", "MouseMoveEvent", "LeftButtonReleaseEvent"):
            self._guide_tags[event] = interactor.AddObserver(event, self._on_guide_mouse, 1.0)
        self._guide_observers = [(interactor, tag) for tag in self._guide_tags.values()]
        if self._guide_active_tool == "bone_lasso":
            return   # the bone lasso needs no surface locators
        mesh = self._guide_mesh_node().GetPolyData()
        self._guide_cell_locator = vtk.vtkCellLocator()
        self._guide_cell_locator.SetDataSet(mesh)
        self._guide_cell_locator.BuildLocator()
        self._guide_bone_locator = self._guide_cell_locator
        self._guide_point_locator = vtk.vtkPointLocator()
        self._guide_point_locator.SetDataSet(mesh)
        self._guide_point_locator.BuildLocator()

    def _remove_guide_observers(self):
        for interactor, tag in getattr(self, "_guide_observers", []):
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                logging.exception("Doggybone: non-critical step failed (continuing)")
        self._guide_observers = []
        self._guide_drag = False
        self._clear_lasso_feedback()

    def _abort_guide_event(self, caller, event):
        tag = getattr(self, "_guide_tags", {}).get(event)
        if tag is not None:
            command = caller.GetCommand(tag)
            if command is not None:
                command.SetAbortFlag(1)

    def _on_guide_mouse(self, caller, event):
        tool = getattr(self, "_guide_active_tool", None)
        if tool is None:
            return
        x, y = caller.GetEventPosition()
        if event == "LeftButtonPressEvent":
            # Cmd/Ctrl + click on a pin (reduction): select it for the gizmo
            if tool == "pins" and caller.GetControlKey() and not caller.GetShiftKey() \
                    and self._guide_mode_is_reduction():
                mark = self._pin_under_mouse(x, y)
                if mark is not None:
                    self._guide_drag = False
                    self._abort_guide_event(caller, event)
                    qt.QTimer.singleShot(0, lambda i=mark: self._select_reduction_pin(i, announce=True))
                    return
            # With Shift / Ctrl / Alt the mouse keeps its normal function (Shift+drag = pan).
            if caller.GetShiftKey() or caller.GetControlKey() or caller.GetAltKey():
                self._guide_drag = False
                return
            if tool == "pins":
                self._guide_drag = False
                if self._pin_gizmo_hovered():
                    return                         # the gizmo gets the drag, no new pin
                mark = self._pin_under_mouse(x, y)
                hit = None if mark is not None else self._guide_pick(x, y)
                if mark is None and hit is None:
                    return                         # background: rotate the view
                self._abort_guide_event(caller, event)
                if mark is not None:
                    qt.QTimer.singleShot(0, lambda i=mark: self._remove_pin(i))
                else:
                    qt.QTimer.singleShot(0, lambda h=list(hit): self._toggle_pin(h))
                return
            if tool.startswith("brush"):
                hit = self._guide_pick(x, y)
                if hit is None:
                    self._guide_drag = False      # drag started on the background: rotate the view
                    return
                self._guide_drag = True
                self._guide_brush_at(x, y, tool == "brush_add", hit=hit)
            else:
                self._guide_drag = True
                self._guide_lasso_points = [(x, y)]
                self._update_lasso_feedback()
            self._abort_guide_event(caller, event)
        elif event == "MouseMoveEvent":
            if not self._guide_drag:
                return
            if tool.startswith("brush"):
                self._guide_brush_at(x, y, tool == "brush_add")
            else:
                last = self._guide_lasso_points[-1]
                if abs(x - last[0]) + abs(y - last[1]) >= 3:
                    self._guide_lasso_points.append((x, y))
                    self._update_lasso_feedback()
            self._abort_guide_event(caller, event)
        elif event == "LeftButtonReleaseEvent":
            if not self._guide_drag:
                return
            self._guide_drag = False
            if tool == "bone_lasso":
                points = list(getattr(self, "_guide_lasso_points", []))
                self._clear_lasso_feedback()
                self._abort_guide_event(caller, event)
                if len(points) >= 3:
                    self._bone_lasso_apply(points)
                qt.QTimer.singleShot(0, self._stop_lasso)
                return
            if tool.startswith("lasso"):
                points = list(getattr(self, "_guide_lasso_points", []))
                self._clear_lasso_feedback()
                if len(points) >= 3:
                    self._guide_apply_lasso(points, tool == "lasso_add")
            self._refresh_guide_body_if_present()
            self._abort_guide_event(caller, event)
            if tool.startswith("lasso"):
                # One lasso per click: switch the tool off so the view can be rotated/panned again.
                button = self._guide_tool_buttons.get(tool)
                qt.QTimer.singleShot(0, lambda b=button: b.setChecked(False) if b is not None else None)

    def _guide_renderer(self):
        return self._guide_view.renderWindow().GetRenderers().GetFirstRenderer()

    def _guide_pick(self, x, y):
        '''First bone-surface point under the display position (x, y), or None.'''
        renderer = self._guide_renderer()
        ends = []
        for z in (0.0, 1.0):
            renderer.SetDisplayPoint(float(x), float(y), z)
            renderer.DisplayToWorld()
            w = renderer.GetWorldPoint()
            ends.append([w[0] / w[3], w[1] / w[3], w[2] / w[3]])
        t = vtk.reference(0.0)
        sub_id = vtk.reference(0)
        hit = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        if not self._guide_cell_locator.IntersectWithLine(ends[0], ends[1], 0.0, t, hit, pcoords, sub_id):
            return None
        return hit

    def _guide_brush_at(self, x, y, add, hit=None):
        renderer = self._guide_renderer()
        if hit is None:
            hit = self._guide_pick(x, y)
            if hit is None:
                return
        ids = vtk.vtkIdList()
        radius = float(self.guide_brush_radius.value)
        self._guide_point_locator.FindPointsWithinRadius(radius, hit, ids)
        if ids.GetNumberOfIds() == 0:
            return
        idx = np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=np.int64)
        mesh = self._guide_mesh_node().GetPolyData()
        P, N = guide_logic.points_and_normals(mesh)
        camera = np.asarray(renderer.GetActiveCamera().GetPosition(), dtype=float)
        facing = np.sum(N[idx] * (camera - P[idx]), axis=1) > 0.0
        keep = facing & self._guide_external(mesh)[idx] & (
            self._guide_outer(mesh)[idx] if self.guide_side_only_checkbox.checked
            else np.ones(len(idx), dtype=bool))
        # Line of sight: a point counts only if nothing of the bone lies between it and the camera
        # (no painting of cancellous / inner surfaces behind the visible cortex).
        t = vtk.reference(0.0); sub_id = vtk.reference(0); hit_point = [0.0, 0.0, 0.0]; pc = [0.0, 0.0, 0.0]
        for n, i in enumerate(idx):
            if not keep[n]:
                continue
            direction = P[i] - camera
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue
            end = P[i] - direction / length * 0.3
            if self._guide_cell_locator.IntersectWithLine(camera.tolist(), end.tolist(), 0.0, t, hit_point, pc, sub_id):
                keep[n] = False
        selection = guide_logic.get_selection(mesh)
        selection[idx[keep]] = add
        guide_logic.set_selection(mesh, selection)
        self._update_guide_footprint_display()

    def _guide_apply_lasso(self, polygon, add):
        from vtk.util.numpy_support import vtk_to_numpy
        render_window = self._guide_view.renderWindow()
        renderer = self._guide_renderer()
        render_window.Render()
        width, height = render_window.GetSize()
        mesh = self._guide_mesh_node().GetPolyData()
        P, N = guide_logic.points_and_normals(mesh)
        camera = renderer.GetActiveCamera()
        m = camera.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)
        M = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])
        hom = np.c_[P, np.ones(len(P))] @ M.T
        ndc = hom[:, :3] / hom[:, 3:4]
        vx0, vy0, vx1, vy1 = renderer.GetViewport()
        xs = (ndc[:, 0] + 1.0) * 0.5 * (vx1 - vx0) * width + vx0 * width
        ys = (ndc[:, 1] + 1.0) * 0.5 * (vy1 - vy0) * height + vy0 * height
        depth = (ndc[:, 2] + 1.0) * 0.5
        # Depth from the bone surface ONLY: slice planes shown in 3D, cutting planes, gizmos and
        # other models must not hide the bone for the visibility test.
        Z = self._guide_surface_depth(render_window, renderer, mesh, width, height)
        xi = np.clip(np.round(xs).astype(int), 0, width - 1)
        yi = np.clip(np.round(ys).astype(int), 0, height - 1)
        on_screen = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        visible = on_screen & (depth <= Z[yi, xi] + 2e-3)
        facing = np.sum(N * (np.asarray(camera.GetPosition(), dtype=float) - P), axis=1) > 0.0
        inside = guide_logic.points_in_polygon(np.c_[xs, ys], polygon)
        affected = inside & visible & facing & self._guide_external(mesh)
        if self.guide_side_only_checkbox.checked:
            affected &= self._guide_outer(mesh)
        selection = guide_logic.get_selection(mesh)
        selection[affected] = add
        guide_logic.set_selection(mesh, selection)
        area = self._update_guide_footprint_display()
        self.guide_status.setText(f"{'Added' if add else 'Removed'} {int(affected.sum())} surface points; "
                                  f"footprint now ~{area:.0f} mm\N{SUPERSCRIPT TWO}.")

    def _guide_surface_depth(self, render_window, renderer, mesh, width, height):
        '''Z-buffer of the 3D view with only the prepared bone surface rendered (all other props
        hidden for one frame, then restored).'''
        from vtk.util.numpy_support import vtk_to_numpy
        props = renderer.GetViewProps()
        props.InitTraversal()
        hidden = []
        for _ in range(props.GetNumberOfItems()):
            prop = props.GetNextProp()
            if prop is not None and prop.GetVisibility():
                prop.VisibilityOff()
                hidden.append(prop)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        try:
            render_window.Render()
            zbuffer = vtk.vtkFloatArray()
            render_window.GetZbufferData(0, 0, width - 1, height - 1, zbuffer)
            Z = vtk_to_numpy(zbuffer).reshape(height, width).copy()
        finally:
            renderer.RemoveActor(actor)
            for prop in hidden:
                prop.VisibilityOn()
            render_window.Render()
        return Z

    def _update_lasso_feedback(self):
        renderer = self._guide_renderer()
        if getattr(self, "_guide_lasso_actor", None) is None:
            self._guide_lasso_polydata = vtk.vtkPolyData()
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(self._guide_lasso_polydata)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.55, 0.05)
            actor.GetProperty().SetLineWidth(2)
            renderer.AddActor2D(actor)
            self._guide_lasso_actor = actor
        points = vtk.vtkPoints()
        for x, y in self._guide_lasso_points:
            points.InsertNextPoint(float(x), float(y), 0.0)
        lines = vtk.vtkCellArray()
        n = len(self._guide_lasso_points)
        if n >= 2:
            lines.InsertNextCell(n + 1)
            for i in range(n):
                lines.InsertCellPoint(i)
            lines.InsertCellPoint(0)
        self._guide_lasso_polydata.SetPoints(points)
        self._guide_lasso_polydata.SetLines(lines)
        self._guide_lasso_polydata.Modified()
        self._guide_view.renderWindow().Render()

    def _clear_lasso_feedback(self):
        actor = getattr(self, "_guide_lasso_actor", None)
        if actor is not None:
            try:
                self._guide_renderer().RemoveActor2D(actor)
                self._guide_view.renderWindow().Render()
            except Exception:
                logging.exception("Doggybone: non-critical step failed (continuing)")
        self._guide_lasso_actor = None


    def _set_guide_body(self, polydata):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if polydata is None:
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
            return
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_BODY_NAME)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(0.35, 0.7, 0.95)
            node.GetDisplayNode().SetOpacity(0.85)
        node.SetAndObservePolyData(polydata)
        # remember what this guide is for - the export names the file from these, not from the
        # side / bone selected at export time
        node.SetAttribute("Doggybone.GuideSide", self.osteotomy_side.currentText)
        node.SetAttribute("Doggybone.GuideBone", self.osteotomy_bone.currentText)
        node.SetAttribute("Doggybone.GuideMode", getattr(self, "_guide_mode", "saw"))
        # fingerprint of everything the guide was built from - the export compares it with the scene
        try:
            node.SetAttribute("Doggybone.PlanSignature", json.dumps(self._guide_plan_signature(
                self.osteotomy_side.currentText, self.osteotomy_bone.currentText, getattr(self, "_guide_mode", "saw"))))
        except Exception:
            logging.exception("Doggybone: guide plan signature failed (the export cannot check this guide)")


    def onGuideDelete(self):
        if not self._guide_mode_is_reduction():
            for label in ("proximal", "distal"):
                node = slicer.mrmlScene.GetFirstNodeByName(self._saw_box_gizmo_name(label))
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
        '''Delete this guide: guide body, this mode's pins (+ axes, row handles) and - if there is one -
        the painted footprint (the surface stays, so painting can start from scratch).'''
        self._stop_guide_edit()
        mesh_node = self._guide_mesh_node()
        if mesh_node is not None and mesh_node.GetPolyData() is not None:
            mesh = mesh_node.GetPolyData()
            guide_logic.set_selection(mesh, np.zeros(mesh.GetNumberOfPoints(), dtype=bool))
            self._update_guide_footprint_display()
        pins_node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        if pins_node is not None:
            pins_node.RemoveAllControlPoints()          # this mode's own pin marks
        if self._guide_mode_is_reduction():
            handles = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
            if handles is not None:
                handles.RemoveAllControlPoints()
            self._pin_rows_active = False
            self._select_reduction_pin(None)
        self._pins_dirty = False
        self._set_pin_axes(None)
        self._set_guide_body(None)
        if self._guide_mode_is_reduction():
            self.guide_status.setText("Reduction guide and its pins deleted. Place the pin rows again.")
        else:
            self.guide_status.setText("Footprint and guide body deleted. Paint again with Brush + / Lasso +.")
        if not self._guide_mode_is_reduction():
            self._show_saw_pin_preview()

    def _refresh_guide_body_if_present(self):
        '''After a footprint edit or a parameter change: an existing body is rebuilt so it always
        matches - DEBOUNCED: several strokes / spin-box steps in a row cost ONE rebuild (0.8 s after
        the last change), the painting itself never waits for the volume body.'''
        if slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME) is None:
            return
        timer = getattr(self, "_guide_refresh_timer", None)
        if timer is None:
            timer = qt.QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_guide_refresh)
            self._guide_refresh_timer = timer
        self._guide_refresh_for = (getattr(self, "_guide_mode", "saw"), self.osteotomy_bone.currentText)
        timer.start(800)

    def _run_guide_refresh(self):
        # mode or bone switched meanwhile: the scheduled rebuild was for the other guide - skip it
        if getattr(self, "_guide_refresh_for", None) != (getattr(self, "_guide_mode", "saw"),
                                                           self.osteotomy_bone.currentText):
            return
        if slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME) is not None:
            self._extrude_guide_body()

    def _stop_guide_edit(self):
        self._remove_guide_observers()
        for button in getattr(self, "_guide_tool_buttons", {}).values():
            if button.checked:
                was_blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(was_blocked)
        self._guide_active_tool = None
        button = getattr(self, "place_pin_rows_button", None)
        if button is not None and button.text != "Place pins":
            button.setText("Place pins")          # manual pin placement ended with the tool
        self._restore_guide_hidden()

    def _restore_guide_hidden(self):
        shown = getattr(self, "_guide_shown_surface", None)
        if shown is not None:
            node = slicer.mrmlScene.GetNodeByID(shown)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(False)
            self._guide_shown_surface = None
        for node_id, kind, value in getattr(self, "_guide_saved_opacity", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            display = node.GetDisplayNode() if node is not None else None
            if display is None:
                continue
            if kind == "3d":
                display.SetOpacity3D(value)
            else:
                display.SetOpacity(value)
        self._guide_saved_opacity = []
        for node_id, locked, handles in getattr(self, "_guide_locked_planes", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                node.SetLocked(locked)
                if node.GetDisplayNode() is not None:
                    node.GetDisplayNode().SetHandlesInteractive(handles)
        self._guide_locked_planes = []
        hidden_originals = {}                     # original overlay hidden meanwhile: only note it
        for saved in self.__dict__.get("_hidden_original_overlay", {}).values():
            for node_id in saved:
                hidden_originals[node_id] = saved
        for node_id in getattr(self, "_guide_hidden_preview", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and node.GetDisplayNode() is not None:
                if node_id in hidden_originals:
                    hidden_originals[node_id][node_id] = True
                    continue
                node.GetDisplayNode().SetVisibility(True)
        self._guide_hidden_preview = []
        # back to the module-wide bone opacity (it may have been changed while painting)
        if hasattr(self, "_set_bone_opacity") and getattr(self, "opacity_slider", None) is not None:
            self._set_bone_opacity(self.opacity_slider.value)
