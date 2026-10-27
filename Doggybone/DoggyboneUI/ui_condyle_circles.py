# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""CondyleCirclesMixin: positioning aid for the two condyle centres (e.g. two condylar parts of a joint).

As soon as both centre points are placed, a circle is drawn around each of them, PERPENDICULAR to
the condylar axis (the line through the two centres) - i.e. in the sagittal plane of the condyle.
The radius of each circle is set with a spin box. Shown in a slice perpendicular to the condylar
axis (button "Yellow view: medial / lateral circle") the circle lies over the articular contour:
move the centre point until the circle follows the contour - circle and points update live.

The radius is variable: spin box, or drag the small handle "r" on the circle (in the slice view or
in 3D) - the radius follows the handle.

The circles are only a visual aid; the measurements use the two centre points. Display: a torus
(ring radius = circle radius, adjustable ring thickness); a slice perpendicular to the axis shows it
as a ring band, an oblique slice as an ellipse - a hint that the slice is not perpendicular.

Configured per limb with CONDYLE_CIRCLE_LANDMARKS = (medial centre, lateral centre) and optionally
CONDYLE_CIRCLE_SHAFT = (proximal, distal) for the orientation of the yellow view."""

from DoggyboneUI.common import *


class CondyleCirclesMixin:

    CONDYLE_CIRCLE_LANDMARKS = ()        # (medial centre, lateral centre) - set per limb
    CONDYLE_CIRCLE_SHAFT = ()            # (proximal, distal) shaft points: "up" in the yellow view
    CONDYLE_CIRCLE_DEFAULT_RADIUS = 8.0
    CONDYLE_CIRCLE_WIDTH = 1.5           # default ring thickness = tube diameter of the torus (mm)
    CONDYLE_CIRCLE_LINE = 4              # line thickness in the slice views (px)
    CONDYLE_CIRCLE_NODE_PREFIX = "Doggybone condyle circle "
    _CIRCLE_COLORS = ((0.2, 0.9, 1.0), (1.0, 0.35, 0.9))     # medial cyan, lateral magenta
    # landmark ON the condylar axis (distal point of the mechanical axis): starts in the middle, is moved
    # by hand to the narrowest point of the condyle and is always put back onto the axis
    CONDYLE_WAIST_LANDMARK = ""

    # ------------------------------------------------------------------ UI
    def _build_condyle_circles(self, layout):
        self._circle_observers = []
        self._circle_update_pending = False
        if len(self.CONDYLE_CIRCLE_LANDMARKS) != 2:
            self.condyle_circle_box = None
            return
        box = qt.QGroupBox("Condyle circles (positioning aid)")
        box.setToolTip(
            "After both condyle centres are placed: a circle around each centre, perpendicular to the condylar "
            "axis. Show it in the yellow view, set the radius to the articular contour and move the centre "
            "point until the circle follows the contour. Only a visual aid - the measurements use the points.")
        form = qt.QFormLayout(box)
        self._waist_state = {}
        self._waist_pending = False
        if self.CONDYLE_WAIST_LANDMARK:
            button = qt.QPushButton("Green view: frontal through the condylar axis")
            button.setToolTip(
                "Green slice view through the condylar axis and the shaft (frontal plane of the condyle): the "
                "condyle profile with trochlea and capitulum. Move the condyle centre ALONG the axis to the "
                "narrowest point - it always stays on the axis.")
            button.clicked.connect(lambda _checked=False: self._align_green_to_condylar_axis())
            form.addRow(button)
            self.condyle_waist_status = StatusLabel("")
            self.condyle_waist_status.setWordWrap(True)
            self.condyle_waist_status.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
            form.addRow(self.condyle_waist_status)
        self.condyle_circle_checkbox = qt.QCheckBox("Show circles")
        self.condyle_circle_checkbox.setChecked(True)
        self.condyle_circle_checkbox.toggled.connect(lambda _checked: self._schedule_condyle_circles())
        form.addRow(self.condyle_circle_checkbox)
        self.condyle_circle_spinboxes = []
        for index, (label, name) in enumerate(zip(("Medial radius", "Lateral radius"), self.CONDYLE_CIRCLE_LANDMARKS)):
            spin = qt.QDoubleSpinBox()
            spin.setRange(1.0, 40.0)
            spin.setToolTip("Circle radius - or drag the handle 'r' on the circle in a slice view / 3D.")
            spin.setSingleStep(0.25)
            spin.setDecimals(2)
            spin.setSuffix(" mm")
            spin.setValue(self.CONDYLE_CIRCLE_DEFAULT_RADIUS)
            spin.valueChanged.connect(lambda value, name=name: self._on_condyle_radius_changed(name, value))
            button = qt.QPushButton("Yellow view")
            button.setToolTip("Yellow slice view through this centre, perpendicular to the condylar axis.")
            button.clicked.connect(lambda _checked=False, name=name: self._align_yellow_to_condyle(name))
            row = qt.QWidget()
            row_layout = qt.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(spin, 1)
            row_layout.addWidget(button)
            form.addRow(label, row)
            self.condyle_circle_spinboxes.append(spin)
        self.condyle_ring_thickness = qt.QDoubleSpinBox()
        self.condyle_ring_thickness.setRange(0.2, 5.0)
        self.condyle_ring_thickness.setSingleStep(0.25)
        self.condyle_ring_thickness.setDecimals(2)
        self.condyle_ring_thickness.setSuffix(" mm")
        self.condyle_ring_thickness.setValue(self.CONDYLE_CIRCLE_WIDTH)
        self.condyle_ring_thickness.setToolTip("Thickness of the ring (tube diameter of the torus).")
        self.condyle_ring_thickness.valueChanged.connect(lambda _v: self._schedule_condyle_circles())
        form.addRow("Ring thickness", self.condyle_ring_thickness)
        self.condyle_circle_status = StatusLabel("")
        self.condyle_circle_status.setWordWrap(True)
        self.condyle_circle_status.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        form.addRow(self.condyle_circle_status)
        layout.addWidget(box)
        self.condyle_circle_box = box
        self._install_condyle_circle_observers()
        self._schedule_condyle_circles()

    def _install_condyle_circle_observers(self):
        self._remove_condyle_circle_observers()
        for panel in (getattr(self, "left_panel", None), getattr(self, "right_panel", None)):
            node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id) if panel is not None else None
            if node is None:
                continue
            for event in (slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                          slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                          slicer.vtkMRMLMarkupsNode.PointRemovedEvent):
                tag = node.AddObserver(event, lambda caller, ev: self._schedule_condyle_circles())
                self._circle_observers.append((node, tag))
            if self.CONDYLE_WAIST_LANDMARK:
                for event in (slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                              slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                              slicer.vtkMRMLMarkupsNode.PointRemovedEvent):
                    tag = node.AddObserver(event, lambda caller, ev: self._schedule_waist())
                    self._circle_observers.append((node, tag))

    def _remove_condyle_circle_observers(self):
        self._remove_circle_handle_observers()
        for node, tag in getattr(self, "_circle_observers", []):
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._circle_observers = []

    # ------------------------------------------------------------------ data
    def _condyle_circle_panel(self):
        panel = self._current_panel() if hasattr(self, "_current_panel") else None
        return panel if panel is not None else getattr(self, "right_panel", None)

    def _condyle_positions(self, panel):
        by_name = {landmark.name: landmark for landmark in panel.landmarks}
        positions = []
        for name in self.CONDYLE_CIRCLE_LANDMARKS:
            landmark = by_name.get(name)
            if landmark is None or not landmark.placed:
                return None
            position = landmark.get_position()
            if position is None:
                return None
            positions.append(np.array(position, dtype=float))
        return positions

    def _condyle_radius(self, panel, name):
        node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id)
        text = node.GetAttribute(f"Doggybone.circle.{name}") if node is not None else None
        try:
            return float(text) if text else self.CONDYLE_CIRCLE_DEFAULT_RADIUS
        except ValueError:
            return self.CONDYLE_CIRCLE_DEFAULT_RADIUS

    def _on_condyle_radius_changed(self, name, value):
        panel = self._condyle_circle_panel()
        node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id) if panel is not None else None
        if node is not None:
            node.SetAttribute(f"Doggybone.circle.{name}", f"{value:.3f}")
        self._schedule_condyle_circles()

    # ------------------------------------------------------------------ display
    def _schedule_condyle_circles(self):
        if getattr(self, "_circle_update_pending", False) or getattr(self, "condyle_circle_box", None) is None:
            return
        self._circle_update_pending = True

        def run():
            self._circle_update_pending = False
            try:
                self._update_condyle_circles()
            except Exception:
                logging.exception("Doggybone: condyle circles failed (continuing)")
        qt.QTimer.singleShot(0, run)

    # ------------------------------------------------------------------ condyle centre (narrowest point)
    def _schedule_waist(self):
        if not self.CONDYLE_WAIST_LANDMARK or getattr(self, "_waist_pending", False) \
                or getattr(self, "_waist_busy", False):
            return
        self._waist_pending = True

        def run():
            self._waist_pending = False
            try:
                self._update_waist()
            except Exception:
                logging.exception("Doggybone: condyle centre failed (continuing)")
        qt.QTimer.singleShot(0, run)

    def _waist_landmark(self, panel):
        landmark = next((l for l in panel.landmarks if l.name == self.CONDYLE_WAIST_LANDMARK), None)
        if landmark is not None and not getattr(landmark, "always_movable", False):
            landmark.always_movable = True               # can be dragged at any time - it is kept on the axis
            node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id)
            if landmark.placed and landmark._id is not None and node is not None:
                node.SetNthControlPointLocked(landmark._id, False)
        return landmark

    def _shaft_up(self, panel, axis):
        """Proximal direction perpendicular to the condylar axis (from the shaft points)."""
        up = np.array([0.0, 0.0, 1.0])
        by_name = {landmark.name: landmark for landmark in panel.landmarks}
        if len(self.CONDYLE_CIRCLE_SHAFT) == 2:
            a, b = (by_name.get(n) for n in self.CONDYLE_CIRCLE_SHAFT)
            if a is not None and b is not None and a.placed and b.placed:
                up = np.array(a.get_position(), dtype=float) - np.array(b.get_position(), dtype=float)
        up = up - np.dot(up, axis) * axis
        if np.linalg.norm(up) < 1e-6:
            up = np.cross(axis, [1.0, 0.0, 0.0])
        return up / np.linalg.norm(up)

    def _align_green_to_condylar_axis(self):
        panel = self._condyle_circle_panel()
        positions = self._condyle_positions(panel) if panel is not None else None
        if positions is None:
            self.condyle_waist_status.setText("Place both condyle centres first.")
            return
        medial, lateral = positions
        axis = lateral - medial
        if np.linalg.norm(axis) < 1.0:
            return
        axis = axis / np.linalg.norm(axis)
        up = self._shaft_up(panel, axis)
        landmark = self._waist_landmark(panel)
        centre = np.array(landmark.get_position(), float) if landmark is not None and landmark.placed \
            else 0.5 * (medial + lateral)
        normal = np.cross(axis, up)                       # cranio-caudal: the slice holds axis + shaft
        slice_node = slicer.app.layoutManager().sliceWidget("Green").mrmlSliceNode()
        slice_node.SetSliceToRASByNTP(*normal, *axis, *centre, 0)
        self.condyle_waist_status.setText(
            "Green view: frontal through the condylar axis. Move the condyle centre along the axis to the "
            "narrowest point of the condyle.")

    def _set_waist_position(self, panel, landmark, position):
        node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id)
        if node is None:
            return
        position = [float(c) for c in position]
        if landmark.placed and landmark._id is not None:
            node.SetNthControlPointPosition(landmark._id, *position)
        else:
            # another landmark may be waiting for a click (place mode on the same list): pause it,
            # otherwise it would take this point as its own
            current = None
            try:
                current = panel.measurement_stack.currentWidget().landmark_stack.currentWidget().landmark
            except Exception:
                pass
            paused = current is not None and current is not landmark and not current.placed
            if paused:
                current.stop_interaction()
            try:
                index = node.AddControlPoint(*position)
                if not landmark.placed:                       # its own place mode did not catch it
                    node.SetNthControlPointLabel(index, landmark.name)
                    landmark._id = index
                    landmark.placed = True
                    node.SetNthControlPointLocked(index, False)
            finally:
                if paused:
                    current.start_interaction()
        for callback in landmark.change_callbacks:
            callback()

    def _update_waist(self):
        """Keep the condyle centre ON the condylar axis: placed at the middle when both condyle centres
        exist, projected back onto the axis when moved, and kept at its place on the axis (same
        fraction) when a condyle centre moves."""
        panel = self._condyle_circle_panel()
        landmark = self._waist_landmark(panel) if panel is not None else None
        if landmark is None:
            return
        positions = self._condyle_positions(panel)
        if positions is None:
            self.condyle_waist_status.setText("Condyle centre: place both condyle centres first.")
            return
        medial, lateral = positions
        d = lateral - medial
        if np.linalg.norm(d) < 1.0:
            return
        current = np.array(landmark.get_position(), float) if landmark.placed else None
        state = self._waist_state.get(panel.side)
        centres_moved = state is not None and max(np.linalg.norm(medial - state["medial"]),
                                                  np.linalg.norm(lateral - state["lateral"])) > 0.05
        if current is None:
            fraction = 0.5
        elif centres_moved:
            fraction = state["fraction"]
        else:
            fraction = float(np.dot(current - medial, d) / np.dot(d, d))
        target = medial + fraction * d
        self._waist_busy = True
        try:
            if current is None or np.linalg.norm(target - current) > 0.05:
                self._set_waist_position(panel, landmark, target)
        finally:
            self._waist_busy = False
        self._waist_state[panel.side] = {"medial": medial, "lateral": lateral, "fraction": fraction}
        self.condyle_waist_status.setText(
            f"Condyle centre at {100 * fraction:.0f} % of the condylar axis (medial 0 % - lateral 100 %, "
            f"{fraction * np.linalg.norm(d):.1f} mm from the medial centre).")

    def _update_condyle_circles(self):
        panel = self._condyle_circle_panel()
        if panel is None:
            return
        for spin, name in zip(self.condyle_circle_spinboxes, self.CONDYLE_CIRCLE_LANDMARKS):
            blocked = spin.blockSignals(True)
            spin.setValue(self._condyle_radius(panel, name))
            spin.blockSignals(blocked)
        for other in (getattr(self, "left_panel", None), getattr(self, "right_panel", None)):
            if other is not None and other is not panel:
                self._hide_condyle_circles(other.side)
        if self.condyle_circle_box.isHidden():          # a level without condyles (e.g. radius)
            self._hide_condyle_circles(panel.side)
            return
        if hasattr(self, "_original_overlay_hidden") and self._original_overlay_hidden(panel.side):
            self._hide_condyle_circles(panel.side)       # corrected view: the circles belong to the original
            self.condyle_circle_status.setText("Corrected view - circles shown at the original landmarks only.")
            return
        positions = self._condyle_positions(panel)
        if positions is None or not self.condyle_circle_checkbox.checked:
            self._hide_condyle_circles(panel.side)
            self.condyle_circle_status.setText(
                "" if not self.condyle_circle_checkbox.checked else "Place both condyle centres - then the circles appear.")
            return
        axis = positions[1] - positions[0]
        distance = np.linalg.norm(axis)
        if distance < 1.0:
            self._hide_condyle_circles(panel.side)
            self.condyle_circle_status.setText("The two condyle centres are less than 1 mm apart.")
            return
        axis = axis / distance
        for index, (name, centre) in enumerate(zip(self.CONDYLE_CIRCLE_LANDMARKS, positions)):
            radius = self._condyle_radius(panel, name)
            self._show_condyle_circle(panel.side, index, name, centre, axis, radius)
            self._place_circle_handle(panel, index, name, centre, axis, radius)
        self.condyle_circle_status.setText(f"Condylar axis {distance:.1f} mm (centre to centre).")

    def _circle_polydata(self, centre, axis, radius):
        """Torus around the condylar axis: ring radius = circle radius, tube diameter = ring thickness.
        In a slice perpendicular to the axis it appears as a ring band (inner / outer edge)."""
        tube = 0.5 * float(self._ring_thickness())
        torus = vtk.vtkParametricTorus()
        torus.SetRingRadius(float(radius))
        torus.SetCrossSectionRadius(tube)
        source = vtk.vtkParametricFunctionSource()
        source.SetParametricFunction(torus)
        source.SetUResolution(120)
        source.SetVResolution(16)
        source.Update()
        # torus axis is z: rotate z onto the condylar axis, move to the centre
        z = np.array([0.0, 0.0, 1.0])
        rotation_axis = np.cross(z, axis)
        angle = np.degrees(np.arctan2(np.linalg.norm(rotation_axis), np.dot(z, axis)))
        transform = vtk.vtkTransform()
        transform.Translate(*[float(c) for c in centre])
        if np.linalg.norm(rotation_axis) > 1e-9:
            transform.RotateWXYZ(float(angle), *[float(c) for c in rotation_axis])
        elif np.dot(z, axis) < 0:
            transform.RotateWXYZ(180.0, 1.0, 0.0, 0.0)
        mover = vtk.vtkTransformPolyDataFilter()
        mover.SetTransform(transform)
        mover.SetInputConnection(source.GetOutputPort())
        mover.Update()
        return mover.GetOutput()

    def _ring_thickness(self):
        spin = getattr(self, "condyle_ring_thickness", None)
        return spin.value if spin is not None else self.CONDYLE_CIRCLE_WIDTH

    def _show_condyle_circle(self, side, index, name, centre, axis, radius):
        node_name = f"{self.CONDYLE_CIRCLE_NODE_PREFIX}{side} {name}"
        node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", node_name)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(*self._CIRCLE_COLORS[index])
            display.SetOpacity(0.9)
            display.SetSliceIntersectionThickness(self.CONDYLE_CIRCLE_LINE)
            display.SetVisibility2D(True)
            display.BackfaceCullingOff()
        node.SetAndObservePolyData(self._circle_polydata(centre, axis, radius))
        node.GetDisplayNode().SetSliceIntersectionThickness(self.CONDYLE_CIRCLE_LINE)
        node.GetDisplayNode().SetVisibility(True)

    def _hide_condyle_circles(self, side):
        prefix = f"{self.CONDYLE_CIRCLE_NODE_PREFIX}{side} "
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if (node.GetName() or "").startswith(prefix) and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(False)
        for name in self.CONDYLE_CIRCLE_LANDMARKS:
            handle = slicer.mrmlScene.GetFirstNodeByName(self._circle_handle_name(side, name))
            if handle is not None and handle.GetDisplayNode() is not None:
                handle.GetDisplayNode().SetVisibility(False)

    # ------------------------------------------------------------------ radius handle
    def _circle_handle_name(self, side, name):
        return f"Doggybone condyle radius {side} {name}"

    def _circle_handle(self, panel, index, name):
        node_name = self._circle_handle_name(panel.side, name)
        node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if node is None:
            # a new markups node may become the ACTIVE place list - keep the landmarks' list active
            selection = slicer.app.applicationLogic().GetSelectionNode()
            active = selection.GetActivePlaceNodeID()
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
            selection.SetReferenceActivePlaceNodeID(active or panel.markup_node_id)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(*self._CIRCLE_COLORS[index])
            display.SetSelectedColor(*self._CIRCLE_COLORS[index])
            display.SetGlyphScale(2.5)
            display.SetTextScale(2.0)
            display.SetSnapMode(slicer.vtkMRMLMarkupsDisplayNode.SnapModeUnconstrained)
            node.SetMarkupLabelFormat("")
            node.SetHideFromEditors(True)
        if not hasattr(self, "_circle_handle_tags"):
            self._circle_handle_tags = {}
        if node.GetID() not in self._circle_handle_tags:
            tags = [node.AddObserver(slicer.vtkMRMLMarkupsNode.PointStartInteractionEvent,
                                     lambda caller, ev: setattr(self, "_circle_dragging", True)),
                    node.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                                     lambda caller, ev, name=name: self._on_circle_handle_moved(caller, name)),
                    node.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                                     lambda caller, ev: self._on_circle_handle_released())]
            self._circle_handle_tags[node.GetID()] = (node, tags)
        return node

    def _place_circle_handle(self, panel, index, name, centre, axis, radius):
        node = self._circle_handle(panel, index, name)
        node.GetDisplayNode().SetVisibility(True)
        if getattr(self, "_circle_dragging", False) and getattr(self, "_circle_dragged_name", None) == name:
            return                                   # the user is moving this handle - leave it there
        direction = getattr(self, "_circle_handle_dirs", {}).get((panel.side, name))
        if direction is None:
            direction = np.array([0.0, 0.0, 1.0])
        direction = np.asarray(direction, float) - np.dot(direction, axis) * axis
        if np.linalg.norm(direction) < 1e-6:
            direction = np.cross(axis, [1.0, 0.0, 0.0])
        direction /= np.linalg.norm(direction)
        position = centre + radius * direction
        self._circle_handle_updating = True
        try:
            if node.GetNumberOfControlPoints() == 0:
                node.AddControlPointWorld(vtk.vtkVector3d(*[float(v) for v in position]))
                node.SetNthControlPointLabel(0, "r")
            else:
                node.SetNthControlPointPositionWorld(0, *[float(v) for v in position])
        finally:
            self._circle_handle_updating = False

    def _on_circle_handle_moved(self, node, name):
        if getattr(self, "_circle_handle_updating", False) or node.GetNumberOfControlPoints() == 0:
            return
        panel = self._condyle_circle_panel()
        positions = self._condyle_positions(panel) if panel is not None else None
        if positions is None:
            return
        axis = positions[1] - positions[0]
        if np.linalg.norm(axis) < 1.0:
            return
        axis /= np.linalg.norm(axis)
        centre = positions[list(self.CONDYLE_CIRCLE_LANDMARKS).index(name)]
        offset = np.array(node.GetNthControlPointPositionWorld(0), float) - centre
        offset -= np.dot(offset, axis) * axis               # radius in the circle's plane
        radius = float(np.linalg.norm(offset))
        if radius < 1.0:
            return
        self._circle_dragged_name = name
        if not hasattr(self, "_circle_handle_dirs"):
            self._circle_handle_dirs = {}
        self._circle_handle_dirs[(panel.side, name)] = offset / radius
        index = list(self.CONDYLE_CIRCLE_LANDMARKS).index(name)
        self.condyle_circle_spinboxes[index].setValue(round(radius, 2))   # stores it, redraws the circle

    def _on_circle_handle_released(self):
        self._circle_dragging = False
        self._circle_dragged_name = None
        self._schedule_condyle_circles()                     # handle back onto the circle

    def _remove_circle_handle_observers(self):
        for node, tags in getattr(self, "_circle_handle_tags", {}).values():
            for tag in tags:
                try:
                    node.RemoveObserver(tag)
                except Exception:
                    pass
        self._circle_handle_tags = {}

    # ------------------------------------------------------------------ slice view
    def _align_yellow_to_condyle(self, name):
        panel = self._condyle_circle_panel()
        positions = self._condyle_positions(panel) if panel is not None else None
        if positions is None:
            self.condyle_circle_status.setText("Place both condyle centres first.")
            return
        axis = positions[1] - positions[0]
        if np.linalg.norm(axis) < 1.0:
            return
        axis = axis / np.linalg.norm(axis)
        centre = positions[list(self.CONDYLE_CIRCLE_LANDMARKS).index(name)]
        # "up" in the view: the shaft (proximal) if placed, else the scanner's superior direction
        up = np.array([0.0, 0.0, 1.0])
        by_name = {landmark.name: landmark for landmark in panel.landmarks}
        if len(self.CONDYLE_CIRCLE_SHAFT) == 2:
            a, b = (by_name.get(n) for n in self.CONDYLE_CIRCLE_SHAFT)
            if a is not None and b is not None and a.placed and b.placed:
                up = np.array(a.get_position(), dtype=float) - np.array(b.get_position(), dtype=float)
        up = up - np.dot(up, axis) * axis
        if np.linalg.norm(up) < 1e-6:
            up = np.cross(axis, [1.0, 0.0, 0.0])
        up = up / np.linalg.norm(up)
        transverse = np.cross(up, axis)
        slice_node = slicer.app.layoutManager().sliceWidget("Yellow").mrmlSliceNode()
        slice_node.SetSliceToRASByNTP(*axis, *transverse, *centre, 0)
        self.condyle_circle_status.setText(
            f"Yellow view: perpendicular to the condylar axis through the {name}. Fit the circle to the contour.")
