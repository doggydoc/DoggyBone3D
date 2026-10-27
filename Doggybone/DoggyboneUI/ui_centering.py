# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""CenteringMixin: shaft-axis landmarks centred in the bone cross-section.

For the landmark pairs in CENTERED_AXIS_PAIRS (two points of one shaft axis), each point is moved
into the AREA centroid of the bone's cross-section, cut perpendicular to the axis through the point.
The axis depends on the points, so both are re-centred together a few times until they no longer
move. Moving a point along the bone keeps working: it is re-centred at its new height. The cut is
made on the bone surface of the segmentation; only the bone the point sits on is used (a neighbouring
bone, or the medullary canal of a hollow segmentation, do not shift the centre).

Switched on / off with the check box "Centre shaft points in the cross-section" (measurements).
Uses attributes created in DoggyboneWidget.setup() and methods of the other mixins."""

from DoggyboneUI.common import *
from DoggyboneLib import section_logic


class CenteringMixin:

    CENTERED_AXIS_PAIRS = ()          # ((landmark, landmark), ...) - set per limb
    CENTERING_ITERATIONS = 4
    SECTION_NODE_PREFIX = "Doggybone cross-section "

    # ------------------------------------------------------------------ UI
    def _build_centering(self, layout):
        """Check box + status line in the measurements section (only if this limb has such pairs)."""
        self._centering_busy = False
        self._centering_observers = []
        if not self.CENTERED_AXIS_PAIRS:
            self.centering_checkbox = None
            return
        self.centering_checkbox = qt.QCheckBox("Centre shaft points in the cross-section")
        self.centering_checkbox.setChecked(True)
        self.centering_checkbox.setToolTip(
            "On: the shaft-axis points jump into the centre (area centroid) of the bone cross-section, cut "
            "perpendicular to their axis. Move a point along the bone - it is re-centred at the new height. "
            "Needs the bone segmentation. Off: the points stay exactly where you put them (callus, implants, "
            "irregular contour).")
        self.centering_checkbox.toggled.connect(self._on_centering_toggled)
        self.centering_status = StatusLabel("")
        self.centering_status.setWordWrap(True)
        layout.addWidget(self.centering_checkbox)
        layout.addWidget(self.centering_status)
        self._install_centering_observers()

    def _install_centering_observers(self):
        self._remove_centering_observers()
        for panel in (getattr(self, "left_panel", None), getattr(self, "right_panel", None)):
            node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id) if panel is not None else None
            if node is None:
                continue
            for event in (slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                          slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent):
                tag = node.AddObserver(event, lambda caller, ev, panel=panel: self._schedule_centering(panel))
                self._centering_observers.append((node, tag))

    def _remove_centering_observers(self):
        for node, tag in getattr(self, "_centering_observers", []):
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._centering_observers = []

    def _on_centering_toggled(self, checked):
        if checked:
            for panel in (self.left_panel, self.right_panel):
                self._schedule_centering(panel)
        else:
            self._hide_sections()
            self.centering_status.setText("")

    def _schedule_centering(self, panel):
        import time
        if getattr(self, "_centering_busy", False) or getattr(self, "centering_checkbox", None) is None \
                or not self.centering_checkbox.checked:
            return
        if time.monotonic() - getattr(self, "_centering_done_at", 0.0) < 0.3:
            return                                   # events caused by our own position update
        qt.QTimer.singleShot(0, lambda: self._center_panel(panel))   # after the landmark's own callbacks

    # ------------------------------------------------------------------ geometry
    def _section_loops(self, polydata, origin, normal):
        """Closed contour loops (3D point arrays) of the bone surface cut by the plane."""
        plane = vtk.vtkPlane()
        plane.SetOrigin(*[float(c) for c in origin])
        plane.SetNormal(*[float(c) for c in normal])
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(polydata)
        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(cutter.GetOutputPort())
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        output = stripper.GetOutput()
        points = output.GetPoints()
        loops = []
        if points is None:
            return loops
        lines = output.GetLines()
        lines.InitTraversal()
        ids = vtk.vtkIdList()
        while lines.GetNextCell(ids):
            loop = np.array([points.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
            if len(loop) >= 4:
                loops.append(loop[:-1] if np.linalg.norm(loop[0] - loop[-1]) < 1e-6 else loop)
        return loops

    def _section_centre(self, polydata, point, axis):
        """(centroid 3D, area mm^2, loop 3D) of the cross-section through `point` perpendicular to `axis`."""
        loops = self._section_loops(polydata, point, axis)
        if not loops:
            return None, 0.0, None
        u, v = section_logic.plane_basis(axis)
        loops_2d = [section_logic.to_plane(loop, point, u, v) for loop in loops]
        index, centre_2d, area = section_logic.pick_loop_centroid(loops_2d, [0.0, 0.0])
        if index is None:
            return None, 0.0, None
        centre = np.asarray(point, dtype=float) + centre_2d[0] * u + centre_2d[1] * v
        return centre, area, loops[index]

    # ------------------------------------------------------------------ centring
    def _center_panel(self, panel):
        if panel is None or getattr(self, "_centering_busy", False):
            return
        polydata = self._get_bone_polydata()
        if polydata is None:
            self.centering_status.setText("Centring: no bone segmentation yet - the points stay where they are.")
            return
        node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id)
        by_name = {landmark.name: landmark for landmark in panel.landmarks}
        messages = []
        self._centering_busy = True
        try:
            for name_a, name_b in self.CENTERED_AXIS_PAIRS:
                a, b = by_name.get(name_a), by_name.get(name_b)
                if a is None or b is None or not a.placed or not b.placed:
                    continue
                pa, pb = np.array(a.get_position()), np.array(b.get_position())
                if np.linalg.norm(pb - pa) < 5.0:
                    messages.append(f"{name_a} / {name_b}: points less than 5 mm apart - not centred.")
                    continue
                ok = True
                loops = {}
                for _ in range(self.CENTERING_ITERATIONS):
                    axis = (pb - pa) / np.linalg.norm(pb - pa)
                    ca, area_a, loop_a = self._section_centre(polydata, pa, axis)
                    cb, area_b, loop_b = self._section_centre(polydata, pb, axis)
                    if ca is None or cb is None:
                        ok = False
                        break
                    moved = max(np.linalg.norm(ca - pa), np.linalg.norm(cb - pb))
                    pa, pb = ca, cb
                    loops = {name_a: (loop_a, area_a), name_b: (loop_b, area_b)}
                    if moved < 0.05:
                        break
                if not ok:
                    messages.append(f"{name_a} / {name_b}: no bone contour at a point - place it on the bone.")
                    continue
                for landmark, position in ((a, pa), (b, pb)):
                    node.SetNthControlPointPosition(landmark._id, *[float(c) for c in position])
                for name, (loop, area) in loops.items():
                    self._show_section(panel.side, name, loop)
                messages.append(f"✓ {name_a.split(' - ')[0].split(' axis')[0]} centred "
                                f"(cross-sections {loops[name_a][1]:.0f} / {loops[name_b][1]:.0f} mm²).")
        except Exception as error:
            logging.exception("Doggybone: centring failed (continuing)")
            messages.append(f"Centring failed: {error}")
        finally:
            import time
            self._centering_busy = False
            self._centering_done_at = time.monotonic()
        if messages:
            self.centering_status.setText("\n".join(messages))
        for landmark in panel.landmarks:                      # measurements / values follow the new positions
            if landmark.placed:
                for callback in landmark.change_callbacks:
                    callback()

    # ------------------------------------------------------------------ display
    def _show_section(self, side, landmark_name, loop):
        name = f"{self.SECTION_NODE_PREFIX}{side} {landmark_name}"
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(1.0, 0.85, 0.1)
            display.SetLineWidth(3)
            display.SetVisibility2D(True)
        points = vtk.vtkPoints()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(loop) + 1)
        for i, p in enumerate(loop):
            points.InsertNextPoint(*[float(c) for c in p])
            line.GetPointIds().SetId(i, i)
        line.GetPointIds().SetId(len(loop), 0)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        node.SetAndObservePolyData(polydata)
        hidden = hasattr(self, "_original_overlay_hidden") and self._original_overlay_hidden(side)
        node.GetDisplayNode().SetVisibility(not hidden)       # corrected view: original aids stay hidden

    def _hide_sections(self):
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if (node.GetName() or "").startswith(self.SECTION_NODE_PREFIX) and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(False)
