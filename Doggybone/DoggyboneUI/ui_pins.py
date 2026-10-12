# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""PinsMixin: Pins: reduction pins, pin rows, pin gizmo, planned pins, pin feedback.

Part of DoggyboneWidget (see Doggybone.py). Methods were moved here unchanged; they rely on
attributes created in DoggyboneWidget.setup() and on methods of the other mixins.
"""

from DoggyboneUI.common import *


class PinsMixin:

    def _plan_reduction_pins(self, mesh, gap):
        """Sleeves of the reduction guide: the saw guide's pins in the corrected position."""
        reduced, moved, error = self._reduction_guide_pins()
        if error:
            return [], error
        if not reduced:
            return [], "No pins yet - switch Pins on and click on the corrected bone."
        obb = self._pin_obb_tree(mesh)
        planned, rejected = [], []
        for pin in reduced:
            q = guide_logic.plan_pin(obb, pin["entry"] + 2.0 * pin["axis"], pin["axis"], pin["diameter"],
                                     gap, guide_logic.BLOCK_HEIGHT)
            if "error" in q:
                rejected.append(f"{pin['label']}: {q['error']} (outside the prepared surface)")
                continue
            q["label"], q["segment"] = pin["label"], pin["segment"]
            planned.append(q)
        axes = [p["axis"] for p in planned]
        spread = max((float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))) for a in axes for b in axes),
                     default=0.0)
        per_segment = {}
        for p in planned:
            per_segment[p["segment"]] = per_segment.get(p["segment"], 0) + 1
        note = (f"Sleeves at the saw-guide pins (corrected position): "
                + ", ".join(f"{v} {k}" for k, v in per_segment.items())
                + f", \u00d8 {planned[0]['diameter']:.1f} mm, axes parallel within {spread:.1f}\u00b0" if planned
                else "No sleeves.")
        if any(v < 2 for v in per_segment.values()) or len(per_segment) < 2:
            note += "\nWARNING: each fragment needs 2 pins, otherwise it can rotate about its pin."
        if spread > 1.0:
            note += "\nWARNING: pins not parallel after reduction - re-extrude the saw guide."
        if rejected:
            note += "\nNot built: " + "; ".join(rejected)
        return planned, note

    def _reduction_is_bar(self):
        box = getattr(self, "reduction_type", None)
        return box is not None and box.currentIndex == 0

    def _build_reduction_bar(self):
        """Reduction guide as a bar over the saw guide's pins in the corrected position."""
        from vtk.util.numpy_support import vtk_to_numpy
        reduced, moved, error = self._reduction_guide_pins()
        if error:
            self.guide_status.setText(error)
            return
        if len({p["segment"] for p in reduced}) < 2:
            self.guide_status.setText("The reduction guide needs pins in BOTH fragments - switch Pins on and "
                                      "click on the corrected bone.")
            return
        reference = guide_logic.unit(reduced[0]["axis"])
        axis = guide_logic.unit(np.sum([guide_logic.unit(p["axis"]) * (1.0 if float(np.dot(p["axis"], reference)) >= 0
                                                                           else -1.0) for p in reduced], axis=0))
        wall = float(getattr(guide_logic, "SLEEVE_WALL", 2.0))
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            side = self.osteotomy_side.currentText.lower()
            bone = self.osteotomy_bone.currentText
            combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
            bone_obb = None
            if combined is not None and combined.GetPolyData() is not None and combined.GetPolyData().GetNumberOfPoints():
                bone_obb = vtk.vtkOBBTree()
                bone_obb.SetDataSet(combined.GetPolyData())
                bone_obb.BuildLocator()
            if bone_obb is not None:
                # bosses AND connecting webs over the full height, seated on the corrected bone
                body, info = guide_logic.reduction_web(
                    reduced, axis, bone_obb, height=float(self.reduction_standoff.value),
                    web_width=float(self.reduction_bar_diameter.value), wall=wall,
                    gap=float(self.guide_gap.value),
                    ring=bool(getattr(self, "reduction_ring_checkbox", None) and self.reduction_ring_checkbox.isChecked()),
                    filled=bool(getattr(self, "reduction_filled_checkbox", None) and self.reduction_filled_checkbox.isChecked()),
                    contact_only=bool(getattr(self, "reduction_contact_only_checkbox", None)
                                      and self.reduction_contact_only_checkbox.isChecked()),
                    clearance=float(self.reduction_clearance.value) if getattr(self, "reduction_clearance", None) else 6.0)
            else:
                body, info = guide_logic.reduction_bar(
                    reduced, axis, standoff=float(self.reduction_standoff.value),
                    bar_diameter=float(self.reduction_bar_diameter.value),
                    boss_height=float(self.reduction_boss_height.value), wall=wall)
            label_note = ""
            if self._label_wanted():
                if self._label_mode() == "Break-off tag":
                    label, label_note = self._label_tag(body, axis, bone_obb)
                else:
                    label, label_note = self._reduction_guide_label(body, axis)
                    if label is None:          # no flat spot on the guide: the break-off tag instead
                        label, tag_note = self._label_tag(body, axis, bone_obb)
                        label_note = ("No flat area for the label on the guide - break-off tag used instead. "
                                      + tag_note) if label is not None else label_note
                if label is not None:
                    append = vtk.vtkAppendPolyData()
                    append.AddInputData(body)
                    append.AddInputData(label)
                    append.Update()
                    body = vtk.vtkPolyData()
                    body.DeepCopy(append.GetOutput())
            self._set_guide_body(body)
            shown = []
            for pin in reduced:
                q = dict(pin)
                h = float(np.dot(pin["entry"], axis))
                q["top"] = pin["entry"] + (info["top"] - h + 3.0) / max(float(np.dot(pin["axis"], axis)), 1e-6) * pin["axis"]
                shown.append(q)
            self._set_pin_axes(guide_logic.pin_axis_polydata(shown))
            footprint = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_FOOTPRINT_NAME)
            if footprint is not None and footprint.GetDisplayNode() is not None:
                footprint.GetDisplayNode().SetVisibility(False)    # the bar does not use the footprint
            clearance = None
            if "guide_height_range" in info:
                combined = None                         # seated web: nothing hovers over the bone
            side = self.osteotomy_side.currentText.lower()
            bone = self.osteotomy_bone.currentText
            if "guide_height_range" not in info:
                combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
            if combined is not None and combined.GetPolyData() is not None and combined.GetPolyData().GetNumberOfPoints():
                local = vtk_to_numpy(combined.GetPolyData().GetPoints().GetData()).astype(float) @ info["frame"].T
                near = np.zeros(len(local), dtype=bool)
                nodes = info["nodes"]                 # rods only - the bosses sit on the bone on purpose
                for i, j in info["edges"]:
                    A, B = nodes[i], nodes[j]
                    AB = B - A
                    t = np.clip(((local[:, 0] - A[0]) * AB[0] + (local[:, 1] - A[1]) * AB[1]) / max(float(AB @ AB), 1e-12), 0, 1)
                    near |= np.hypot(local[:, 0] - (A[0] + t * AB[0]), local[:, 1] - (A[1] + t * AB[1])) <= info["r_bar"]
                for n in nodes:                        # not under a boss
                    near &= np.hypot(local[:, 0] - n[0], local[:, 1] - n[1]) > info["r_boss"]
                if near.any():
                    clearance = float(info["zc"] - info["r_bar"] - local[near, 2].max())
        finally:
            qt.QApplication.restoreOverrideCursor()
        per_segment = {}
        for p in reduced:
            per_segment[p["segment"]] = per_segment.get(p["segment"], 0) + 1
        lines = [f"Reduction bar over {len(reduced)} pins (" + ", ".join(f"{v} {k}" for k, v in per_segment.items())
                 + f"), bores \u00d8 {2 * info['r_in']:.2f} mm, pin axes parallel within "
                 f"{info['max_axis_deviation_deg']:.1f}\N{DEGREE SIGN}; "
                 + (f"bosses and webs connected over the full height, seated on the corrected bone (fit gap "
                    f"{self.guide_gap.value:.2f} mm), {info['guide_height_range'][0]:.0f}-"
                    f"{info['guide_height_range'][1]:.0f} mm high." if "guide_height_range" in info
                    else "no bone contact."),
                 f"Pins must stand out at least {max(info['protrusion']):.0f} mm above the bone."]
        if clearance is not None:
            lines.append(("WARNING: the bar reaches into the corrected bone - increase Height. " if clearance < 1.0
                          else "") + f"Clearance bar - bone {clearance:.1f} mm.")
        if any(v < 2 for v in per_segment.values()):
            lines.append("WARNING: each fragment needs 2 pins, otherwise it can rotate about its pin.")
        if info["max_axis_deviation_deg"] > 1.0:
            lines.append("WARNING: pins not parallel after reduction - re-extrude the saw guide first.")
        if getattr(self, "reduction_ring_checkbox", None) is not None and self.reduction_ring_checkbox.isChecked() \
                and "guide_height_range" in info:
            if info.get("contact_only"):
                lines.append(f"Bone contact only through the cylinders around the pins; webs / frame "
                             f"{info['web_clearance']:.1f} mm above the bone (room for the plate).")
            lines.append(("Filled frame over all pins." if info.get("filled") else
                          "Ring: closed frame around all pins (two bridges across the osteotomy).") if info.get("ring")
                         else "Ring not possible - the pins lie (nearly) in one line; built as a single chain. "
                              "Offset some pins sideways for a ring.")
        if label_note:
            lines.append(label_note)
        self.guide_status.setText("\n".join(lines))

    PLANNED_PINS_ATTRIBUTE = "Doggybone.PlannedPins"

    def _store_planned_pins(self, pins):
        """Keep the planned pins in the scene (pins node attribute) - survives a module Reload."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.SAW_PINS_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.SAW_PINS_NAME)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
        keys = ("entry", "far", "top", "axis", "point")
        data = [dict({k: [float(x) for x in pin[k]] for k in keys},
                     r_in=float(pin["r_in"]), r_out=float(pin["r_out"]), diameter=float(pin["diameter"]),
                     label=str(pin.get("label", "")), segment=str(pin.get("segment", "")))
                for pin in pins]
        node.SetAttribute(self.PLANNED_PINS_ATTRIBUTE, json.dumps(data))

    def _load_planned_pins(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.SAW_PINS_NAME)
        text = node.GetAttribute(self.PLANNED_PINS_ATTRIBUTE) if node is not None else None
        if not text:
            return []
        pins = json.loads(text)
        for pin in pins:
            for k in ("entry", "far", "top", "axis", "point"):
                pin[k] = np.asarray(pin[k], dtype=float)
        return pins

    def _reduced_pins(self, pins):
        """Pins in the CORRECTED position: pins of the fragment that carries the correction
        transform are moved with it. Returns (pins, moved_segment, error)."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        transform = slicer.mrmlScene.GetFirstNodeByName(names[2])
        if transform is None:
            return [], None, "No correction transform - run the correction preview first."
        moved = None
        for segment, name in (("proximal", names[0]), ("distal", names[1])):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetParentTransformNode() is not None \
                    and node.GetParentTransformNode().GetID() == transform.GetID():
                moved = segment
        if moved is None:
            return [], None, "Cannot tell which fragment carries the correction transform."
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)
        M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        R, t = M[:3, :3], M[:3, 3]
        out = []
        for pin in pins:
            q = dict(pin)
            if pin["segment"] == moved:
                for k in ("entry", "far", "top", "point"):
                    q[k] = R @ pin[k] + t
                q["axis"] = guide_logic.unit(R @ pin["axis"])
            out.append(q)
        return out, moved, None

    SAW_PINS_NAME = property(lambda self: "Doggybone guide pins" + self._bone_tag())
    # reduction mode: pins set on the corrected bone ("... (reduction)"), saw mode: legacy marks
    GUIDE_PINS_NAME = property(lambda self: self._mode_name("Doggybone guide pins"))
    GUIDE_PIN_AXES_NAME = property(lambda self: self._mode_name("Doggybone guide pin axes"))

    def _pin_obb_tree(self, mesh):
        """Ray locator of a surface, cached per surface (the footprint ROI mesh and the whole bone are
        both in use - two entries, no rebuilding when they alternate)."""
        key = (id(mesh), mesh.GetMTime())
        cache = getattr(self, "_obb_cache", None)
        if cache is None:
            cache = self._obb_cache = {}
        if key not in cache:
            obb = vtk.vtkOBBTree()
            obb.SetDataSet(mesh)
            obb.BuildLocator()
            if len(cache) >= 3:
                cache.pop(next(iter(cache)))
            cache[key] = obb
        return cache[key]

    def _pin_diameter(self):
        chosen = self.guide_pin.currentText
        if chosen.startswith("Auto"):
            return guide_logic.auto_pin_diameter(self._bone_diameter_at_cuts())
        return float(chosen.split()[0])

    def _bone_diameter_at_cuts(self):
        """Smallest bone diameter at the cutting planes of the CURRENT bone, measured on demand (cached
        per bone and plane pose) - the same value for the reduction guide (pins set first) and the
        saw guide, independent of which guide was built before."""
        bone_polydata = self._get_bone_polydata() if hasattr(self, "_get_bone_polydata") else None
        planes = self._cut_planes()
        if bone_polydata is None or not planes:
            return None
        key = (self.osteotomy_bone.currentText, bone_polydata.GetNumberOfPoints(),
               tuple(round(float(x), 2) for _l, o, n in planes for x in list(o) + list(n)))
        cache = getattr(self, "_bone_diameter_cache", {})
        if key in cache:
            return cache[key]
        diameters = []
        for _label, origin, normal in planes:
            try:
                contour = guide_logic.section_contour(bone_polydata, origin, normal)
                d = guide_logic.contour_diameter(contour, normal) if contour is not None else None
            except Exception:
                logging.exception("Doggybone: bone diameter at the cut failed (continuing)")
                d = None
            if d is not None:
                diameters.append(float(d))
        value = min(diameters) if diameters else None
        cache[key] = value
        self._bone_diameter_cache = cache
        return value

    def _cut_planes(self):
        """[(label, origin, unit normal)] of the existing cutting planes."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        planes = []
        for label, name in (("proximal", self._plane_node_name(side, bone)),
                            ("distal", self._distal_plane_node_name(side, bone))):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                origin, normal = self._get_plane_origin_normal(node)
                planes.append((label, np.asarray(origin, float), guide_logic.unit(np.asarray(normal, float))))
        return planes

    def _pin_segment(self, point):
        """('proximal' | 'distal', None) or (None, reason) for a pin through point."""
        planes = self._cut_planes()
        if not planes:
            return None, "no cutting plane"
        p = np.asarray(point, float)
        # the BORE must stay outside the saw block (sleeve may merge into the block)
        margin = guide_logic.SLOT_WIDTH + guide_logic.SLOT_WALL + guide_logic.sleeve_radii(self._pin_diameter())[0] + 0.3
        if len(planes) == 2:
            for index, (label, o, n) in enumerate(planes):
                o2 = planes[1 - index][1]
                towards_wedge = np.sign(float((o2 - o) @ n)) or 1.0
                s_val = float((p - o) @ n) * towards_wedge
                if s_val < 0:                                    # kept side of this cut
                    if -s_val < margin:
                        return None, f"too close to the {label} cut ({-s_val:.1f} mm, needs {margin:.1f} mm)"
                    return label, None
            return None, "inside the wedge"
        label, o, n = planes[0]
        shaft, _, _, error = self._get_anatomical_frame(self.osteotomy_side.currentText.lower(),
                                                        self.osteotomy_bone.currentText)
        s_val = float((p - o) @ n)
        if abs(s_val) < margin:
            return None, f"too close to the cut ({abs(s_val):.1f} mm, needs {margin:.1f} mm)"
        proximal = float((p - o) @ np.asarray(shaft, float)) < 0 if not error else s_val < 0
        return ("proximal" if proximal else "distal"), None

    def _pin_feedback(self, message, point, ok):
        """Rejected click: red status text and a short red cross at the spot."""
        self.guide_status.setStyleSheet("" if ok else STATUS_ERROR_STYLE)
        if ok:
            return
        self.guide_status.setText(message)
        slicer.util.showStatusMessage(message, 4000)
        name = "Doggybone pin rejected"
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
            node.GetDisplayNode().SetSelectedColor(1.0, 0.1, 0.1)
            node.GetDisplayNode().SetColor(1.0, 0.1, 0.1)
            node.GetDisplayNode().SetGlyphTypeFromString("Cross2D")
            node.GetDisplayNode().SetGlyphScale(3.0)
            node.GetDisplayNode().SetPointLabelsVisibility(False)
        node.RemoveAllControlPoints()
        node.AddControlPointWorld(vtk.vtkVector3d(*[float(x) for x in point]))

        def clear(node_id=node.GetID()):
            existing = slicer.mrmlScene.GetNodeByID(node_id)
            if existing is not None:
                existing.RemoveAllControlPoints()
        qt.QTimer.singleShot(1500, clear)

    def onDeletePins(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        count = node.GetNumberOfControlPoints() if node is not None else 0
        if node is not None:
            node.RemoveAllControlPoints()
        self._pins_dirty = False
        self._set_pin_axes(None)
        if self._guide_mode_is_reduction():
            # the reduction guide is built on these pins - without pins there is no guide
            self._set_guide_body(None)
            handles = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
            if handles is not None:
                handles.RemoveAllControlPoints()
            self._pin_rows_active = False
            self._select_reduction_pin(None)
        elif count and slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME) is not None:
            self._extrude_guide_body()
        self.guide_status.setText(f"{count} pin(s) deleted." if count else "No pins to delete.")

    def _pin_under_mouse(self, x, y, tolerance=2.5):
        """Index of the pin mark closest to the view ray through (x, y), if within tolerance mm.
        Works although the mark sits inside the (hollow) sleeve bore after extrusion."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        if node is None or node.GetNumberOfControlPoints() == 0:
            return None
        renderer = self._guide_renderer()
        ends = []
        for z in (0.0, 1.0):
            renderer.SetDisplayPoint(float(x), float(y), z)
            renderer.DisplayToWorld()
            wp = renderer.GetWorldPoint()
            ends.append(np.array([wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]]))
        d = guide_logic.unit(ends[1] - ends[0])

        def ray_distance(point):
            q = np.asarray(point, float) - ends[0]
            return float(np.linalg.norm(q - (q @ d) * d))
        best, best_dist = None, tolerance
        for i in range(node.GetNumberOfControlPoints()):
            dist = ray_distance(node.GetNthControlPointPositionWorld(i))
            if dist < best_dist:
                best, best_dist = i, dist
        if best is not None or not self._guide_mode_is_reduction():
            return best
        # reduction: the pins are shown as long shafts - a click anywhere on a shaft hits that pin
        labels = {node.GetNthControlPointLabel(i): i for i in range(node.GetNumberOfControlPoints())}
        best_gap = None
        for label, start, end, radius in getattr(self, "_reduction_pin_shafts", []):
            index = labels.get(label)
            if index is None:
                continue
            dist = min(ray_distance(start + t * (end - start)) for t in np.linspace(0.0, 1.0, 41))
            gap = dist - (radius + 1.0)
            if gap <= 0.0 and (best_gap is None or gap < best_gap):
                best, best_gap = index, gap
        return best

    def _remove_pin(self, index):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        if node is None or index >= node.GetNumberOfControlPoints():
            return
        label = node.GetNthControlPointLabel(index)
        node.RemoveNthControlPoint(index)
        self._relabel_pins(node)
        self._pins_dirty = True
        self.guide_status.setStyleSheet("")
        self.guide_status.setText(f"Pin {label} removed ({node.GetNumberOfControlPoints()} marked). "
                                  + ("Press 'Place pins' again to finish and rebuild the guide."
                                     if getattr(self, "_guide_active_tool", None) == "pins"
                                     else "The guide is rebuilt."))
        if self._guide_mode_is_reduction():
            self._show_reduction_pin_axes()
            self._place_pin_gizmo()

    # --- automatic pin rows (reduction mode) -------------------------------------------------
    PIN_ROW_HANDLES = property(lambda self: "Doggybone pin row handles" + self._bone_tag())

    def _pin_row_defaults(self):
        """(first-pin distance from the cut, spacing) from the pin diameter."""
        diameter = self._pin_diameter()
        r_in, r_out = guide_logic.sleeve_radii(diameter)
        margin = guide_logic.SLOT_WIDTH + guide_logic.SLOT_WALL + r_in + 0.3      # as _pin_segment checks
        first = np.ceil((margin + 1.0) * 2.0) / 2.0
        spacing = np.ceil(max(2.0 * r_out + 2.0, 3.0 * diameter) * 2.0) / 2.0
        return float(first), float(spacing)

    def _fragment_cut_frames(self, M, moved, fragments):
        """{segment: (cut origin, unit direction along the shaft INTO the fragment)} in the corrected pose.
        CWO: each fragment's own cut plane; OWO: the single (master) cut."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        planes = {"proximal": slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone)),
                  "distal": slicer.mrmlScene.GetFirstNodeByName(self._distal_plane_node_name(side, bone))}
        master = self.reference_plane_selector.currentText.lower()
        is_cwo = self.osteotomy_technique.currentText.startswith("CWO")
        _direction, shafts, _error = self._reduction_pin_frame(M, moved)
        frames = {}
        for segment, (polydata, _obb, _loc) in fragments.items():
            node = planes[segment] if is_cwo else planes.get(master)
            if node is None or shafts is None:
                continue
            origin, _normal = self._get_plane_origin_normal(node)
            origin = np.asarray(origin, float)
            if segment == moved:
                origin = (M @ np.append(origin, 1.0))[:3]
            from vtk.util.numpy_support import vtk_to_numpy
            centroid = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float).mean(axis=0)
            s_dir = shafts[segment]
            if float(np.dot(centroid - origin, s_dir)) < 0.0:
                s_dir = -s_dir
            frames[segment] = (origin, s_dir)
        return frames

    @staticmethod
    def _pin_section_normal(shaft, direction):
        """Normal of the cross-section used to centre a pin: the shaft direction with its component
        along the pin removed - the section then contains the whole pin line."""
        shaft = np.asarray(shaft, float)
        direction = guide_logic.unit(np.asarray(direction, float))
        normal = shaft - float(np.dot(shaft, direction)) * direction
        return guide_logic.unit(normal) if np.linalg.norm(normal) > 1e-6 else guide_logic.unit(shaft)

    def _row_pin(self, segment, level_point, s_dir, fragments, direction):
        polydata, obb, _loc = fragments[segment]
        centre = self._section_centroid(polydata, level_point, self._pin_section_normal(s_dir, direction),
                                        largest=True)
        if centre is None:
            return {"error": "no bone at that level"}
        pin = guide_logic.plan_pin(obb, centre, direction, self._pin_diameter(), float(self.guide_gap.value),
                                   guide_logic.BLOCK_HEIGHT)
        if "error" not in pin:
            pin["segment"], pin["centre"] = segment, centre
        return pin

    def _pin_distribution_manual(self):
        box = getattr(self, "pin_row_pattern", None)
        return box is not None and box.currentText == "Manual"

    def _pin_distribution_changed(self):
        if self._pin_distribution_manual():
            self._pin_rows_active = False          # nothing is generated automatically any more
            handles = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
            if handles is not None:                # 'P row' / 'D row' only move automatic rows
                handles.RemoveAllControlPoints()
            return
        self._pin_rows_changed()

    def onPlacePinRows(self):
        if not self._guide_mode_is_reduction():
            self.guide_status.setText("Pin rows are placed in the 'Reduction guide' mode.")
            return
        if self._pin_distribution_manual():
            # manual placement: 'Place pins' starts it, a second 'Place pins' finishes it (the hidden
            # pin tool is switched off -> the guide is rebuilt with the new pins)
            button = getattr(self, "_guide_tool_buttons", {}).get("pins")
            if button is None:
                return
            if button.checked:
                button.setChecked(False)
                self.place_pin_rows_button.setText("Place pins")
                self.guide_status.setStyleSheet("")
                self.guide_status.setText("Manual pin placement finished. Click a pin to select / move it.")
                return
            button.setChecked(True)
            if button.checked:
                self.place_pin_rows_button.setText("Finish pins")
            self.guide_status.setStyleSheet("")
            self.guide_status.setText("Manual pins: click on the corrected bone to set a pin (it gets the gizmo), "
                                      "click a pin to delete it, Cmd/Ctrl + click a pin to move it. Press "
                                      "'Finish pins' when done.")
            return
        self._stop_guide_edit()
        self._pin_rows_active = True
        self._place_pin_rows()

    def _pin_rows_changed(self):
        if self._pin_distribution_manual():
            return
        if getattr(self, "_pin_rows_active", False) and self._guide_mode_is_reduction():
            self._place_pin_rows()

    def _place_pin_rows(self):
        M, moved = self._correction_state()
        if M is None:
            self.guide_status.setText("No correction preview - show the correction preview first.")
            return
        direction, _shafts, error = self._reduction_pin_frame(M, moved)
        if error:
            self.guide_status.setText(error)
            return
        fragments = self._corrected_fragments()
        frames = self._fragment_cut_frames(M, moved, fragments)
        auto_first, auto_spacing = self._pin_row_defaults()
        spacing = float(self.pin_row_spacing.value) or auto_spacing
        count = int(self.pin_row_count.value)
        distances = {"proximal": float(self.pin_row_distance_proximal.value) or auto_first,
                     "distal": float(self.pin_row_distance_distal.value) or auto_first}
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.REDUCTION_PIN_MARKS)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
            node.GetDisplayNode().SetSelectedColor(0.95, 0.95, 0.3)
            node.GetDisplayNode().SetGlyphScale(1.5)
            node.GetDisplayNode().SetTextScale(2.5)
        kept = []                            # pins added or moved by hand survive a new row layout
        for i in range(node.GetNumberOfControlPoints()):
            if self._pin_kind(node, i)[1] not in ("auto", "autofree"):
                kept.append((list(node.GetNthControlPointPositionWorld(i)),
                             node.GetNthControlPointDescription(i), node.GetNthControlPointID(i)))
        selected_id = getattr(self, "_selected_pin_id", None)
        node.RemoveAllControlPoints()
        handles_at, problems, placed, shifted, replaced, candidates = {}, [], {}, {}, {}, []
        for segment in ("proximal", "distal"):
            if segment not in frames:
                problems.append(f"{segment}: no cut plane / fragment")
                continue
            origin, s_dir = frames[segment]
            shift = 0.0                      # extra distance so the saw guide accepts the pin (oblique cut)
            Minv = np.linalg.inv(M)
            for k in range(count):
                pin = None
                for _attempt in range(6):
                    level = origin + (distances[segment] + shift + k * spacing) * s_dir
                    pin = self._row_pin(segment, level, s_dir, fragments, direction)
                    if "error" in pin:
                        break
                    raw_entry = pin["entry"] if segment != moved else (Minv @ np.append(pin["entry"], 1.0))[:3]
                    ok_segment, reason = self._pin_segment(raw_entry)
                    found = re.search(r"\(([\d.]+) mm, needs ([\d.]+) mm\)", reason or "")
                    if ok_segment is not None or not found:
                        break
                    shift += float(found.group(2)) - float(found.group(1)) + 0.3
                if "error" in pin:
                    problems.append(f"{segment} pin {k + 1}: {pin['error']} - the row does not fit, "
                                    "reduce Pins / Spacing / distance")
                    break
                if k == 0:
                    handles_at[segment] = pin["entry"] + 4.0 * np.asarray(direction, float)
                    shifted[segment] = shift
                kind = segment
                if self._pin_row_staggered():
                    sign = (1.0 if k % 2 == 0 else -1.0) * (1.0 if segment == "proximal" else -1.0)
                    offset_pin = self._staggered_pin(pin, sign, s_dir, direction, fragments, segment, moved, Minv)
                    if offset_pin is not None:
                        pin, kind = offset_pin, f"{segment}|autofree"
                candidates.append((kind, np.asarray(pin["entry"], float)))
        # a pin added / moved by hand takes the place of the ONE automatic pin nearest to it
        a_dir = np.asarray(direction, float)

        def lateral(p, q):
            offset = np.asarray(p, float) - np.asarray(q, float)
            return float(np.linalg.norm(offset - float(np.dot(offset, a_dir)) * a_dir))
        claimed = set()
        for position, description, _old_id in kept:
            options = [(lateral(position, entry), c) for c, (kind, entry) in enumerate(candidates)
                       if kind.split("|")[0] == description.split("|")[0] and c not in claimed]
            options = [o for o in options if o[0] < 0.9 * spacing]
            if options:
                claimed.add(min(options)[1])
        for c, (kind, entry) in enumerate(candidates):
            segment = kind.split("|")[0]
            if c in claimed:
                replaced[segment] = replaced.get(segment, 0) + 1
                continue
            index = node.AddControlPointWorld(vtk.vtkVector3d(*entry))
            node.SetNthControlPointDescription(index, kind)
            placed[segment] = placed.get(segment, 0) + 1
        reselect = None
        for position, description, old_id in kept:
            index = node.AddControlPointWorld(vtk.vtkVector3d(*position))
            node.SetNthControlPointDescription(index, description)
            if old_id == selected_id:
                reselect = index
        self._relabel_pins(node)
        close = []                           # sleeves that would touch: pins of one fragment too close
        for i in range(node.GetNumberOfControlPoints()):
            for j in range(i + 1, node.GetNumberOfControlPoints()):
                if self._pin_kind(node, i)[0] != self._pin_kind(node, j)[0]:
                    continue
                gap_ij = lateral(node.GetNthControlPointPositionWorld(i), node.GetNthControlPointPositionWorld(j))
                if gap_ij < 0.9 * spacing:
                    close.append(f"{node.GetNthControlPointLabel(i)}-{node.GetNthControlPointLabel(j)} "
                                 f"{gap_ij:.1f} mm")
        self._update_pin_row_handles(handles_at, frames)
        self._show_reduction_pin_axes()
        self._select_reduction_pin(reselect)
        self._pins_dirty = True
        text = (f"Pin rows: " + ", ".join(f"{v} {k}" for k, v in placed.items()) +
                (f" (+{len(kept)} added by hand" + (", replacing " + ", ".join(f"{v} {k}" for k, v in replaced.items())
                                                     if replaced else "") + ")" if kept else "") +
                f"; first pin {distances['proximal']:.1f} mm (proximal) / {distances['distal']:.1f} mm (distal) "
                f"from the cut, spacing {spacing:.1f} mm - "
                + ("staggered to both sides of the bone centre" if self._pin_row_staggered() else "centred")
                + ", parallel to each other and to the cut plane(s). Drag the row handles along the bone to "
                "adjust, then 'Extrude'.")
        if getattr(self, "_pin_tilt_note", ""):
            text += "\n" + self._pin_tilt_note
        moved_on = [f"{k} +{v:.1f} mm" for k, v in shifted.items() if v > 0.0]
        if moved_on:
            text += ("\nMoved further from the (oblique) cut so the saw-guide sleeves keep their distance: "
                     + ", ".join(moved_on) + ".")
        if problems:
            text += "\n" + "; ".join(problems)
        if close:
            text += (f"\nPins too close (< {0.9 * spacing:.1f} mm, sleeves touch): " + ", ".join(close)
                     + " - move one with the gizmo.")
        self.guide_status.setStyleSheet(STATUS_WARN_STYLE if close else "")
        self.guide_status.setText(text)

    def _pin_row_staggered(self):
        box = getattr(self, "pin_row_pattern", None)
        return box is not None and box.currentText == "Staggered"

    def _staggered_pin(self, pin, sign, s_dir, direction, fragments, segment, moved, Minv, max_offset=4.0):
        """The centred row pin shifted sideways (across the bone, perpendicular to shaft and pin) by as much
        as the bone allows: the pin must keep >= 80 % of its centred bone length and >= 1.5 mm bone around
        it. None if no offset of at least 1 mm fits (the pin then stays centred)."""
        z = guide_logic.unit(np.asarray(direction, float))
        across = np.cross(z, np.asarray(s_dir, float))
        if np.linalg.norm(across) < 1e-6:
            return None
        across = guide_logic.unit(across)
        _poly, obb, _loc = fragments[segment]
        centre = np.asarray(pin.get("centre", pin["entry"]), float)
        r_pin = 0.5 * float(pin.get("diameter", self._pin_diameter()))
        for offset in np.arange(max_offset, 0.99, -0.5):
            shifted = guide_logic.plan_pin(obb, centre + sign * offset * across, z, self._pin_diameter(),
                                           float(self.guide_gap.value), guide_logic.BLOCK_HEIGHT)
            if "error" in shifted or shifted["bone_length"] < 0.8 * float(pin["bone_length"]):
                continue
            # bone left beside the pin: the line r_pin + 1.5 mm further out must still hit the bone
            beside = guide_logic.plan_pin(obb, centre + sign * (offset + r_pin + 1.5) * across, z,
                                          self._pin_diameter(), float(self.guide_gap.value), guide_logic.BLOCK_HEIGHT)
            if "error" in beside or beside["bone_length"] < 0.5 * float(pin["bone_length"]):
                continue
            raw_entry = shifted["entry"] if segment != moved else (Minv @ np.append(shifted["entry"], 1.0))[:3]
            if self._pin_segment(raw_entry)[0] is None:
                continue
            shifted["segment"], shifted["centre"] = segment, centre + sign * offset * across
            return shifted
        return None

    def _update_pin_row_handles(self, handles_at, frames):
        node = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.PIN_ROW_HANDLES)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetSelectedColor(0.2, 0.8, 1.0)
            node.GetDisplayNode().SetGlyphScale(3.0)
            node.GetDisplayNode().SetTextScale(2.5)
            self._pin_row_handle_tag = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                                                        lambda caller, event: self._on_pin_row_handle_moved())
        elif getattr(self, "_pin_row_handle_tag", None) is None:
            self._pin_row_handle_tag = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                                                        lambda caller, event: self._on_pin_row_handle_moved())
        self._pin_row_handle_node_id = node.GetID()      # for releasing the observer on reload
        self._pin_row_frames = frames
        self._updating_handles = True
        try:
            node.RemoveAllControlPoints()
            for segment, position in handles_at.items():
                index = node.AddControlPointWorld(vtk.vtkVector3d(*position))
                node.SetNthControlPointLabel(index, f"{segment[0].upper()} row")
                node.SetNthControlPointDescription(index, segment)
            node.SetLocked(False)
            if node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(True)
        finally:
            self._updating_handles = False

    def _on_pin_row_handle_moved(self):
        if getattr(self, "_updating_handles", False):
            return
        node = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
        frames = getattr(self, "_pin_row_frames", {})
        if node is None:
            return
        boxes = {"proximal": self.pin_row_distance_proximal, "distal": self.pin_row_distance_distal}
        changed = False
        for i in range(node.GetNumberOfControlPoints()):
            segment = node.GetNthControlPointDescription(i)
            if segment not in frames:
                continue
            origin, s_dir = frames[segment]
            distance = float(np.dot(np.array(node.GetNthControlPointPositionWorld(i)) - origin, s_dir))
            distance = max(0.5, round(distance * 2.0) / 2.0)
            box = boxes[segment]
            if abs(box.value - distance) > 1e-6:
                was = box.blockSignals(True)
                box.setValue(distance)
                box.blockSignals(was)
                changed = True
        if changed:
            qt.QTimer.singleShot(0, self._place_pin_rows)

    # --- pins set AFTER the reduction (on the corrected bone) --------------------------------
    REDUCTION_PIN_MARKS = property(lambda self: "Doggybone guide pins" + self._bone_tag() + " (reduction)")

    def _correction_state(self):
        """(4x4 correction incl. fine shift, moved segment) or (None, None)."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        transform = slicer.mrmlScene.GetFirstNodeByName(names[2])
        if transform is None:
            return None, None
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)
        M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        moved = None
        for segment, name in (("proximal", names[0]), ("distal", names[1])):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetParentTransformNode() is not None \
                    and node.GetParentTransformNode().GetID() == transform.GetID():
                moved = segment
        return M, moved

    def _corrected_fragments(self):
        """{segment: (polydata in the corrected pose, OBB tree, cell locator)} - cached while neither the
        fragments nor the correction (incl. fine shift) changed."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        key = []
        for name in names[:2]:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None or node.GetPolyData() is None:
                key.append(None)
                continue
            matrix = vtk.vtkMatrix4x4()
            if node.GetParentTransformNode() is not None:
                node.GetParentTransformNode().GetMatrixTransformToWorld(matrix)
            key.append((node.GetID(), node.GetPolyData().GetMTime(),
                        tuple(round(matrix.GetElement(r, c), 9) for r in range(4) for c in range(4))))
        key = tuple(key)
        cached = getattr(self, "_corrected_fragments_cache", None)
        if cached is not None and cached[0] == key:
            return dict(cached[1])
        out = self._corrected_fragments_uncached(names)
        self._corrected_fragments_cache = (key, out)
        return dict(out)

    def _corrected_fragments_uncached(self, names):
        out = {}
        for segment, name in (("proximal", names[0]), ("distal", names[1])):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None or node.GetPolyData() is None or node.GetPolyData().GetNumberOfPoints() == 0:
                continue
            polydata = node.GetPolyData()
            parent = node.GetParentTransformNode()
            if parent is not None:
                matrix = vtk.vtkMatrix4x4()
                parent.GetMatrixTransformToWorld(matrix)
                transform = vtk.vtkTransform()
                transform.SetMatrix(matrix)
                moved = vtk.vtkTransformPolyDataFilter()
                moved.SetInputData(polydata)
                moved.SetTransform(transform)
                moved.Update()
                polydata = vtk.vtkPolyData()
                polydata.DeepCopy(moved.GetOutput())
            obb = vtk.vtkOBBTree()
            obb.SetDataSet(polydata)
            obb.BuildLocator()
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(polydata)
            locator.BuildLocator()
            out[segment] = (polydata, obb, locator)
        return out

    def _reduction_pin_frame(self, M, moved):
        """(common pin direction, {segment: shaft axis}, error) in the corrected pose.
        The pins lie PARALLEL TO THE CUT PLANE(S), so no pin can cross a cut anywhere along its length:
        CWO (faces in contact): in the contact plane and perpendicular to the condylar axis;
        OWO (faces apart): along the line both cut planes share. All pins parallel, towards the guide side."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        d, shaft, _across, error = self._guide_direction_frame(side, bone)
        if error:
            return None, None, error
        R = M[:3, :3] if M is not None else np.eye(3)
        condylar = "distal" if bone == "Femur" else "proximal"
        guide_side = guide_logic.unit(R @ d if moved == condylar else np.asarray(d, float))
        shafts = {segment: guide_logic.unit(R @ shaft if segment == moved else np.asarray(shaft, float))
                  for segment in ("proximal", "distal")}
        # the fragments' cut planes in the corrected pose
        planes = {"proximal": slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone)),
                  "distal": slicer.mrmlScene.GetFirstNodeByName(self._distal_plane_node_name(side, bone))}
        master = self.reference_plane_selector.currentText.lower()
        is_cwo = self.osteotomy_technique.currentText.startswith("CWO")
        normals = []
        for segment in ("proximal", "distal"):
            node = planes[segment] if is_cwo else planes.get(master)
            if node is None:
                continue
            _origin, normal = self._get_plane_origin_normal(node)
            normal = guide_logic.unit(np.asarray(normal, float))
            normals.append(guide_logic.unit(R @ normal) if segment == moved else normal)
        if not normals:
            return None, None, "Insert the cutting plane(s) first."
        ml_vector = self._get_ml_vector(side, bone)
        ml = None
        if ml_vector is not None and np.linalg.norm(ml_vector) > 1e-6:
            ml = guide_logic.unit(np.asarray(ml_vector, float))
            ml = guide_logic.unit(R @ ml) if moved == condylar else ml
        n0 = normals[0]
        n1 = normals[-1]
        common = np.cross(n0, n1)
        if np.linalg.norm(common) > np.sin(np.radians(1.0)):
            direction = guide_logic.unit(common)              # OWO: line shared by both cut planes
        elif ml is not None and np.linalg.norm(np.cross(n0, ml)) > 1e-6:
            direction = guide_logic.unit(np.cross(n0, ml))    # CWO: in the contact plane, perpendicular to condyles
        else:
            direction = guide_side - float(np.dot(guide_side, n0)) * n0
            if np.linalg.norm(direction) < 1e-6:
                return None, None, "Guide side is perpendicular to the cut - cannot orient the pins."
            direction = guide_logic.unit(direction)
        if float(np.dot(direction, guide_side)) < 0.0:
            direction = -direction
        # Pin tilt: the pins are turned WITHIN the cutting plane (about its normal) - they stay parallel
        # to the cut, so they can never meet the saw blade, but reach the bone from a better angle.
        # The axis is the cut of the condylar (unmoved) fragment; positive = towards proximal.
        tilt = float(self.pin_tilt.value) if getattr(self, "pin_tilt", None) is not None else 0.0
        self._pin_tilt_note = ""
        if abs(tilt) > 1e-6:
            axis = normals[0] if condylar == "proximal" else normals[-1]
            proximal_dir = -shafts["distal"] if shafts.get("distal") is not None else shafts["proximal"]
            if float(np.dot(np.cross(axis, direction), proximal_dir)) < 0.0:
                axis = -axis                                       # + tilts the pin heads proximally
            t = np.radians(tilt)
            direction = guide_logic.unit(direction * np.cos(t) + np.cross(axis, direction) * np.sin(t))
            other = normals[-1] if condylar == "proximal" else normals[0]
            oblique = float(np.degrees(np.arcsin(min(1.0, abs(float(np.dot(direction, other)))))))
            self._pin_tilt_note = (f"Pins tilted {tilt:+.0f}\N{DEGREE SIGN} within the cutting plane"
                                   + (f" (oblique {oblique:.1f}\N{DEGREE SIGN} to the other fragment's cut - "
                                      "distance from the cut is checked along the whole pin)" if oblique > 0.5 else "")
                                   + ".")
        return direction, shafts, None

    @staticmethod
    def _section_centroid(polydata, point, normal, largest=False):
        """Area centroid of the bone cross-section (outer contour piece nearest to point, or the
        largest loop of the whole section with largest=True)."""
        plane = vtk.vtkPlane()
        plane.SetOrigin(*[float(v) for v in point])
        plane.SetNormal(*[float(v) for v in normal])
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(polydata)
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputConnection(cutter.GetOutputPort())
        if largest:
            connectivity.SetExtractionModeToAllRegions()
        else:
            connectivity.SetExtractionModeToClosestPointRegion()
            connectivity.SetClosestPoint(*[float(v) for v in point])
        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(connectivity.GetOutputPort())
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        contour = stripper.GetOutput()
        if contour.GetNumberOfPoints() < 3:
            return None
        from vtk.util.numpy_support import vtk_to_numpy
        P = vtk_to_numpy(contour.GetPoints().GetData()).astype(float)
        n = guide_logic.unit(np.asarray(normal, float))
        e1, e2 = guide_logic._perp_frame(n)
        best = None
        lines = contour.GetLines()
        lines.InitTraversal()
        ids = vtk.vtkIdList()
        while lines.GetNextCell(ids):
            loop = np.array([P[ids.GetId(k)] for k in range(ids.GetNumberOfIds())])
            if len(loop) < 3:
                continue
            x, y = (loop - point) @ e1, (loop - point) @ e2
            area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            if abs(area) < 1e-9:
                continue
            cx = float(np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))) / (6.0 * area)
            cy = float(np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))) / (6.0 * area)
            if best is None or abs(area) > best[0]:
                best = (abs(area), np.asarray(point, float) + cx * e1 + cy * e2)
        return best[1] if best is not None else P.mean(axis=0)

    def _center_reduction_pin(self, point, fragments=None, frame=None):
        """Pin through the cross-section centre of the fragment at `point` (corrected pose),
        common direction. Returns pin dict (entry/far/top/axis/radii/segment) or dict(error=...)."""
        if fragments is None:
            fragments = self._corrected_fragments()
        if frame is None:
            M, moved = self._correction_state()
            if M is None:
                return {"error": "no correction preview"}
            direction, shafts, error = self._reduction_pin_frame(M, moved)
            if error:
                return {"error": error}
        else:
            direction, shafts = frame
        p = np.asarray(point, float)
        best = None
        for segment, (_poly, _obb, locator) in fragments.items():
            closest = [0.0, 0.0, 0.0]
            cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
            locator.FindClosestPoint(p.tolist(), closest, cell_id, sub_id, dist2)
            if best is None or float(dist2) < best[0]:
                best = (float(dist2), segment)
        if best is None:
            return {"error": "no corrected fragments"}
        segment = best[1]
        polydata, obb, _locator = fragments[segment]
        # section plane that CONTAINS the pin direction: re-centring from any point on the pin line
        # (e.g. its mark at the entry) gives the same pin, even with pins oblique to the shaft
        centre = self._section_centroid(polydata, p, self._pin_section_normal(shafts[segment], direction))
        if centre is None:
            return {"error": "no bone cross-section at that level"}
        pin = guide_logic.plan_pin(obb, centre, direction, self._pin_diameter(),
                                   float(self.guide_gap.value), guide_logic.BLOCK_HEIGHT)
        if "error" in pin:
            return pin
        pin["segment"] = segment
        pin["centre"] = centre
        return pin

    def _reduction_marked_pins(self):
        """Centred pins (corrected pose) for all marks set in reduction mode -> (pins, rejected)."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is None or node.GetNumberOfControlPoints() == 0:
            return [], []
        M, moved = self._correction_state()
        if M is None:
            return [], ["no correction preview"]
        direction, shafts, error = self._reduction_pin_frame(M, moved)
        if error:
            return [], [error]
        fragments = self._corrected_fragments()
        pins, rejected = [], []
        for i in range(node.GetNumberOfControlPoints()):
            label = node.GetNthControlPointLabel(i)
            point = np.array(node.GetNthControlPointPositionWorld(i))
            if self._pin_kind(node, i)[1] in ("free", "autofree"):
                pin = self._free_reduction_pin(point, fragments, direction)
            else:
                pin = self._center_reduction_pin(point, fragments, (direction, shafts))
            if "error" in pin:
                rejected.append(f"{label}: {pin['error']}")
                continue
            pin["label"] = label
            pins.append(pin)
        return pins, rejected

    def _reduction_guide_pins(self):
        """Pins for the reduction guide in the corrected pose: from the marks set after the reduction,
        else (older plans) the saw guide's pins moved into the corrected pose. -> (pins, moved, error)"""
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is not None:
            _M, moved = self._correction_state()
            if node.GetNumberOfControlPoints() == 0:
                return [], moved, "No pins - place the pin rows or set pins on the corrected bone."
            pins, rejected = self._reduction_marked_pins()
            if not pins:
                return [], moved, "No usable pins: " + "; ".join(rejected)
            return pins, moved, None
        # older plans without pins set after the reduction: the saw guide's pins moved into place
        return self._reduced_pins(self._load_planned_pins())

    # --- single pins: added by hand, moved with a small gizmo (always parallel) ------------------
    PIN_GIZMO = "Doggybone pin gizmo"
    PIN_GIZMO_LIFT = 10.0      # mm above the pin entry, on the pin axis - the pin itself stays clickable

    @staticmethod
    def _pin_kind(node, index):
        """(segment, kind) of a reduction pin mark; kind: 'auto' (pin row), 'manual' (added / moved by
        hand, centred in the bone) or 'free' (exactly where it was put)."""
        parts = (node.GetNthControlPointDescription(index) or "").split("|")
        return parts[0], (parts[1] if len(parts) > 1 and parts[1] else "auto")

    def _free_reduction_pin(self, point, fragments, direction):
        """Pin exactly through `point` (no centring), common direction, in the nearest fragment."""
        p = np.asarray(point, float)
        best = None
        for segment, (_poly, _obb, locator) in fragments.items():
            closest = [0.0, 0.0, 0.0]
            cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
            locator.FindClosestPoint(p.tolist(), closest, cell_id, sub_id, dist2)
            if best is None or float(dist2) < best[0]:
                best = (float(dist2), segment)
        if best is None:
            return {"error": "no corrected fragments"}
        segment = best[1]
        _poly, obb, _locator = fragments[segment]
        pin = guide_logic.plan_pin(obb, p, direction, self._pin_diameter(), float(self.guide_gap.value),
                                   guide_logic.BLOCK_HEIGHT)
        if "error" in pin:
            return pin
        pin["segment"], pin["centre"] = segment, p
        return pin

    def onDeleteSelectedPin(self):
        index = self._selected_pin_index()
        if index is None:
            self.guide_status.setStyleSheet("")
            self.guide_status.setText("No pin selected - click the pin in the 3D view first.")
            return
        self._remove_pin(index)
        self._select_reduction_pin(None)
        self._rebuild_after_pin_change()

    # --- reduction guide: pins are selected by clicking them (no tool button needed) ----------------
    def _install_pin_picker(self):
        self._remove_pin_picker()
        try:
            view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        except Exception:
            return
        interactor = view.interactor()
        if getattr(self, "_guide_view", None) is None:
            self._guide_view = view
        tags = [interactor.AddObserver(event, self._on_pin_picker, 0.9)
                for event in ("LeftButtonPressEvent", "LeftButtonReleaseEvent", "KeyPressEvent")]
        self._pin_picker = (interactor, tags)

    def _remove_pin_picker(self):
        entry = getattr(self, "_pin_picker", None)
        if entry:
            interactor, tags = entry
            for tag in tags:
                try:
                    interactor.RemoveObserver(tag)
                except Exception:
                    pass
        self._pin_picker = None

    def _on_pin_picker(self, caller, event):
        """Click (no drag) on a pin of the reduction guide: select it - the gizmo appears, drag it to move
        the pin (it stays parallel). Click elsewhere: deselect. Delete / Backspace: delete the selected
        pin. A drag rotates the view as usual."""
        try:
            if not self._guide_mode_is_reduction() or getattr(self, "_guide_active_tool", None) is not None:
                return
            if event == "KeyPressEvent":
                if caller.GetKeySym() in ("Delete", "BackSpace") and self._selected_pin_index() is not None:
                    qt.QTimer.singleShot(0, self.onDeleteSelectedPin)
                return
            x, y = caller.GetEventPosition()
            if event == "LeftButtonPressEvent":
                self._pin_press = None
                if caller.GetShiftKey() or caller.GetControlKey() or caller.GetAltKey() or self._pin_gizmo_hovered():
                    return
                self._pin_press = (x, y)
                return
            press, self._pin_press = getattr(self, "_pin_press", None), None
            if press is None or abs(x - press[0]) + abs(y - press[1]) > 4:
                return
            node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
            if node is None or node.GetNumberOfControlPoints() == 0:
                return
            mark = self._pin_under_mouse(x, y)
            if mark is not None:
                qt.QTimer.singleShot(0, lambda i=mark: self._select_reduction_pin(i, announce=True))
            elif self._selected_pin_index() is not None:
                qt.QTimer.singleShot(0, lambda: self._select_reduction_pin(None))
        except Exception:
            logging.exception("Doggybone: pin picker failed (continuing)")

    def _rebuild_after_pin_change(self):
        """An existing reduction guide follows the moved / deleted pin right away."""
        self._pins_dirty = False
        if slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME) is not None:
            qt.QTimer.singleShot(0, self._extrude_guide_body)

    def _keep_clear_of_cut(self, pin, fragments, direction, centred=False):
        """A pin too close to its cut (the saw guide could not take it) is moved away from the cut along
        the bone - just far enough. -> (pin, shift in mm) or (pin with 'error', 0)."""
        M, moved = self._correction_state()
        if M is None or "error" in pin:
            return pin, 0.0
        frames = self._fragment_cut_frames(M, moved, fragments)
        segment = pin["segment"]
        if segment not in frames:
            return pin, 0.0
        _origin, s_dir = frames[segment]
        z = guide_logic.unit(np.asarray(direction, float))
        along = np.asarray(s_dir, float) - float(np.dot(s_dir, z)) * z      # along the bone, away from the cut
        if np.linalg.norm(along) < 1e-6:
            return pin, 0.0
        along = guide_logic.unit(along)
        Minv = np.linalg.inv(M)
        shift = 0.0
        for _attempt in range(6):
            raw = pin["entry"] if segment != moved else (Minv @ np.append(pin["entry"], 1.0))[:3]
            ok_segment, reason = self._pin_segment(raw)
            found = re.search(r"\(([\d.]+) mm, needs ([\d.]+) mm\)", reason or "")
            if ok_segment is not None or not found:
                return pin, shift
            step = float(found.group(2)) - float(found.group(1)) + 0.3
            shift += step
            point = np.asarray(pin.get("centre", pin["entry"]), float) + step * along
            new = (self._center_reduction_pin(point, fragments, self._reduction_pin_frame(M, moved)[:2]) if centred
                   else self._free_reduction_pin(point, fragments, z))
            if "error" in new:
                return pin, 0.0
            if not centred:
                new["centre"] = point
            pin = new
        return pin, shift

    def _selected_pin_index(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        pin_id = getattr(self, "_selected_pin_id", None)
        if node is None or pin_id is None:
            return None
        index = node.GetNthControlPointIndexByID(pin_id)
        return index if index is not None and index >= 0 else None

    def _select_reduction_pin(self, index, announce=False):
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is None or index is None or index < 0 or index >= node.GetNumberOfControlPoints():
            self._selected_pin_id = None
            self._remove_pin_gizmo()
            return
        self._selected_pin_id = node.GetNthControlPointID(index)
        self._place_pin_gizmo()
        if announce:
            self.guide_status.setStyleSheet("")
            self.guide_status.setText(
                f"Pin {node.GetNthControlPointLabel(index)} selected - move it with the gizmo above it (it "
                "stays parallel); 'Delete pin' or the Delete key removes it, a click elsewhere "
                "deselects it.")

    def _pin_gizmo_axes(self, segment):
        M, moved = self._correction_state()
        if M is None:
            return None
        direction, shafts, error = self._reduction_pin_frame(M, moved)
        if error:
            return None
        z = guide_logic.unit(np.asarray(direction, float))
        s = np.asarray(shafts.get(segment, shafts["proximal"]), float)
        x = s - float(np.dot(s, z)) * z
        x = guide_logic.unit(x) if np.linalg.norm(x) > 1e-6 else guide_logic._perp_frame(z)[0]
        return x, np.cross(z, x), z

    def _remove_pin_gizmo(self):
        self._remove_pin_preview()
        gizmo = slicer.mrmlScene.GetFirstNodeByName(self.PIN_GIZMO)
        if gizmo is not None:
            slicer.mrmlScene.RemoveNode(gizmo)
        self._pin_gizmo_tags = None

    def _place_pin_gizmo(self):
        index = self._selected_pin_index()
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if index is None or node is None:
            self._remove_pin_gizmo()
            return
        segment, _kind = self._pin_kind(node, index)
        axes = self._pin_gizmo_axes(segment)
        if axes is None:
            self._remove_pin_gizmo()
            return
        gizmo = slicer.mrmlScene.GetFirstNodeByName(self.PIN_GIZMO)
        tags = getattr(self, "_pin_gizmo_tags", None)
        if gizmo is not None and (tags is None or tags[0] != gizmo.GetID()):
            # left over from an earlier widget (module reload): its observers would move THAT widget's
            # selected pin as well - always start with a fresh gizmo
            slicer.mrmlScene.RemoveNode(gizmo)
            gizmo = None
        if gizmo is None:
            gizmo = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", self.PIN_GIZMO)
            gizmo.CreateDefaultDisplayNodes()
            display = gizmo.GetDisplayNode()
            display.SetSelectedColor(1.0, 0.55, 0.1)
            display.SetColor(1.0, 0.55, 0.1)
            display.SetFillOpacity(0.25)
            display.SetPropertiesLabelVisibility(False)
            display.SetPointLabelsVisibility(False)
            display.SetHandlesInteractive(True)
            display.SetTranslationHandleVisibility(True)
            display.SetRotationHandleVisibility(False)
            display.SetScaleHandleVisibility(False)
            # two arrows perpendicular to the pin + the centre ball for free dragging; any movement
            # along the pin is removed, so the pin always stays parallel
            display.SetTranslationHandleComponentVisibility([True, True, False, True])
            display.SetInteractionHandleScale(4.0)
            # no snapping onto visible surfaces (bone, sleeves, the preview): smooth, continuous drag
            display.SetSnapMode(slicer.vtkMRMLMarkupsDisplayNode.SnapModeUnconstrained)
            display.SetOccludedVisibility(True)
            display.SetOccludedOpacity(0.6)
            self._pin_gizmo_tags = None
        if getattr(self, "_pin_gizmo_tags", None) is None or self._pin_gizmo_tags[0] != gizmo.GetID():
            self._pin_gizmo_tags = (gizmo.GetID(), [
                gizmo.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._on_pin_gizmo_moved),
                gizmo.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_pin_gizmo_moved),
                gizmo.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self._on_pin_gizmo_released)])
        position = np.array(node.GetNthControlPointPositionWorld(index), float)
        lifted = position + self.PIN_GIZMO_LIFT * np.asarray(axes[2], float)
        self._pin_mark_origin = position
        self._pin_gizmo_updating = True
        try:
            gizmo.SetAxesWorld(*[[float(v) for v in a] for a in axes])
            gizmo.SetOriginWorld(*[float(v) for v in lifted])
            gizmo.SetSize(5.0, 5.0)
            gizmo.GetDisplayNode().SetVisibility(True)
        finally:
            self._pin_gizmo_updating = False
        self._pin_gizmo_origin, self._pin_gizmo_z = lifted, np.asarray(axes[2], float)

    PIN_PREVIEW = "Doggybone pin preview"

    def _pin_gizmo_hovered(self):
        gizmo = slicer.mrmlScene.GetFirstNodeByName(self.PIN_GIZMO)
        display = gizmo.GetDisplayNode() if gizmo is not None else None
        if display is None or not display.GetVisibility():
            return False
        try:
            return display.GetActiveComponentType() != slicer.vtkMRMLMarkupsDisplayNode.ComponentNone
        except Exception:
            return False

    def _show_pin_preview(self, point):
        """Thin line along the pin axis through `point` while the gizmo is dragged (no recalculation)."""
        z = getattr(self, "_pin_gizmo_z", None)
        if z is None:
            return
        node = slicer.mrmlScene.GetFirstNodeByName(self.PIN_PREVIEW)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.PIN_PREVIEW)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(1.0, 0.55, 0.1)
            node.GetDisplayNode().SetVisibility2D(False)
            node.SetSelectable(False)               # never picked / snapped onto while dragging
        line = vtk.vtkLineSource()
        p = np.asarray(point, float)
        line.SetPoint1(*[float(v) for v in p + 15.0 * z])
        line.SetPoint2(*[float(v) for v in p - 40.0 * z])
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(max(0.4, 0.5 * self._pin_diameter()))
        tube.SetNumberOfSides(12)
        tube.CappingOn()
        tube.Update()
        node.SetAndObservePolyData(tube.GetOutput())

    def _remove_pin_preview(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.PIN_PREVIEW)
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    def _on_pin_gizmo_moved(self, caller, event=None):
        if getattr(self, "_pin_gizmo_updating", False):
            return
        index = self._selected_pin_index()
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if index is None or node is None or getattr(self, "_pin_gizmo_origin", None) is None:
            return
        origin = [0.0, 0.0, 0.0]
        caller.GetOriginWorld(origin)
        z = self._pin_gizmo_z
        delta = np.asarray(origin, float) - self._pin_gizmo_origin
        delta = delta - float(np.dot(delta, z)) * z          # sideways only: the pin stays on its line
        target = getattr(self, "_pin_mark_origin", self._pin_gizmo_origin) + delta
        node.SetNthControlPointPositionWorld(index, *[float(v) for v in target])
        self._show_pin_preview(target)

    def _on_pin_gizmo_released(self, caller=None, event=None):
        if getattr(self, "_pin_gizmo_updating", False):
            return
        qt.QTimer.singleShot(0, self._finish_pin_move)

    def _finish_pin_move(self):
        self._remove_pin_preview()
        index = self._selected_pin_index()
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if index is None or node is None:
            return
        point = np.array(node.GetNthControlPointPositionWorld(index), float)
        M, moved = self._correction_state()
        if M is None:
            return
        direction, shafts, error = self._reduction_pin_frame(M, moved)
        if error:
            self.guide_status.setText(error)
            return
        fragments = self._corrected_fragments()
        centred = self.pin_centre_checkbox.isChecked()
        pin = (self._center_reduction_pin(point, fragments, (direction, shafts)) if centred
               else self._free_reduction_pin(point, fragments, direction))
        label = node.GetNthControlPointLabel(index)
        if "error" in pin:
            back = getattr(self, "_pin_mark_origin", self._pin_gizmo_origin)
            node.SetNthControlPointPositionWorld(index, *[float(v) for v in back])
            self._place_pin_gizmo()
            self._pin_feedback(f"Pin {label} not moved: {pin['error']}.", point, ok=False)
            return
        pin.setdefault("centre", point)
        pin, shift = self._keep_clear_of_cut(pin, fragments, direction, centred)
        node.SetNthControlPointPositionWorld(index, *[float(v) for v in pin["entry"]])
        node.SetNthControlPointDescription(index, f"{pin['segment']}|{'manual' if centred else 'free'}")
        self._relabel_pins(node)
        self._pins_dirty = True
        self._show_reduction_pin_axes()
        self._place_pin_gizmo()
        # would the saw guide accept it (distance to the cut in the as-scanned pose)?
        raw_entry = np.asarray(pin["entry"], float)
        if pin["segment"] == moved:
            raw_entry = (np.linalg.inv(M) @ np.append(raw_entry, 1.0))[:3]
        ok_segment, reason = self._pin_segment(raw_entry)
        label = node.GetNthControlPointLabel(index)
        text = (f"Pin {label} moved ({pin['segment']} fragment, {'centred' if centred else 'free'}), "
                f"{pin['bone_length']:.1f} mm in bone, parallel to the other pins. Rebuild the guide "
                "('Extrude') when all pins are placed."
                + (f"\nPushed {shift:.1f} mm away from the cut so the saw guide can take it." if shift > 0 else ""))
        if ok_segment is None:
            self.guide_status.setStyleSheet(STATUS_WARN_STYLE)
            self.guide_status.setText(text + f"\nWarning: the saw guide cannot take this pin - {reason}.")
        else:
            self.guide_status.setStyleSheet("")
            self.guide_status.setText(text)
        if getattr(self, "_guide_active_tool", None) != "pins":
            self._rebuild_after_pin_change()

    def _show_reduction_pin_axes(self):
        """Axes of the pins set after the reduction (corrected pose) - ALWAYS into the reduction node,
        also when called while the saw guide is shown (they must never overwrite the saw guide's axes)."""
        pins, _rejected = self._reduction_marked_pins()
        self._reduction_pin_shafts = [
            (pin["label"], np.asarray(pin["far"], float) - 3.0 * np.asarray(pin["axis"], float),
             np.asarray(pin["top"], float) + 2.0 * np.asarray(pin["axis"], float), 0.5 * float(pin["diameter"]))
            for pin in pins if all(k in pin for k in ("far", "top", "axis", "diameter", "label"))]
        saved = getattr(self, "_guide_mode", "saw")
        self._guide_mode = "reduction"
        try:
            self._set_pin_axes(guide_logic.pin_axis_polydata(pins) if pins else None)
            node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PIN_AXES_NAME)
        finally:
            self._guide_mode = saved
        if node is not None and node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(saved == "reduction")

    def _toggle_reduction_pin(self, hit):
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.REDUCTION_PIN_MARKS)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
            node.GetDisplayNode().SetSelectedColor(0.95, 0.95, 0.3)
            node.GetDisplayNode().SetGlyphScale(1.5)
            node.GetDisplayNode().SetTextScale(2.5)
        p = np.asarray(hit, float)
        for i in range(node.GetNumberOfControlPoints()):
            if np.linalg.norm(np.array(node.GetNthControlPointPositionWorld(i)) - p) < 2.0:
                node.RemoveNthControlPoint(i)
                self._relabel_pins(node)
                self._pins_dirty = True
                self._show_reduction_pin_axes()
                self.guide_status.setText(f"Pin removed ({node.GetNumberOfControlPoints()} set).")
                return
        centred = self.pin_centre_checkbox.isChecked() if hasattr(self, "pin_centre_checkbox") else True
        if centred:
            pin = self._center_reduction_pin(p)
        else:
            M, moved = self._correction_state()
            direction, _shafts, error = self._reduction_pin_frame(M, moved) if M is not None \
                else (None, None, "no correction preview")
            pin = {"error": error} if error else self._free_reduction_pin(p, self._corrected_fragments(), direction)
        if "error" in pin:
            self._pin_feedback(f"No pin there: {pin['error']}.", p, ok=False)
            return
        if self._pin_distribution_manual() and hasattr(self, "pin_row_count"):
            limit = int(self.pin_row_count.value)
            same = sum(1 for i in range(node.GetNumberOfControlPoints())
                       if self._pin_kind(node, i)[0] == pin["segment"])
            if same >= limit:
                self._pin_feedback(
                    f"No pin added: the {pin['segment']} fragment already has {same} pin(s) - 'Pins' is set "
                    f"to {limit}. Click a pin to remove it, or raise 'Pins'.", p, ok=False)
                return
        M_, moved_ = self._correction_state()
        direction_ = self._reduction_pin_frame(M_, moved_)[0] if M_ is not None else None
        shift = 0.0
        if direction_ is not None:
            pin.setdefault("centre", p)
            pin, shift = self._keep_clear_of_cut(pin, self._corrected_fragments(), direction_, centred)
        index = node.AddControlPointWorld(vtk.vtkVector3d(*[float(v) for v in pin["entry"]]))
        node.SetNthControlPointDescription(index, f"{pin['segment']}|{'manual' if centred else 'free'}")
        self._relabel_pins(node)
        self._pins_dirty = True
        self._pin_feedback(None, p, ok=True)
        self._show_reduction_pin_axes()
        self._select_reduction_pin(index)
        self.guide_status.setText(
            f"Pin {node.GetNthControlPointLabel(index)} set ({pin['segment']} fragment) "
            f"{'through the bone centre' if centred else 'where you clicked'}, "
            f"{pin['bone_length']:.1f} mm in bone; {node.GetNumberOfControlPoints()} set. All pins parallel to "
            "each other and to the cut plane(s). Fine-tune it with the small gizmo; press 'Place pins' "
            "again when all pins are set - the reduction guide is then built." + (f"\nMoved {shift:.1f} mm away from the cut so the saw guide can take it."
                                            if shift > 0 else ""))

    def _toggle_pin(self, hit):
        if self._guide_mode_is_reduction():
            self._toggle_reduction_pin(hit)
            return
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.GUIDE_PINS_NAME)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
            node.GetDisplayNode().SetSelectedColor(0.95, 0.95, 0.3)
            node.GetDisplayNode().SetGlyphScale(1.5)
            node.GetDisplayNode().SetTextScale(2.5)
        p = np.asarray(hit, float)
        for i in range(node.GetNumberOfControlPoints()):
            q = np.array(node.GetNthControlPointPositionWorld(i))
            if np.linalg.norm(q - p) < 2.0:
                node.RemoveNthControlPoint(i)
                self._relabel_pins(node)
                self._pins_dirty = True
                self.guide_status.setText(f"Pin removed ({node.GetNumberOfControlPoints()} marked). "
                                          "Press Extrude to update the sleeves.")
                return
        segment, reason = self._pin_segment(p)
        if segment is None:
            self._pin_feedback(f"No pin there: {reason}.", p, ok=False)
            return
        index = node.AddControlPointWorld(vtk.vtkVector3d(*p))
        node.SetNthControlPointDescription(index, segment)
        self._relabel_pins(node)
        self._pins_dirty = True
        self._pin_feedback(None, p, ok=True)
        self.guide_status.setText(f"Pin {node.GetNthControlPointLabel(index)} marked ({segment}); "
                                  f"{node.GetNumberOfControlPoints()} marked. Mark further pins, then "
                                  "press Extrude to build sleeves and bores.")

    @staticmethod
    def _relabel_pins(node):
        counts = {}
        for i in range(node.GetNumberOfControlPoints()):
            segment = (node.GetNthControlPointDescription(i) or "pin").split("|")[0] or "pin"
            counts[segment] = counts.get(segment, 0) + 1
            node.SetNthControlPointLabel(i, f"{segment[0].upper()}{counts[segment]}")

    def _set_pin_axes(self, polydata):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PIN_AXES_NAME)
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
            return
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_PIN_AXES_NAME)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(0.75, 0.75, 0.78)
        node.SetAndObservePolyData(polydata)
