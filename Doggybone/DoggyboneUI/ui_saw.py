# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""SawMixin: Saw guide: saw pins from reduction, saw box gizmos, saw slots.

Part of DoggyboneWidget (see Doggybone.py). Methods were moved here unchanged; they rely on
attributes created in DoggyboneWidget.setup() and on methods of the other mixins.
"""

from DoggyboneUI.common import *


class SawMixin:

    def _plan_saw_pins_from_reduction(self, mesh, gap):
        """Saw guide sleeves from the pins set after the reduction: pins of the moved fragment are
        taken back into its as-scanned pose (pre-angled automatically - parallel after reduction)."""
        reduced, rejected = self._reduction_marked_pins()
        M, moved = self._correction_state()
        if M is None:
            return [], "Pins not built: no correction preview."
        Minv = np.linalg.inv(M)
        obb = self._pin_obb_tree(mesh)
        pins, lengths, counts = [], [], {}
        for pin in reduced:
            entry, axis = np.asarray(pin["entry"], float), np.asarray(pin["axis"], float)
            if pin["segment"] == moved:
                entry = (Minv @ np.append(entry, 1.0))[:3]
                axis = guide_logic.unit(Minv[:3, :3] @ axis)
            segment, reason = self._pin_segment(entry)
            if segment is None:
                rejected.append(f"{pin['label']}: {reason}")
                continue
            planned = guide_logic.plan_pin(obb, entry, axis, pin["diameter"], gap, guide_logic.BLOCK_HEIGHT)
            if "error" in planned:
                rejected.append(f"{pin['label']}: {planned['error']} (outside the saw-guide surface)")
                continue
            planned["label"], planned["segment"] = pin["label"], pin["segment"]
            pins.append(planned)
            lengths.append(planned["bone_length"])
            counts[pin["segment"]] = counts.get(pin["segment"], 0) + 1
        touching = []
        for a_index in range(len(pins)):
            for b_index in range(a_index + 1, len(pins)):
                a, b = pins[a_index], pins[b_index]
                if np.linalg.norm(a["entry"] - b["entry"]) < a["r_out"] + b["r_out"]:
                    touching.append(f"{a['label']}/{b['label']}")
        note = ("Pin sleeves from the pins set after the reduction: " + ", ".join(f"{v} {k}" for k, v in counts.items())
                + (f", in bone {min(lengths):.1f}-{max(lengths):.1f} mm" if lengths else "")
                + " - pre-angled, parallel after the reduction.")
        if touching:
            note += ("\nSleeves touching (" + ", ".join(touching) + "): merged into one block, every bore "
                     "is cut through - no sleeve reaches into a neighbouring bore.")
        if rejected:
            note += "\nNot built: " + "; ".join(rejected)
        return pins, note

    SAW_PIN_PREVIEW_NAME = "Doggybone saw guide pin preview"

    def _show_saw_pin_preview(self):
        """Saw guide, before the guide exists: the sleeves of the pins set on the reduction guide,
        shown on the bone as it is (pre-angled), so the footprint can be painted around them."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.SAW_PIN_PREVIEW_NAME)
        body = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        has_body = body is not None and body.GetPolyData() is not None and body.GetPolyData().GetNumberOfPoints() > 0
        if self._guide_mode_is_reduction() or has_body:
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(False)
            return ""
        mesh_node = self._guide_mesh_node()
        mesh = mesh_node.GetPolyData() if mesh_node is not None else self._get_bone_polydata()
        if mesh is None or mesh.GetNumberOfPoints() == 0:
            return ""
        marks = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if marks is None or marks.GetNumberOfControlPoints() == 0:
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
            return "No pins yet - set them on the reduction guide first; they are then shown here before painting."
        try:
            pins, _note = self._plan_saw_pins_from_reduction(mesh, float(self.guide_gap.value))
        except Exception:
            logging.exception("Doggybone: saw pin preview failed")
            return ""
        if not pins:
            return ""
        append = vtk.vtkAppendPolyData()
        obb = self._pin_obb_tree(mesh)
        for pin in pins:
            append.AddInputData(guide_logic.sleeve_shell(obb, pin, float(self.guide_gap.value)))
        append.AddInputData(guide_logic.pin_axis_polydata(pins))
        append.Update()
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.SAW_PIN_PREVIEW_NAME)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(1.0, 0.55, 0.1)
            node.GetDisplayNode().SetOpacity(0.7)
        preview = vtk.vtkPolyData()
        preview.DeepCopy(append.GetOutput())
        node.SetAndObservePolyData(preview)
        node.GetDisplayNode().SetVisibility(True)
        return (f"Pin sleeves of the saw guide shown in orange ({len(pins)} pins, pre-angled) - paint the "
                "footprint so that it reaches around them.")

    def _plan_pins(self, slots, mesh, gap):
        """Plan every pin of the pins node. Returns (pins, note)."""
        marks = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if marks is not None and marks.GetNumberOfControlPoints() > 0:
            return self._plan_saw_pins_from_reduction(mesh, gap)
        node = slicer.mrmlScene.GetFirstNodeByName(self.SAW_PINS_NAME)
        if node is None or node.GetNumberOfControlPoints() == 0:
            return [], ""
        master = self.reference_plane_selector.currentText.lower()
        master_slot = next((slot for slot in slots if slot.get("label") == master), None)
        if master_slot is None:
            return [], "Pins not built: the master cut has no slot (paint the footprint across it)."
        direction = guide_logic.unit(master_slot["v"])          # outward, parallel to the master plane
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        transform = slicer.mrmlScene.GetFirstNodeByName(names[2])
        rotation = np.eye(3)
        moved = None                      # fragment that carries the correction transform
        if transform is not None:
            matrix = vtk.vtkMatrix4x4()
            transform.GetMatrixTransformToWorld(matrix)
            rotation = np.array([[matrix.GetElement(r, c) for c in range(3)] for r in range(3)])
            for segment, name in (("proximal", names[0]), ("distal", names[1])):
                fragment = slicer.mrmlScene.GetFirstNodeByName(name)
                if fragment is not None and fragment.GetParentTransformNode() is not None \
                        and fragment.GetParentTransformNode().GetID() == transform.GetID():
                    moved = segment
        # CONDYLAR segment (femur: distal, tibia: proximal) = anatomical reference: its pins stand
        # exactly perpendicular to the condylar axis (guide-side direction with its component along
        # the condylar axis removed). The OPPOSITE segment's pins are pre-angled by the correction so
        # that all pins are parallel after the reduction (fragment `moved` is rotated by R).
        condylar = "distal" if bone == "Femur" else "proximal"
        opposite = "proximal" if condylar == "distal" else "distal"
        condylar_direction = direction
        ml_vector = self._get_ml_vector(side, bone)
        if ml_vector is not None and np.linalg.norm(ml_vector) > 1e-6:
            ml_axis = guide_logic.unit(np.asarray(ml_vector, dtype=float))
            perpendicular = direction - float(np.dot(direction, ml_axis)) * ml_axis
            if np.linalg.norm(perpendicular) > 1e-6:
                condylar_direction = guide_logic.unit(perpendicular)
        if moved == condylar:
            opposite_direction = guide_logic.unit(rotation @ condylar_direction)
        else:
            opposite_direction = guide_logic.unit(rotation.T @ condylar_direction)
        pin_directions = {condylar: condylar_direction, opposite: opposite_direction}
        diameter = self._pin_diameter()
        obb = self._pin_obb_tree(mesh)
        pins, rejected, lengths, counts = [], [], [], {}
        for i in range(node.GetNumberOfControlPoints()):
            p = np.array(node.GetNthControlPointPositionWorld(i))
            segment, reason = self._pin_segment(p)
            label = node.GetNthControlPointLabel(i)
            if segment is None:
                rejected.append(f"{label}: {reason}")
                continue
            node.SetNthControlPointDescription(i, segment)
            axis = pin_directions[segment]
            pin = guide_logic.plan_pin(obb, p, axis, diameter, gap, guide_logic.BLOCK_HEIGHT)
            if "error" in pin:
                rejected.append(f"{label}: {pin['error']}")
                continue
            pin["label"], pin["segment"] = label, segment
            pins.append(pin)
            mark = pin["top"] + 0.3 * pin["axis"]            # keep the mark visible on the sleeve top
            node.SetNthControlPointPositionWorld(i, float(mark[0]), float(mark[1]), float(mark[2]))
            lengths.append(pin["bone_length"])
            counts[segment] = counts.get(segment, 0) + 1
        self._relabel_pins(node)
        touching = []
        for a_index in range(len(pins)):
            for b_index in range(a_index + 1, len(pins)):
                a, b = pins[a_index], pins[b_index]
                if np.linalg.norm(a["entry"] - b["entry"]) < a["r_out"] + b["r_out"]:
                    touching.append(f"{a['label']}/{b['label']}")
        note = (f"Pin sleeves: " + ", ".join(f"{v} {k}" for k, v in counts.items()) +
                (f", \u00d8 {diameter:.1f} mm, in bone {min(lengths):.1f}-{max(lengths):.1f} mm" if lengths else ""))
        group_angle = float(np.degrees(np.arccos(np.clip(float(np.dot(condylar_direction, opposite_direction)), -1.0, 1.0))))
        note += (f"\n{condylar.capitalize()} (condylar) pins perpendicular to the condylar axis; {opposite} pins "
                 f"pre-angled {group_angle:.1f}\N{DEGREE SIGN} - parallel after the reduction.")
        if touching:
            note += ("\nSleeves touching (" + ", ".join(touching) + "): merged into one block, every bore "
                     "is cut through - no sleeve reaches into a neighbouring bore.")
        if rejected:
            note += "\nNot built: " + "; ".join(rejected)
        return pins, note

    # --- saw boxes: one gizmo per box, moving / turning it only WITHIN its cutting plane -------------
    def _saw_box_gizmo_name(self, label):
        return f"Doggybone saw box{self._bone_tag()} {label}"        # per bone (femur: plain name)

    def _saw_box_adjustment(self, label):
        node = slicer.mrmlScene.GetFirstNodeByName(self._saw_box_gizmo_name(label))
        if node is None:
            return {}
        try:
            theta, du, dv = (float(x) for x in (node.GetAttribute("Doggybone.adjust") or "0,0,0").split(","))
        except ValueError:
            return {}
        return {"theta": theta, "du": du, "dv": dv}

    def _place_saw_box_gizmos(self, slots):
        """Plane gizmo on each saw box's face in the cutting plane: arrows along the kerf and up/down,
        a ring to turn the saw direction - the kerf itself never leaves the planned cut."""
        tags = getattr(self, "_saw_box_gizmo_tags", {})
        for slot in slots:
            label = slot.get("label", "proximal")
            name = self._saw_box_gizmo_name(label)
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", name)
                node.CreateDefaultDisplayNodes()
                node.SetAttribute("Doggybone.adjust", "0,0,0")
                display = node.GetDisplayNode()
                display.SetSelectedColor(0.2, 0.9, 0.5)
                display.SetColor(0.2, 0.9, 0.5)
                display.SetFillOpacity(0.15)
                display.SetPropertiesLabelVisibility(False)
                display.SetPointLabelsVisibility(False)
                display.SetHandlesInteractive(True)
                display.SetTranslationHandleVisibility(True)
                display.SetRotationHandleVisibility(True)
                display.SetScaleHandleVisibility(False)
                display.SetTranslationHandleComponentVisibility([True, True, False, False])
                display.SetRotationHandleComponentVisibility([False, False, True, False])
                display.SetInteractionHandleScale(4.0)
            old = tags.get(label)
            if old is not None and old[0] == node.GetID():
                node.RemoveObserver(old[1])
            C, u, v, n = (np.asarray(slot[k], float) for k in ("centre", "u", "v", "normal"))
            bottom = float(np.mean(slot["bottom"]))
            top = float(slot["v_top"])
            centre = C + 0.5 * (float(slot["u0"]) + float(slot["u1"])) * u + 0.5 * (bottom + top) * v
            self._saw_box_updating = True
            try:
                node.SetAxesWorld([float(x) for x in u], [float(x) for x in v], [float(x) for x in np.cross(u, v)])
                node.SetOriginWorld(*[float(x) for x in centre])
                node.SetSize(float(slot["u1"]) - float(slot["u0"]), max(top - bottom, 2.0))
                node.SetAttribute("Doggybone.ref", ",".join(f"{x:.6f}" for x in list(centre) + list(u) + list(v)))
                node.GetDisplayNode().SetVisibility(True)
            finally:
                self._saw_box_updating = False
            tag = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                                   lambda caller, event, l=label: self._on_saw_box_moved(l))
            tags[label] = (node.GetID(), tag)
        self._saw_box_gizmo_tags = tags

    def _on_saw_box_moved(self, label):
        if getattr(self, "_saw_box_updating", False):
            return
        node = slicer.mrmlScene.GetFirstNodeByName(self._saw_box_gizmo_name(label))
        ref = (node.GetAttribute("Doggybone.ref") or "") if node is not None else ""
        values = [float(x) for x in ref.split(",")] if ref else []
        if len(values) != 9:
            return
        c0, u0, v0 = np.array(values[0:3]), np.array(values[3:6]), np.array(values[6:9])
        origin = [0.0, 0.0, 0.0]
        node.GetOriginWorld(origin)
        x_axis, y_axis, z_axis = [0.0] * 3, [0.0] * 3, [0.0] * 3
        node.GetAxesWorld(x_axis, y_axis, z_axis)
        x_axis = np.asarray(x_axis, float)
        adj = self._saw_box_adjustment(label)
        # rotation of the box about the plane normal (from the old to the new kerf direction)
        dtheta = float(np.arctan2(np.dot(np.cross(u0, x_axis), np.cross(v0, u0)), np.dot(u0, x_axis)))
        shift = np.asarray(origin, float) - c0
        theta = adj.get("theta", 0.0) + dtheta
        du = adj.get("du", 0.0) + float(np.dot(shift, u0))
        dv = adj.get("dv", 0.0) + float(np.dot(shift, v0))
        node.SetAttribute("Doggybone.adjust", f"{theta:.6f},{du:.4f},{dv:.4f}")
        # the current pose becomes the new reference - a second drag adds only ITS own change
        y_now = [0.0] * 3
        node.GetAxesWorld([0.0] * 3, y_now, [0.0] * 3)
        node.SetAttribute("Doggybone.ref", ",".join(f"{x:.6f}" for x in list(origin) + list(x_axis) + list(y_now)))
        self.guide_status.setStyleSheet(STATUS_WARN_STYLE)
        self.guide_status.setText(
            f"Saw box {label} adjusted in its cutting plane: turned {np.degrees(theta):+.1f}\N{DEGREE SIGN}, "
            f"extended {du:+.1f} mm along the kerf, height {dv:+.1f} mm - press 'Extrude' to "
            "rebuild the guide. The kerf stays exactly in the planned cut.")

    def _bridge_size(self):
        """Corner bridge size (mm) from the UI - closed slot ends / break-off corners of the saw block."""
        spin = getattr(self, "guide_bridge_size", None)
        return float(spin.value) if spin is not None else guide_logic.BRIDGE_WIDTH

    def _on_bridge_size_changed(self):
        guide_logic.configure_slots(bridge_size=self._bridge_size())
        self._refresh_guide_body_if_present()

    def _plan_saw_slots(self, patch, gap):
        """Slots for the cutting planes that exist. Returns (slots, notes, pin text)."""
        guide_logic.configure_slots(bridge_size=self._bridge_size())
        from vtk.util.numpy_support import vtk_to_numpy
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        planes = []
        # opening wedge: only the MASTER plane is a real cut - the second plane is a virtual target
        # and must never get a slot / saw face
        owo = self.osteotomy_technique.currentText.startswith("OWO")
        master = self.reference_plane_selector.currentText.lower()
        for label, name in (("proximal", self._plane_node_name(side, bone)),
                            ("distal", self._distal_plane_node_name(side, bone))):
            if owo and label != master:
                continue
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                origin, normal = self._get_plane_origin_normal(node)
                planes.append((label, np.asarray(origin, float), guide_logic.unit(np.asarray(normal, float))))
        bone_polydata = self._get_bone_polydata()
        if not planes or bone_polydata is None:
            return [], [], "no cutting plane / bone surface"
        bone_points = vtk_to_numpy(bone_polydata.GetPoints().GetData()).astype(float)
        patch_points = vtk_to_numpy(patch.GetPoints().GetData()).astype(float)
        patch_normals = patch.GetPointData().GetNormals()
        if patch_normals is None:
            nrm = vtk.vtkPolyDataNormals()
            nrm.SetInputData(patch)
            nrm.SplittingOff()
            nrm.ConsistencyOn()
            nrm.Update()
            patch_normals = nrm.GetOutput().GetPointData().GetNormals()
        patch_normals = vtk_to_numpy(patch_normals).astype(float)
        slots, notes, diameters = [], [], []
        near_any = np.zeros(len(patch_points), dtype=bool)
        for _label, origin_, normal_ in planes:
            near_any |= np.abs((patch_points - origin_) @ normal_) <= 4.0
        common_v = patch_normals[near_any].mean(axis=0) if near_any.any() else None
        # saw direction = the chosen guide side (projected into each cutting plane). The mean normal of
        # the footprint was used before - with a footprint reaching round the bone it tilted the saw
        # direction by up to ~70 deg, the slot then ran along the long diagonal of the cross-section and
        # stuck out far beyond the bone. Turn it with the saw box gizmo if needed.
        d_side, _shaft, _across, frame_error = self._guide_direction_frame(side, bone)
        if not frame_error and d_side is not None:
            common_v = np.asarray(d_side, float)
        for index, (label, origin, normal) in enumerate(planes):
            other = (planes[1 - index][1], planes[1 - index][2]) if len(planes) == 2 else None
            strip = np.abs((patch_points - origin) @ normal) <= 4.0
            if not strip.any():
                notes.append(f"{label}: footprint does not reach this cut - no slot")
                continue
            contour = guide_logic.section_contour(bone_polydata, origin, normal)
            diameter = guide_logic.contour_diameter(contour, normal)
            if diameter is not None:
                diameters.append(diameter)
            slot = guide_logic.plan_slot(bone_points, contour, origin, normal, other,
                                         common_v if common_v is not None else patch_normals[strip].mean(axis=0), gap,
                                         adjust=self._saw_box_adjustment(label),
                                         near_fraction=0.5 if (getattr(self, "guide_slot_extent", None) is not None
                                                               and self.guide_slot_extent.currentText.startswith("Guide"))
                                         else None)
            if "error" in slot:
                notes.append(f"{label}: {slot['error']}")
                continue
            slot["label"] = label
            slots.append(slot)
            notes.append(f"{label} {slot['length']:.1f} mm long (bone {slot['bone_width']:.1f} mm)")
        stop_box = getattr(self, "guide_hinge_stop", None)
        if stop_box is not None and len(slots) == 2:
            note = guide_logic.apply_hinge_stop(slots, float(stop_box.value))
            if note:
                notes.append(note)
        chosen = self.guide_pin.currentText
        d = min(diameters) if diameters else None
        if not chosen.startswith("Auto"):
            pin = chosen + (f" (chosen; bone \u00d8 {d:.1f} mm)" if d is not None else " (chosen)")
        elif d is not None:
            pin = (f"{guide_logic.auto_pin_diameter(d):.1f} mm auto (bone \u00d8 {d:.1f} mm; up to "
                   f"{guide_logic.PIN_THRESHOLD_MM:.0f} mm -> 1.2 mm, larger -> 1.6 mm)")
        else:
            pin = "1.6 mm auto (bone diameter not measurable)"
        return slots, notes, pin
