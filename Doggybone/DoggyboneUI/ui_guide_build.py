# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""GuideBuildMixin: Guide body: extrusion, STL export, patient node, labels/embossing, guide mode, ROI extraction.

Part of DoggyboneWidget (see Doggybone.py). Methods were moved here unchanged; they rely on
attributes created in DoggyboneWidget.setup() and on methods of the other mixins.
"""

from DoggyboneUI.common import *


class GuideBuildMixin:

    # --- extrusion --------------------------------------------------------
    def onExtrudeGuideBody(self):
        self._stop_guide_edit()
        self._extrude_guide_body()

    # fixed guide-body parameters (the former fields Rim smoothing / Top smoothing / Min. thickness)
    GUIDE_RIM_ROUND = 1.0          # mm, rounded rim where the painted region ends
    GUIDE_OUTER_SMOOTH = 0.75      # mm, smoothing of the outer surface (inner fit surface stays exact)
    GUIDE_MIN_FEATURE = 0.3        # mm, thinner parts of the guide are removed

    def _extrude_guide_body(self):
        self._pins_dirty = False
        if self._guide_mode_is_reduction() and self._reduction_is_bar():
            self._build_reduction_bar()
            return
        if self._ensure_current_reduction_surface():
            return          # footprint had to be discarded - nothing to extrude yet
        self.guide_status.setStyleSheet("")
        mesh_node = self._guide_mesh_node()
        if mesh_node is None:
            self.guide_status.setText("Paint a footprint first (Brush + / Lasso +).")
            return
        mesh = mesh_node.GetPolyData()
        external = self._guide_external(mesh)
        painted = guide_logic.get_selection(mesh)
        main_part = guide_logic.largest_selection_component(mesh, painted & external)
        if (painted & ~main_part & external).any():
            guide_logic.set_selection(mesh, main_part)          # drop small free painted islands
            self._update_guide_footprint_display()
        selection = guide_logic.close_selection(mesh, main_part) & external
        selection = guide_logic.smooth_selection_rim(mesh, selection, self.GUIDE_RIM_ROUND) & external
        # the painted patch is only used to plan the saw slots (which cut it reaches, its normals) -
        # the guide body itself is the volume body built from mesh + selection below
        patch = guide_logic.patch_from_selection(mesh, selection, lift=0.0)
        patch = guide_logic.make_patch_manifold(patch)
        patch = guide_logic.clean_patch(patch, keep_fraction=0.999)
        if patch.GetNumberOfCells() == 0:
            self._set_guide_body(None)
            self.guide_status.setText("The footprint is empty - no guide body.")
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            gap = float(self.guide_gap.value)
            thickness = float(self.guide_thickness.value)
            guide_logic.configure_slots(
                blade_thickness=float(self.guide_blade.currentText.split()[0]),
                block_height=float(self.guide_block_height.currentText.split()[0]),
                bridge_size=self._bridge_size(), wall=self._box_wall())
            if self._guide_mode_is_reduction():
                slots, slot_notes, pin = [], [], ""          # no saw slots on the reduction guide
            else:
                slots, slot_notes, pin = self._plan_saw_slots(patch, gap)
                self._place_saw_box_gizmos(slots)
            block = (getattr(self, "guide_saw_type", None) is not None
                     and self.guide_saw_type.currentText == "Block" and not self._guide_mode_is_reduction())
            block_side = None
            if slots and block and len(slots) == 1:
                # single cut: the plate goes on the side of the cut where most of the guide sits
                from vtk.util.numpy_support import vtk_to_numpy
                side_values = (vtk_to_numpy(patch.GetPoints().GetData()).astype(float)
                               - np.asarray(slots[0]["centre"], float)) @ np.asarray(slots[0]["normal"], float)
                block_side = 1.0 if float(np.count_nonzero(side_values > 0)) >= 0.5 * len(side_values) else -1.0
            if self._guide_mode_is_reduction():
                pins, pin_note = self._plan_reduction_pins(mesh, gap)
            else:
                pins, pin_note = self._plan_pins(slots, mesh, gap)
                self._store_planned_pins(pins)
            # saw block(s): exact meshes, fused into the volume body and kept as they are
            parts = []
            if len(slots) == 2:
                # ONE block for both cuts: outer faces extended until they close, space between filled,
                # both kerfs cut through - no crossing plates, no fragments
                # the seat of the block follows the WHOLE bone - the footprint ROI mesh is cropped and the block
                # reaches beyond it (a seat guessed there ran into the bone and left ragged ends)
                bone_seat = self._get_bone_polydata()
                if bone_seat is None or bone_seat.GetNumberOfPoints() == 0:
                    bone_seat = mesh
                parts.append(guide_logic.saw_block_pair(slots[0], slots[1], self._pin_obb_tree(bone_seat), gap,
                                                        block=block, pins=pins))
                strut_box = getattr(self, "guide_hinge_strut", None)
                if strut_box is not None and strut_box.currentText.startswith("Break"):
                    strut = guide_logic.hinge_strut(slots, block=parts[-1])
                    if strut is not None:
                        parts.append(strut)
            else:
                for slot in slots:
                    parts += guide_logic.saw_block_shells(slot, None, block_side=block_side)
            self._set_pin_axes(guide_logic.pin_axis_polydata(pins) if pins else None)
            side_v = self.osteotomy_side.currentText.lower()
            d_v, _s, _a, frame_error = self._guide_direction_frame(side_v, self.osteotomy_bone.currentText)
            bone_v = self._guide_source_polydata()
            if frame_error or bone_v is None:
                body, volume_info = None, {"error": frame_error or "no bone surface"}
            else:
                # sleeves are built inside the volume body (exact cylinders, every bore cut through all)
                body, volume_info = guide_logic.sdf_guide_body(
                    bone_v, mesh, selection, d_v, gap, thickness, slots=slots, pins=pins, block=block,
                    parts=parts, bridges=False, trim_undercut=True, rim_round=self.GUIDE_RIM_ROUND,
                    min_feature=self.GUIDE_MIN_FEATURE, outer_smooth=self.GUIDE_OUTER_SMOOTH, sleeves=True)
            if body is None:
                self.guide_status.setStyleSheet(STATUS_WARN_STYLE)
                self.guide_status.setText("Volume guide: " + str(volume_info.get("error", "failed")))
                return
            parts.insert(0, body)
            volume_note = (f"Volume body {volume_info['volume_mm3']:.0f} mm\N{SUPERSCRIPT THREE}"
                           + (f", undercut trimmed {volume_info['undercut_trimmed_mm3']:.0f} mm\N{SUPERSCRIPT THREE}"
                              if volume_info.get("undercut_trimmed_mm3", 0) >= 1 else "")
                           + (f", thin edges < {self.GUIDE_MIN_FEATURE:.1f} mm removed "
                              f"({volume_info['thin_removed_mm3']:.1f} mm\N{SUPERSCRIPT THREE})"
                              if volume_info.get("thin_removed_mm3", 0) > 0 else "")
                           + (f", pits in the bone bridged ({volume_info['pits_closed_mm3']:.1f} mm\N{SUPERSCRIPT THREE})"
                              if volume_info.get("pits_closed_mm3", 0) >= 0.5 else "")
                           + (f", {volume_info['islands_dropped']} loose piece(s) removed"
                              if volume_info.get("islands_dropped") else "")
                           + f" ({volume_info.get('t_total', 0):.0f} s).")
            if volume_info.get("pieces", 1) > 1:
                volume_note += (f"\nWARNING: the volume body is in {volume_info['pieces']} pieces - the "
                                "footprint does not reach around the kerf / the saw block. Paint further "
                                "towards the ends of the cut.")
            open_edges = sum(guide_logic.open_edge_count(part) for part in parts)
            label_note = ""
            if not self._guide_mode_is_reduction() and self._label_wanted():
                if self._label_mode() == "Break-off tag":
                    joined = vtk.vtkAppendPolyData()
                    for part in parts:
                        joined.AddInputData(part)
                    joined.Update()
                    side_ = self.osteotomy_side.currentText.lower()
                    up_, _shaft, _across, frame_error = self._guide_direction_frame(side_, self.osteotomy_bone.currentText)
                    label, label_note = (None, "Label tag skipped: no guide frame.") if frame_error else \
                        self._label_tag(joined.GetOutput(), up_, self._pin_obb_tree(mesh))
                else:
                    label, label_note = self._saw_guide_label(parts)
                    if label is None:          # no flat spot on the guide: the break-off tag instead
                        joined = vtk.vtkAppendPolyData()
                        for part in parts:
                            joined.AddInputData(part)
                        joined.Update()
                        side_ = self.osteotomy_side.currentText.lower()
                        up_, _shaft, _across, frame_error = self._guide_direction_frame(
                            side_, self.osteotomy_bone.currentText)
                        if not frame_error:
                            label, tag_note = self._label_tag(joined.GetOutput(), up_, self._pin_obb_tree(mesh))
                            if label is not None:
                                label_note = ("No flat area for the label on the guide - break-off tag used "
                                              "instead. " + tag_note)
                if label is not None:
                    parts.append(label)
            append = vtk.vtkAppendPolyData()
            for part in parts:
                append.AddInputData(part)
            append.Update()
            guide = vtk.vtkPolyData()
            guide.DeepCopy(append.GetOutput())
            self._set_guide_body(guide)
            if not self._guide_mode_is_reduction():
                self._show_saw_pin_preview()           # the real sleeves are in the guide now - hide the preview
            closed = "all parts closed" if open_edges == 0 else f"NOT closed ({open_edges} open edges)"
            if self._guide_mode_is_reduction():
                self.guide_status.setText(
                    f"Reduction guide extruded on the corrected fragments: fit gap {gap:.2f} mm, "
                    f"wall {thickness:.1f} mm, {closed}." + (f"\n{pin_note}" if pin_note else "")
                    + (f"\n{volume_note}" if volume_note else ""))
            else:
                self.guide_status.setText(
                    f"Guide extruded: fit gap {gap:.2f} mm, wall {thickness:.1f} mm, {closed}.\n"
                    f"Saw slots {guide_logic.SLOT_WIDTH:.1f} mm (blade {guide_logic.BLADE_THICKNESS:.1f} mm), "
                    f"kerf inside the wedge: " + ("; ".join(slot_notes) if slot_notes else "none") +
                    f".\nPins: {pin}" + (f"\n{pin_note}" if pin_note else "") + (f"\n{label_note}" if label_note else "")
                    + (f"\n{volume_note}" if volume_note else ""))
        finally:
            qt.QApplication.restoreOverrideCursor()

    GUIDE_BODY_PATTERN = re.compile(r"^Doggybone guide body( tibia)?( \(reduction\))?$")

    def _all_guide_bodies(self):
        """Every finished guide body in the scene: [(node, side, bone, mode)], femur before tibia,
        saw (osteotomy) before reduction. Side / bone / mode come from the node itself (set when
        the body was built); the bone and mode can also be read from the node name."""
        found = []
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            match = self.GUIDE_BODY_PATTERN.match(node.GetName() or "")
            if not match:
                continue
            polydata = node.GetPolyData()
            if polydata is None or polydata.GetNumberOfPoints() == 0:
                continue
            bone = node.GetAttribute("Doggybone.GuideBone") or ("Tibia" if match.group(1) else "Femur")
            mode = node.GetAttribute("Doggybone.GuideMode") or ("reduction" if match.group(2) else "saw")
            side = node.GetAttribute("Doggybone.GuideSide")       # None: built before sides were stored
            found.append((node, side, bone, mode))
        found.sort(key=lambda item: (item[2] != "Femur", item[3] != "saw"))
        return found

    # What a guide is built from; the export warns when one of these changed after the guide was built.
    PLAN_PART_NAMES = {"planes": "cutting planes", "correction": "correction / fine shift",
                       "reduction pins": "reduction pins", "saw pins": "saw guide pins",
                       "saw blocks": "saw block size / position", "footprint": "painted footprint",
                       "bone surface": "bone segmentation"}

    @staticmethod
    def _polydata_fingerprint(polydata):
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return "none"
        from vtk.util.numpy_support import vtk_to_numpy
        points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
        return "%d;%d;%s;%s" % (polydata.GetNumberOfPoints(), polydata.GetNumberOfCells(),
                                ",".join(f"{v:.2f}" for v in polydata.GetBounds()),
                                ",".join(f"{v:.2f}" for v in points.mean(axis=0)))

    def _guide_plan_signature(self, side, bone, mode):
        """{part: fingerprint} of the inputs of the guide for this side ('Right'/'Left'), bone and
        mode ('saw'/'reduction'), read from the scene by node name (independent of the current
        selection). Only geometry - never widget values, so a parameter change for another guide
        does not mark this one as outdated."""
        import hashlib
        side = side.lower()
        tag = " tibia" if bone == "Tibia" else ""
        suffix = " (reduction)" if mode == "reduction" else ""
        get = slicer.mrmlScene.GetFirstNodeByName

        def digest(text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

        def plane_text(node):
            if node is None:
                return "none"
            origin, normal = self._get_plane_origin_normal(node)
            return ",".join(f"{v:.2f}" for v in list(origin) + list(normal))

        def markups_text(node):
            if node is None:
                return "none"
            points = [node.GetNthControlPointPositionWorld(i) for i in range(node.GetNumberOfControlPoints())]
            attributes = [f"{name}={node.GetAttribute(name)}" for name in sorted(node.GetAttributeNames() or [])
                          if name.startswith("Doggybone.")]
            return ";".join(",".join(f"{v:.2f}" for v in p) for p in points) + "|" + ";".join(attributes)

        def model_text(node):
            return "none" if node is None else self._polydata_fingerprint(node.GetPolyData())

        parts = {}
        parts["planes"] = digest(plane_text(get(self._plane_node_name(side, bone))) + "|" +
                                 plane_text(get(self._distal_plane_node_name(side, bone))))
        transform = get(self._preview_node_names(side, bone)[2])
        if transform is None:
            parts["correction"] = "none"
        else:
            matrix = vtk.vtkMatrix4x4()
            transform.GetMatrixTransformToWorld(matrix)
            parts["correction"] = digest(";".join(f"{matrix.GetElement(r, c):.3f}" for r in range(3) for c in range(4)))
        parts["reduction pins"] = digest(markups_text(get("Doggybone guide pins" + tag + " (reduction)")))
        if mode == "saw":
            parts["saw pins"] = digest(markups_text(get("Doggybone guide pins" + tag)))
            parts["saw blocks"] = digest("|".join(markups_text(get(f"Doggybone saw box{tag} {label}"))
                                                  for label in ("proximal", "distal")))
        else:
            del parts["planes"]          # the reduction guide follows the correction and its pins
        parts["footprint"] = digest(model_text(get("Doggybone guide footprint" + tag + suffix)))
        parts["bone surface"] = digest(model_text(get("Doggybone guide bone surface" + tag + suffix)))
        return parts

    def _guide_changed_parts(self, node, side, bone, mode):
        """Names of the inputs that changed since this guide body was built ([] = up to date),
        or None if the guide has no stored fingerprint (built with an older version)."""
        stored = node.GetAttribute("Doggybone.PlanSignature")
        if not stored or side is None:
            return None
        try:
            stored = json.loads(stored)
        except ValueError:
            return None
        current = self._guide_plan_signature(side, bone, mode)
        return [self.PLAN_PART_NAMES.get(part, part) for part in current if stored.get(part) != current[part]]

    @staticmethod
    def _mesh_check(polydata):
        """(number of separate pieces, number of open edges) of a guide mesh. A guide is made of
        several closed shells that OVERLAP (body, saw block, sleeves, label, strut - the print software
        unites them); shells whose boxes touch count as one piece, so only really loose parts count."""
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(polydata)
        connectivity.SetExtractionModeToAllRegions()
        connectivity.ColorRegionsOn()
        connectivity.Update()
        from vtk.util.numpy_support import vtk_to_numpy
        labelled = connectivity.GetOutput()
        regions = connectivity.GetNumberOfExtractedRegions()
        ids = labelled.GetPointData().GetArray("RegionId")
        if ids is not None and regions > 1:
            region_of = vtk_to_numpy(ids)
            points = vtk_to_numpy(labelled.GetPoints().GetData())
            boxes = [(points[region_of == r].min(axis=0) - 0.2, points[region_of == r].max(axis=0) + 0.2)
                     for r in range(regions) if (region_of == r).any()]
            parent = list(range(len(boxes)))

            def root(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if np.all(boxes[i][0] <= boxes[j][1]) and np.all(boxes[j][0] <= boxes[i][1]):
                        parent[root(i)] = root(j)
            regions = len({root(i) for i in range(len(boxes))})
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(polydata)
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        return regions, edges.GetOutput().GetNumberOfCells()

    @staticmethod
    def _write_centred_stl(polydata, path):
        """Binary STL in mm, moved so the centre of the bounding box lies at 0,0,0 (the scene is
        unchanged). Returns (success, bounds)."""
        b = polydata.GetBounds()
        centre = [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]
        shift = vtk.vtkTransform()
        shift.Translate(-centre[0], -centre[1], -centre[2])
        moved = vtk.vtkTransformPolyDataFilter()
        moved.SetInputData(polydata)
        moved.SetTransform(shift)
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(moved.GetOutputPort())
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(triangles.GetOutputPort())
        writer.SetFileName(path)
        writer.SetFileTypeToBinary()
        ok = bool(writer.Write()) and os.path.isfile(path) and os.path.getsize(path) > 84
        return ok, b

    def onExportAllGuides(self):
        """ONE click: every finished guide (femur / tibia, osteotomy and reduction guide) is saved as
        its own binary STL into one folder. Each guide is the last one built for that bone and type.
        File names come from the guide itself: patient name - number - fem/tib - dx/sx - cut/red.stl."""
        guides = self._all_guide_bodies()
        if not guides:
            self.guide_status.setText("No guide yet - press 'Extrude guide body' first.")
            return
        start_dir = getattr(self, "_last_stl_dir", None) or qt.QDir.homePath()
        folder = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Export guides to folder", start_dir)
        if not folder:
            return
        jobs = []
        for node, side, bone, mode in guides:
            side_text = side or self.osteotomy_side.currentText
            file_name = self._export_file_name("fem" if bone == "Femur" else "tib",
                                               "dx" if side_text == "Right" else "sx",
                                               "red" if mode == "reduction" else "cut")
            jobs.append((node, side, side_text, bone, mode, os.path.join(folder, file_name)))
        # side check: the file name and the label on the guide carry dx / sx - compare the side of
        # each guide with the side its landmarks actually form before anything is written
        side_checks = {side_text: self.laterality_check(side_text) for side_text in {job[2] for job in jobs}}
        wrong = []
        for side_text, check in sorted(side_checks.items()):
            if check["status"] == "ok" and check["side"] != side_text:
                wrong.append(f"{side_text} guides: the {side_text.lower()} landmarks form a "
                             f"{check['side'].upper()} limb.")
            elif check["status"] in ("check", "conflict"):
                wrong.append(f"{side_text} guides: " + "; ".join(check["messages"]))
        if wrong and not slicer.util.confirmOkCancelDisplay(
                "SIDE CHECK FAILED - file names and guide labels (dx / sx) may carry the wrong side:\n\n"
                + "\n".join(wrong) + "\n\nExport anyway?", windowTitle="Export guides"):
            return
        existing = [os.path.basename(job[-1]) for job in jobs if os.path.exists(job[-1])]
        if existing and not slicer.util.confirmOkCancelDisplay(
                "These files already exist and will be replaced:\n" + "\n".join(existing),
                windowTitle="Export guides"):
            return
        lines, problems = [], 0
        for node, side, side_text, bone, mode, path in jobs:
            kind = "reduction guide" if mode == "reduction" else "osteotomy guide"
            label = f"{side_text} {bone.lower()} {kind}"
            pieces, open_edges = self._mesh_check(node.GetPolyData())
            ok, b = self._write_centred_stl(node.GetPolyData(), path)
            notes = []
            if not ok:
                notes.append("NOT WRITTEN - check the folder / disk")
            if pieces != 1:
                notes.append(f"{pieces} separate pieces")
            if open_edges:
                notes.append(f"{open_edges} open edges - press 'Extrude' again and export again")
            if side is None:
                notes.append("side taken from the current selection (guide built before sides were stored)")
            check = side_checks.get(side_text, {})
            if check.get("status") != "ok" or check.get("side") != side_text:
                notes.append("SIDE NOT CONFIRMED by the landmarks"
                             + (" - " + "; ".join(check["messages"]) if check.get("messages") else ""))
            changed = self._guide_changed_parts(node, side, bone, mode)
            if changed:
                notes.append("OUTDATED - changed since this guide was built: " + ", ".join(changed)
                             + " - extrude this guide again")
            elif changed is None:
                notes.append("not checked for later plan changes (built with an older version)")
            side_ok = check.get("status") == "ok" and check.get("side") == side_text
            problems += 1 if (not ok or pieces != 1 or open_edges or changed or not side_ok) else 0
            lines.append(f"{'OK' if ok and not notes else 'CHECK'}  {label}: {os.path.basename(path)} "
                         f"({b[1] - b[0]:.0f} x {b[3] - b[2]:.0f} x {b[5] - b[4]:.0f} mm)"
                         + (" - " + "; ".join(notes) if notes else ""))
        self._last_stl_dir = folder
        self.guide_status.setStyleSheet(STATUS_WARN_STYLE if problems else STATUS_OK_STYLE)
        self.guide_status.setText(f"{len(jobs)} guide(s) exported to {folder} (each centred at 0,0,0):\n"
                                  + "\n".join(lines))
        if problems:
            slicer.util.warningDisplay(
                "Check the exported guides before printing:\n\n" + "\n".join(l for l in lines if l.startswith("CHECK")),
                windowTitle="Export guides")

    def onExportGuideStl(self):
        """Binary STL of the guide in millimetres, centred at 0,0,0 (for printing only the shape matters). Body and saw blocks are closed shells in ONE file - print preparation
        software (PreForm, Chitubox, PrusaSlicer, ...) unites overlapping shells."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if node is None or node.GetPolyData() is None or node.GetPolyData().GetNumberOfPoints() == 0:
            self.guide_status.setText("No guide body yet - press Extrude first.")
            return
        bone = "fem" if self.osteotomy_bone.currentText == "Femur" else "tib"
        side = "dx" if self.osteotomy_side.currentText == "Right" else "sx"
        kind = "red" if self._guide_mode_is_reduction() else "cut"
        default_name = self._export_file_name(bone, side, kind)    # name-number-fem/tib-dx/sx-red/cut.stl
        start_dir = getattr(self, "_last_stl_dir", None) or qt.QDir.homePath()
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Export cutting guide",
                                              os.path.join(start_dir, default_name), "STL (*.stl)")
        if not path:
            return
        if not path.lower().endswith(".stl"):
            path += ".stl"
        # the guide is moved so the centre of its bounding box lies at 0,0,0 (the scene itself is unchanged)
        b = node.GetPolyData().GetBounds()
        centre = [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]
        shift = vtk.vtkTransform()
        shift.Translate(-centre[0], -centre[1], -centre[2])
        moved = vtk.vtkTransformPolyDataFilter()
        moved.SetInputData(node.GetPolyData())
        moved.SetTransform(shift)
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(moved.GetOutputPort())
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(triangles.GetOutputPort())
        writer.SetFileName(path)
        writer.SetFileTypeToBinary()
        writer.Write()
        self._last_stl_dir = os.path.dirname(path)
        self.guide_status.setText(
            f"Guide exported: {path}\nCentred at 0,0,0 (size {b[1] - b[0]:.1f} x {b[3] - b[2]:.1f} x "
            f"{b[5] - b[4]:.1f} mm; moved by {-centre[0]:.1f}, {-centre[1]:.1f}, {-centre[2]:.1f} mm).")

    def _on_guide_bone_switched(self):
        """Osteotomy bone changed (femur <-> tibia): the other bone's guides, pins and correction preview
        are hidden (kept, never deleted), the original segmentation is shown, this bone's guide objects
        (if any) come back. Guide objects are stored per bone, so planning the tibia never touches the
        finished femur guides."""
        new = self.osteotomy_bone.currentText
        previous = getattr(self, "_guide_bone", "Femur")
        if new == previous:
            return
        for name in ("_stop_guide_edit", "_restore_guide_hidden", "_remove_pin_gizmo", "_remove_pin_preview",
                     "_remove_pin_picker_selection"):
            method = getattr(self, name, None)
            if method is not None:
                try:
                    method()
                except Exception:
                    logging.exception("Doggybone: bone switch step failed (continuing)")
        self._selected_pin_id = None
        self._pin_rows_active = False                 # the rows / handles of the other bone are not ours
        self._pin_row_frames = None
        mode = getattr(self, "_guide_mode", "saw")
        side = self.osteotomy_side.currentText.lower()
        # every guide object of both bones off, then this bone's objects of the current mode on
        for node in slicer.util.getNodesByClass("vtkMRMLDisplayableNode"):
            name = node.GetName() or ""
            if name.startswith(("Doggybone guide", "Doggybone saw guide pin preview", "Doggybone saw box",
                                "Doggybone pin ")) and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(False)
        for name in self._guide_nodes_of_mode(mode):
            if name.startswith("Doggybone guide bone surface"):
                continue
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(True)
        # correction preview of the previous bone off (fragments, combined bone, wedge, fine-shift gizmo)
        for name in list(self._preview_node_names(side, previous)) + list(self._fine_shift_node_names(side, previous)):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(False)
        self._guide_bone = new
        self._ensure_segmentation_visible()
        if mode == "saw":
            try:
                self._show_saw_pin_preview()
            except Exception:
                logging.exception("Doggybone: saw pin preview after bone switch failed (continuing)")
        self._update_guide_tool_visibility()
        if hasattr(self, "guide_status"):
            self.guide_status.setStyleSheet("")
            body = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
            self.guide_status.setText(
                f"{new}: " + ("its guide is shown." if body is not None else
                              "no guide yet - the femur guides are kept (hidden)." if new == "Tibia" else
                              "no guide yet - the tibia guides are kept (hidden)."))

    def _guide_nodes_of_mode(self, mode):
        saved = getattr(self, "_guide_mode", "saw")
        self._guide_mode = mode
        try:
            return [self.GUIDE_FOOTPRINT_NAME, self.GUIDE_BODY_NAME,
                    self.GUIDE_PIN_AXES_NAME, self.GUIDE_PINS_NAME] + (
                        [self.PIN_ROW_HANDLES, self.PIN_GIZMO] if mode == "reduction"
                        else [self._saw_box_gizmo_name("proximal"), self._saw_box_gizmo_name("distal")])
        finally:
            self._guide_mode = saved

    def _show_landmark_set(self, side, corrected):
        """Saw guide: measured (preoperative) landmarks, spheres, axes and labels; reduction guide:
        the ones in the corrected position. The other set is hidden, never deleted."""
        prefix = f"Doggybone {side} "
        corrected_prefix = self.CORRECTED_PREFIX.format(side=side)
        panel = self.left_panel if side == "left" else self.right_panel
        measured_ids = {getattr(panel, "markup_node_id", None)}
        for class_name in ("vtkMRMLModelNode", "vtkMRMLMarkupsFiducialNode"):
            for node in slicer.util.getNodesByClass(class_name):
                name = node.GetName()
                display = node.GetDisplayNode()
                if display is None:
                    continue
                if name.startswith(corrected_prefix):
                    display.SetVisibility(corrected)
                elif node.GetID() in measured_ids or (
                        name.startswith(prefix) and (name.endswith(" sphere") or name.endswith(" axis"))):
                    if corrected:
                        saved = self.__dict__.setdefault("_hidden_original_overlay", {}).setdefault(side, {})
                        saved.setdefault(node.GetID(), bool(display.GetVisibility()))
                    display.SetVisibility(not corrected)
        self._set_sphere_edit_visibility(not corrected)
        # saw guide: the original set is shown again, measurement updates may show it from now on
        if not corrected:
            self.__dict__.setdefault("_hidden_original_overlay", {}).pop(side, None)

    # --- patient + guide label ----------------------------------------------------------------
    PATIENT_NODE_NAME = "Doggybone patient"

    def _patient_node(self, create=False):
        node = slicer.mrmlScene.GetFirstNodeByName(self.PATIENT_NODE_NAME)
        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", self.PATIENT_NODE_NAME)
        return node

    def _load_patient(self):
        node = self._patient_node()
        if node is not None:
            self.patient_name.setText(node.GetParameter("name") or "")
            self.patient_number.setText(node.GetParameter("number") or "")

    def _store_patient(self):
        node = self._patient_node(create=True)
        node.SetParameter("name", self.patient_name.text)
        node.SetParameter("number", self.patient_number.text)

    LABEL_MODES = ("Break-off tag", "On the guide", "No label")       # internal names, same order as the box

    def _label_mode(self):
        box = getattr(self, "guide_label_mode", None)
        if box is None or not 0 <= box.currentIndex < len(self.LABEL_MODES):
            return "Break-off tag"
        return self.LABEL_MODES[box.currentIndex]

    def _label_wanted(self):
        return self._label_mode() != "No label"

    def _label_tag(self, guide, up, bone_obb=None):
        """Break-off tag at the proximal end of the guide. Returns (polydata or None, note)."""
        number = self.patient_number.text.strip() if getattr(self, "patient_number", None) else ""
        text = self._label_text().rstrip(" >").strip()
        rest = text[len(number):].strip() if number and text.startswith(number) else text
        lines = [number, rest] if number else [rest]
        proximal = self._proximal_direction()
        if proximal is None:
            return None, "Label tag skipped: no anatomical frame."
        tag, info = guide_logic.label_tag(guide, up, proximal, lines, bone_obb=bone_obb)
        if tag is None:
            return None, "Label tag skipped: " + info.get("error", "no text")
        note = (f"Break-off label tag {info['length']:.0f} x {info['width']:.0f} x {info['thickness']:.1f} mm at the "
                f"proximal end ('{' / '.join(l for l in lines if l)}' + arrow to proximal), thin neck - break off "
                "with pliers.")
        clearance = info.get("bone_clearance")
        if clearance is not None and clearance < 1.0:
            note += f" Warning: the tag is only {clearance:.1f} mm above the bone - check it."
        return tag, note

    def _label_text(self):
        number = self.patient_number.text.strip() if getattr(self, "patient_number", None) else ""
        bone = {"Femur": "FEM", "Tibia": "TIB"}.get(self.osteotomy_bone.currentText, self.osteotomy_bone.currentText[:3].upper())
        side = self.osteotomy_side.currentText[:1].upper()
        return " ".join(part for part in (number, bone, side) if part) + " >"

    def _proximal_direction(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        _d, shaft, _across, error = self._guide_direction_frame(side, bone)
        return None if error else -guide_logic.unit(np.asarray(shaft, float))   # shaft runs proximal -> distal

    @staticmethod
    def _top_region(shell, up, mask_fn=None, min_dot=0.8):
        from vtk.util.numpy_support import vtk_to_numpy
        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputData(shell)
        nrm.ConsistencyOn()
        nrm.AutoOrientNormalsOn()
        nrm.SplittingOff()
        nrm.Update()
        out = nrm.GetOutput()
        P = vtk_to_numpy(out.GetPoints().GetData()).astype(float)
        N = vtk_to_numpy(out.GetPointData().GetNormals()).astype(float)
        keep = N @ np.asarray(up, float) > min_dot
        if mask_fn is not None:
            keep &= mask_fn(P)
        return P[keep]

    def _label_on(self, guide, up, directions):
        """Raised label on a FREE flat spot of the finished guide: text (no., bone, side) plus an arrow
        that always points proximally. Letter height: the largest of 3 / 2.5 / 2 / 1.6 / 1.3 mm that
        fits. Returns (polydata or None, note)."""
        text = self._label_text().rstrip(" >").strip()
        if not text:
            return None, ""
        units = guide_logic.label_units(text) + 1.4          # + gap + arrow
        spot = guide_logic.find_label_spot(guide, up, directions, units)
        if spot is None:
            return None, "Label skipped: no free flat area on the guide."
        centre, along, height = spot
        up = guide_logic.unit(np.asarray(up, float))
        obb = vtk.vtkOBBTree()
        obb.SetDataSet(guide)
        obb.BuildLocator()
        text_units = guide_logic.label_units(text)
        text_centre = centre - 0.5 * 1.4 * height * along
        letters, _h = guide_logic.emboss_label(obb, text, text_centre, along, up, height=height, raised=0.6)
        proximal = self._proximal_direction()
        arrow_dir = along if proximal is None else proximal - float(np.dot(proximal, up)) * up
        if np.linalg.norm(arrow_dir) < 1e-6:
            arrow_dir = along
        arrow_centre = text_centre + (0.5 * text_units + 0.9) * height * along
        arrow, _h = guide_logic.emboss_label(obb, ">", arrow_centre, arrow_dir, up, height=height, raised=0.6)
        append = vtk.vtkAppendPolyData()
        for part in (letters, arrow):
            if part is not None:
                append.AddInputData(part)
        if append.GetNumberOfInputConnections(0) == 0:
            return None, "Label skipped: no room for the text."
        append.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(append.GetOutput())
        return out, f"Label '{text}' + arrow to proximal ({height:.1f} mm letters, 0.6 mm raised) on a free flat area."

    def _saw_guide_label(self, parts):
        """Free flat spot anywhere on the saw guide (body or saw block), seen from the guide side."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        d, shaft, across, error = self._guide_direction_frame(side, bone)
        if error:
            return None, "Label skipped: no guide frame."
        append = vtk.vtkAppendPolyData()
        for part in parts:
            append.AddInputData(part)
        append.Update()
        proximal = self._proximal_direction()
        return self._label_on(append.GetOutput(), d, [proximal, across])

    def _reduction_guide_label(self, body, axis):
        """Free flat spot on the top of the pin bar, preferably along the bar."""
        axis = guide_logic.unit(np.asarray(axis, float))
        region = self._top_region(body, axis, min_dot=0.95)
        directions = []
        if len(region) >= 10:
            rel = region - region.mean(axis=0)
            rel -= np.outer(rel @ axis, axis)
            _u, _s, vt = np.linalg.svd(rel, full_matrices=False)
            directions.append(vt[0])
        proximal = self._proximal_direction()
        if proximal is not None:
            directions.append(proximal)
        return self._label_on(body, axis, directions or [np.cross(axis, [1.0, 0.0, 0.0])])

    PAINT_TOOLS = ("brush_add", "brush_remove", "lasso_add", "lasso_remove")

    def _place_diameter_field(self):
        """The one pin-diameter field: under 'Prox. / Dist.' in reduction mode, top left of the guide
        parameters in saw mode."""
        pair = getattr(self, "_guide_field_labels", {}).get("Pin size")
        grid = self.reduction_options_widget.layout() if getattr(self, "reduction_options_widget", None) else None
        if pair is None or grid is None or getattr(self, "_reduction_diam_label", None) is None:
            return
        param_label, field = pair
        if getattr(self, "_reduction_row_height", None):
            field.setMinimumHeight(self._reduction_row_height)
        if self._guide_mode_is_reduction():
            grid.addWidget(field, 3, 1)
            param_label.setVisible(False)
            self._reduction_diam_label.setVisible(True)
        else:
            param_grid = param_label.parentWidget().layout()
            param_grid.addWidget(field, 0, 1)
            param_label.setVisible(True)
            self._reduction_diam_label.setVisible(False)
        field.setVisible(True)

    def _update_guide_tool_visibility(self):
        """Pin bar (reduction guide): only what it needs - pin rows, Pins / Pins delete / Delete, pin
        diameter, extrude and export. Painting tools and the footprint / saw-block parameters are
        hidden (their values stay in use). Everything is shown for a painted footprint / saw guide."""
        buttons = getattr(self, "_guide_tool_buttons", None)
        if not buttons:
            return
        full = not (self._guide_mode_is_reduction() and self._reduction_is_bar())
        if not full and getattr(self, "_guide_active_tool", None) in self.PAINT_TOOLS:
            self._stop_guide_edit()
        for key in self.PAINT_TOOLS:
            if key in buttons:
                buttons[key].setVisible(full)
        if getattr(self, "_brush_radius_widget", None) is not None:
            self._brush_radius_widget.setVisible(full)
        if getattr(self, "guide_delete_button", None) is not None:
            # pin bar: 'Delete pins' next to 'Place pins' is the one delete button
            self.guide_delete_button.setVisible(full)
        reduction = self._guide_mode_is_reduction()
        if "pins" in buttons:
            buttons["pins"].setVisible(False)                 # internal (manual placement); pins are clicked
        if getattr(self, "guide_pin_delete_selected_button", None) is not None:
            self.guide_pin_delete_selected_button.setVisible(reduction)
        if getattr(self, "guide_pins_delete_button", None) is not None:
            # footprint variant: remove only the pins, keep the painted footprint (pin bar: 'Delete')
            self.guide_pins_delete_button.setVisible(reduction and full)
        for label, (text, field) in getattr(self, "_guide_field_labels", {}).items():
            if label == "Pin size":
                continue
            text.setVisible(full)
            field.setVisible(full)
        self._place_diameter_field()
        # bar height / web width: defaults are used, not shown for the pin bar
        grid = self.reduction_options_widget.layout() if getattr(self, "reduction_options_widget", None) else None
        if grid is not None:
            for index in range(grid.count()):
                widget = grid.itemAt(index).widget()
                if widget is None:
                    continue
                if widget in (self.reduction_standoff, self.reduction_bar_diameter) or (
                        isinstance(widget, qt.QLabel) and widget.text in ("Height", "Width")):
                    widget.setVisible(full)
        info = getattr(self, "_guide_info_label", None)
        if info is not None:
            info.setText(
                "Reduction guide: place the pins on the corrected bone ('Place pins', click a pin to move "
                "it), then 'Extrude'. The saw guide takes these pins over."
                if self._guide_mode_is_reduction() else
                "Footprint = coloured area on the bone surface where the guide sits. Paint it by hand "
                "with Brush + / Lasso + anywhere on the outer bone surface around the osteotomy (rotate "
                "the view as needed), then 'Extrude'.")
        qt.QTimer.singleShot(0, self._relayout_guide_panel)

    def _relayout_guide_panel(self):
        """After widgets were shown / hidden: lay the guide panel out again at once (otherwise the
        rows keep their old, too small height until the panel is resized - fields overlap)."""
        box = getattr(self, "_guide_collapsible", None)
        if box is None or box.layout() is None:
            return
        for widget in (getattr(self, "reduction_options_widget", None), box):
            if widget is not None:
                widget.updateGeometry()
        box.layout().invalidate()
        box.layout().activate()

    def onGuideModeChanged(self):
        mode = "reduction" if self.guide_mode_reduction.checked else "saw"
        if mode == getattr(self, "_guide_mode", "saw"):
            return
        self._stop_guide_edit()
        self._restore_guide_hidden()
        self._guide_mode = mode
        if getattr(self, "reduction_options_widget", None) is not None:
            self.reduction_options_widget.setVisible(mode == "reduction")
        self._update_guide_tool_visibility()
        other = "saw" if mode == "reduction" else "reduction"
        for visible, names in ((False, self._guide_nodes_of_mode(other)), (True, self._guide_nodes_of_mode(mode))):
            for name in names:
                node = slicer.mrmlScene.GetFirstNodeByName(name)
                if node is not None and node.GetDisplayNode() is not None:
                    node.GetDisplayNode().SetVisibility(visible)
        pins_node = slicer.mrmlScene.GetFirstNodeByName(self.SAW_PINS_NAME)
        if pins_node is not None and pins_node.GetDisplayNode() is not None:
            pins_node.GetDisplayNode().SetVisibility(mode == "saw")
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        self._show_landmark_set(side, corrected=(mode == "reduction"))
        combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
        segmentation = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if combined is not None and combined.GetDisplayNode() is not None:
            combined.GetDisplayNode().SetVisibility(mode == "reduction")
        # saw guide: the bone as it is - the reduced (corrected) fragments and the fine-shift gizmo are hidden
        preview_names = self._preview_node_names(side, bone)
        for name in (preview_names[0], preview_names[1], self._fine_shift_node_names(side, bone)[1]):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(mode == "reduction")
        if segmentation is not None and segmentation.GetDisplayNode() is not None:
            # the segmentation always stays in the slice views; in 3D it gives way to the corrected
            # bone only when that bone really exists (otherwise the 3D view would be empty)
            seg_display = segmentation.GetDisplayNode()
            seg_display.SetVisibility(True)
            has_combined = (combined is not None and combined.GetPolyData() is not None
                            and combined.GetPolyData().GetNumberOfPoints() > 0)
            show_3d = mode == "saw" or not has_combined
            if hasattr(seg_display, "SetVisibility3D"):
                seg_display.SetVisibility3D(show_3d)
            if hasattr(seg_display, "SetVisibility2D"):
                seg_display.SetVisibility2D(True)
        if self._compare_mode():
            self._apply_compare_views()               # compare mode: the original stays in the left 3D view
        note = self._show_saw_pin_preview()          # hidden in the reduction mode
        if mode == "saw":
            self.guide_status.setText("Saw guide: paint / edit the cutting guide on the bone as it is."
                                      + (f"\n{note}" if note else ""))
            return
        if self._ensure_current_reduction_surface():
            return
        problems = []
        marks = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        segments = set()
        if marks is not None:
            for i in range(marks.GetNumberOfControlPoints()):
                segments.add((marks.GetNthControlPointDescription(i) or "").split("|")[0])
        if not {"proximal", "distal"} <= segments:
            problems.append("no pins in both fragments yet ('Place pins')")
        if combined is None:
            problems.append("no corrected bone (run the correction preview)")
        self.guide_status.setText(
            "Reduction guide: 'Place pins' on the corrected bone (click a pin to move it), then "
            "'Extrude'. The saw guide takes these pins over, pre-angled."
            + (("\nMissing: " + "; ".join(problems)) if problems else ""))

    def _guide_source_polydata(self):
        """Bone surface the guide is prepared on: the bone as it is (saw guide) or both fragments
        in the corrected position (reduction guide; kept per fragment for the ROI extraction)."""
        if not self._guide_mode_is_reduction():
            return self._get_bone_polydata()
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        fragments = []
        self._reduction_fragments_raw = []        # (as-scanned polydata, 4x4 to corrected pose or None)
        for name in names[:2]:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None or node.GetPolyData() is None or node.GetPolyData().GetNumberOfPoints() == 0:
                return None
            polydata = node.GetPolyData()
            parent = node.GetParentTransformNode()
            raw_matrix = None
            if parent is not None:
                raw_matrix = vtk.vtkMatrix4x4()
                parent.GetMatrixTransformToWorld(raw_matrix)
            self._reduction_fragments_raw.append((polydata, raw_matrix))
            if parent is not None:
                matrix = vtk.vtkMatrix4x4()
                parent.GetMatrixTransformToWorld(matrix)
                transform = vtk.vtkTransform()
                transform.SetMatrix(matrix)
                moved = vtk.vtkTransformPolyDataFilter()
                moved.SetInputData(polydata)
                moved.SetTransform(transform)
                moved.Update()
                polydata = moved.GetOutput()
            fragments.append(polydata)
        self._reduction_fragments = fragments
        append = vtk.vtkAppendPolyData()
        for fragment in fragments:
            append.AddInputData(fragment)
        append.Update()
        return append.GetOutput()

    def _guide_mode_plane_origins(self, plane_nodes, origins):
        """Reduction guide: the moved fragment's cutting plane goes with it."""
        if not self._guide_mode_is_reduction():
            return origins
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        transform = slicer.mrmlScene.GetFirstNodeByName(names[2])
        if transform is None:
            return origins
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)
        M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        moved = None
        for segment, name in (("proximal", names[0]), ("distal", names[1])):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetParentTransformNode() is not None \
                    and node.GetParentTransformNode().GetID() == transform.GetID():
                moved = segment
        plane_name = {"proximal": self._plane_node_name(side, bone),
                      "distal": self._distal_plane_node_name(side, bone)}.get(moved)
        out = []
        for node, origin in zip(plane_nodes, origins):
            if plane_name is not None and node.GetName() == plane_name:
                origin = M[:3, :3] @ origin + M[:3, 3]
            out.append(origin)
        return out

    @staticmethod
    def _orient_part_outward(part, centre, shaft):
        """Point normals of ONE fragment's surface outward (away from the shaft axis). The fragments
        come from the preview with different triangle orientation; the later global orientation
        (orient_and_suggest) can only fix both if each is consistent on its own."""
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        normals = part.GetPointData().GetNormals()
        if normals is None:
            nrm = vtk.vtkPolyDataNormals()
            nrm.SetInputData(part)
            nrm.SplittingOff()
            nrm.ConsistencyOn()
            nrm.Update()
            part.DeepCopy(nrm.GetOutput())
            normals = part.GetPointData().GetNormals()
        P = vtk_to_numpy(part.GetPoints().GetData()).astype(float)
        N = vtk_to_numpy(normals).astype(float)
        shaft = np.asarray(shaft, dtype=float)
        rel = P - np.asarray(centre, dtype=float)
        radial = rel - np.outer(rel @ shaft, shaft)
        side_wall = np.abs(N @ shaft) < 0.7                 # ignore the cut face
        if np.any(side_wall) and float(np.mean(np.sum(N[side_wall] * radial[side_wall], axis=1))) < 0.0:
            N = -N
        # same name and type for every fragment - otherwise vtkAppendPolyData drops the normals
        array = numpy_to_vtk(np.ascontiguousarray(N.astype(np.float32)), deep=True, array_type=vtk.VTK_FLOAT)
        array.SetName("Normals")
        part.GetPointData().SetNormals(array)

    def _extract_guide_roi(self, bone_polydata, centre, shaft, across, d, t_range, u_half, v_half):
        if not self._guide_mode_is_reduction():
            return guide_logic.extract_roi_mesh(bone_polydata, centre, shaft, across, d, t_range, u_half,
                                                v_half, closest_point=centre)
        append = vtk.vtkAppendPolyData()
        centre = np.asarray(centre, dtype=float)
        for fragment, matrix in getattr(self, "_reduction_fragments_raw", []):
            # Cut the ROI out of the fragment in ITS OWN as-scanned pose, with the frame taken back
            # into that pose - a moved (tilted / derotated) fragment is otherwise cut obliquely by
            # the box and only part of its outer surface survives. Then move it to the corrected pose.
            if matrix is None:
                M = np.eye(4)
            else:
                M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
            Minv = np.linalg.inv(M)
            R_back = Minv[:3, :3]
            c_raw = (Minv @ np.append(centre, 1.0))[:3]
            s_raw, a_raw, d_raw = (guide_logic.unit(R_back @ np.asarray(v, dtype=float)) for v in (shaft, across, d))
            part = guide_logic.extract_roi_mesh(fragment, c_raw, s_raw, a_raw, d_raw, t_range, u_half, v_half,
                                                closest_point=c_raw)
            if not part.GetNumberOfPoints():
                continue
            self._orient_part_outward(part, c_raw, s_raw)
            if matrix is not None:
                transform = vtk.vtkTransform()
                transform.SetMatrix(matrix)
                moved = vtk.vtkTransformPolyDataFilter()
                moved.SetInputData(part)
                moved.SetTransform(transform)
                moved.Update()
                part = vtk.vtkPolyData()
                part.DeepCopy(moved.GetOutput())
                normals = part.GetPointData().GetNormals()
                if normals is not None and normals.GetDataType() != vtk.VTK_FLOAT:
                    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
                    array = numpy_to_vtk(np.ascontiguousarray(vtk_to_numpy(normals).astype(np.float32)),
                                         deep=True, array_type=vtk.VTK_FLOAT)
                    array.SetName("Normals")
                    part.GetPointData().SetNormals(array)
            append.AddInputData(part)
        if append.GetNumberOfInputConnections(0) == 0:
            return vtk.vtkPolyData()
        append.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(append.GetOutputPort())
        clean.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(clean.GetOutput())
        return out
