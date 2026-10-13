# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""CorrectionMixin: Correction preview: fragment split and transform, predicted angles, bisecting / contact-congruent alignment, fine shift, corrected landmarks.

Part of DoggyboneWidget (see Doggybone.py). Methods were moved here unchanged; they rely on
attributes created in DoggyboneWidget.setup() and on methods of the other mixins.
"""

from DoggyboneUI.common import *


class CorrectionMixin:

    def _get_measurement_by_name(self, panel, name):
        for i in range(panel.measurement_stack.count):
            widget = panel.measurement_stack.widget(i)
            if widget.measurement.name == name:
                return widget.measurement
        return None

    def _moving_side_sign(self, cut_o, cut_n, moving_is_distal, side, bone):
        """+1 / -1: on which side of the cut plane (normal cut_n) the MOVING fragment lies. Same rule as
        the fragment split and the landmark classification: the shaft axis points proximal -> distal,
        so the distal side is where (p - o).n has the sign of n.shaft."""
        shaft, _ml, _ap, frame_error = self._get_anatomical_frame(side, bone)
        if not frame_error and shaft is not None:
            n = np.asarray(cut_n, float)
            dot = float(np.dot(n, np.asarray(shaft, float)))
            if abs(dot) > 1e-3:
                distal_sign = 1.0 if dot > 0 else -1.0
                return distal_sign if moving_is_distal else -distal_sign
        # fallback (no anatomical frame): a proximal landmark
        panel = self.left_panel if side == "left" else self.right_panel
        node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id)
        proximal_names = (["femur head", "femur neck", "Femur midpoint 33%"] if bone == "Femur"
                          else ["intercondylar eminence", "proximal tibia midpoint", "condylus medialis tibiae"])
        for name in proximal_names:
            for landmark in panel.landmarks:
                if landmark.name == name and landmark.placed and landmark._id is not None and node is not None:
                    p = np.array(node.GetNthControlPointPositionWorld(landmark._id))
                    proximal_sign = 1.0 if float((p - np.asarray(cut_o, float)) @ np.asarray(cut_n, float)) > 0 else -1.0
                    return -proximal_sign if moving_is_distal else proximal_sign
        return None

    def _cwo_wedge_advice(self, side, bone):
        """CWO with two (nearly) identical cutting planes: the wedge to be removed is thinner than about
        two saw blades - the blade alone takes that much bone, the closing would be inexact. Returns a
        warning text recommending an OWO, or ''."""
        if not self.osteotomy_technique.currentText.startswith("CWO"):
            return ""
        proximal = slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone))
        distal = slicer.mrmlScene.GetFirstNodeByName(self._distal_plane_node_name(side, bone))
        bone_polydata = self._get_bone_polydata()
        if proximal is None or distal is None or bone_polydata is None:
            return ""
        try:
            o1, n1 = self._get_plane_origin_normal(proximal)
            o2, n2 = self._get_plane_origin_normal(distal)
            n1 = osteotomy_logic.normalize(np.asarray(n1, float))
            n2 = osteotomy_logic.normalize(np.asarray(n2, float))
            angle = float(np.degrees(np.arccos(np.clip(abs(float(n1 @ n2)), 0.0, 1.0))))
            contour = guide_logic.section_contour(bone_polydata, o1, n1)
            if contour is None or len(contour) < 3:
                return ""
            # wedge thickness across the bone: distance of the proximal cut's outline from the distal plane
            thickness = np.abs((np.asarray(contour, float) - np.asarray(o2, float)) @ n2)
            widest = float(thickness.max())
        except Exception:
            logging.exception("Doggybone: CWO wedge check failed (continuing)")
            return ""
        try:
            blade = float(self.guide_blade.currentText.split()[0])
        except Exception:
            blade = 1.0
        limit = max(1.5, 2.0 * blade)
        if widest >= limit:
            return ""
        return (f"WARNING: the two cutting planes are almost identical (angle {angle:.1f}\N{DEGREE SIGN}, wedge at "
                f"most {widest:.1f} mm thick across the bone - less than about two saw blades, {limit:.1f} mm). "
                "The blade alone removes about this much bone, a closing wedge would be inexact - an OWO "
                "(one cut, opened) is recommended.")

    def _owo_hinge_contact(self, matrix, cut_o, cut_n, bone_polydata, moving_is_distal, side, bone):
        """(matrix, note): the moved fragment shifted along the cut normal so that its cut face just
        touches the fixed cut face at its deepest point (the hinge) - no overlap, no floating."""
        try:
            contour = guide_logic.section_contour(bone_polydata, cut_o, cut_n)
            sign = self._moving_side_sign(cut_o, cut_n, moving_is_distal, side, bone)
        except Exception:
            logging.exception("Doggybone: OWO hinge contact check failed (continuing)")
            return matrix, ""
        if contour is None or sign is None or len(contour) < 3:
            return matrix, ""
        o = np.asarray(cut_o, float)
        n = osteotomy_logic.normalize(np.asarray(cut_n, float))
        M = osteotomy_logic.matrix_to_numpy(matrix) if not isinstance(matrix, np.ndarray) else np.asarray(matrix, float)
        moved = (M @ np.c_[contour, np.ones(len(contour))].T).T[:, :3]
        gap = sign * ((moved - o) @ n)                 # > 0: open, < 0: overlap with the fixed fragment
        shift = -float(gap.min())
        if abs(shift) < 0.01:
            return matrix, (f"OWO: cut faces in contact at the hinge, opening up to {gap.max():.2f} mm.")
        T = np.eye(4)
        T[:3, 3] = sign * shift * n
        deg = "\N{DEGREE SIGN}"
        note = (f"OWO hinge contact: fragment moved {abs(shift):.2f} mm along the cut normal "
                + ("(it overlapped the fixed fragment)" if shift > 0 else "(it floated above the cut)")
                + f" - hinge cortex now in contact, opening up to {gap.max() + shift:.2f} mm.")
        return osteotomy_logic.numpy_to_vtk_matrix(T @ M), note

    def _get_bone_polydata(self):
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None:
            return None
        segmentation = segmentation_node.GetSegmentation()
        if segmentation.GetNumberOfSegments() == 0:
            return None
        segment_id = segmentation.GetNthSegmentID(0)
        representation_name = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
        polydata = segmentation.GetSegment(segment_id).GetRepresentation(representation_name)
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return None
        return polydata


    def _preview_node_names(self, side, bone):
        return (f"Doggybone {side} {bone.lower()} proximal fragment",
                f"Doggybone {side} {bone.lower()} distal fragment (corrected)",
                f"Doggybone {side} {bone.lower()} correction transform",
                f"Doggybone {side} {bone.lower()} final corrected bone (combined)",
                f"Doggybone {side} {bone.lower()} wedge REMOVE")

    def onPreviewCorrection(self):
        self.correction_status.setStyleSheet("")
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        is_cwo = self.osteotomy_technique.currentText.startswith("CWO")

        proximal_plane_node = slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone))
        if proximal_plane_node is None:
            self.correction_status.setText(
                "Insert the proximal plane first ('Insert/update proximal plane').")
            return

        distal_plane_node = slicer.mrmlScene.GetFirstNodeByName(self._distal_plane_node_name(side, bone))
        if distal_plane_node is None:
            self.correction_status.setText(
                "Insert the distal plane first ('Insert/update distal plane') - both OWO and "
                "CWO need it to define the wedge angle and hinge.")
            return

        proximal_origin, proximal_normal = self._get_plane_origin_normal(proximal_plane_node)
        distal_origin, distal_normal = self._get_plane_origin_normal(distal_plane_node)

        bone_polydata = self._get_bone_polydata()
        if bone_polydata is None:
            self.correction_status.setText(
                "No segmentation found - run 'Apply' under '3D Segmentation' first.")
            return

        hinge_point, hinge_axis = osteotomy_logic.planes_intersection_line(
            proximal_origin, proximal_normal, distal_origin, distal_normal)
        planes_parallel = hinge_point is None
        if planes_parallel:
            # parallel planes: OWO (one real cut) or a pure derotation - the hinge is not defined by the
            # planes; the centre of the cut stands in for it (OWO: the fragment is then set onto the
            # hinge cortex by the hinge-contact step below)
            master_is_proximal = self.reference_plane_selector.currentText == "Proximal"
            hinge_point = np.asarray(proximal_origin if master_is_proximal else distal_origin, float)

        # The wedge MAGNITUDE always comes from the measured deformity vs. the target value
        # (like the rotation correction below) - not from however far apart the two planes
        # happen to be tilted, which would also pick up any incidental tilt in an unrelated
        # anatomical plane (e.g. a sagittal adjustment made only for practical cutting reasons)
        # as if it were part of the correction. The planes still define the hinge (above) and
        # the correction's DIRECTION (below) - just not its size.
        panel = self.left_panel if side == "left" else self.right_panel
        wedge_measurement_name = self.WEDGE_MEASUREMENT_NAME[bone]
        wedge_measurement = self._get_measurement_by_name(panel, wedge_measurement_name)
        if wedge_measurement is None:
            self.correction_status.setText(f"Measurement '{wedge_measurement_name}' not found.")
            return
        wedge_ready, wedge_value, wedge_value_description = wedge_measurement()
        if not wedge_ready and bone == "Tibia" and hasattr(self, "_tibia_mode") and self._tibia_mode():
            # tibia: the corrections come from the target table (solver) - the varus/valgus landmarks
            # are only needed when a frontal row is checked; without them: no frontal correction
            wedge_ready, wedge_value = True, float(self.target_wedge_value.value)
            wedge_value_description = "not measured"
        if not wedge_ready:
            self.correction_status.setText(
                f"Place the landmarks for '{wedge_measurement_name}' ({side}) first.")
            return
        wedge_magnitude = abs(wedge_value - self.target_wedge_value.value)
        tibia_rows = bone == "Tibia" and hasattr(self, "_tibia_mode") and self._tibia_mode()
        # OWO: only ONE cut - the second plane is a virtual target, parallel or not does not matter (the
        # opening comes from the measured / target values; the hinge contact is set afterwards).
        # CWO: the wedge between the two cuts IS the correction - parallel planes cannot remove one.
        if planes_parallel and is_cwo and wedge_magnitude >= 0.3 and not tibia_rows:
            self.correction_status.setText(
                f"CWO: the two cutting planes are parallel, but a wedge of {wedge_magnitude:.1f}\N{DEGREE SIGN} "
                "has to be removed - press 'Create/update SECOND plane from MASTER' (or tilt the distal plane).")
            return

        # Direction comes from which way the distal plane is actually tilted relative to the
        # proximal one - reusing the plane geometry for direction only is robust even if its
        # magnitude would've been distorted by an unrelated tilt component, since the sign only
        # depends on which side of the proximal plane the distal one leans toward.
        # The two planes determine only which SIDE the wedge opens/closes toward.  They no longer
        # define the correction axis.  Projecting the signed comparison onto the anatomical AP
        # axis prevents an incidental sagittal tilt of either plane from contaminating the
        # varus/valgus correction.
        shaft_axis, ml_axis, ap_axis, frame_error = self._get_anatomical_frame(side, bone)
        if frame_error:
            self.correction_status.setText(frame_error)
            return
        moving_is_distal = self.reference_plane_selector.currentText == "Proximal"
        wedge_delta = self.target_wedge_value.value - wedge_value
        wedge_deg = wedge_delta * (1.0 if moving_is_distal else -1.0)
        ok_rot, rotation_delta, rotation_message = self._calculated_rotation_delta(side, bone)
        tibia_targets = bone == "Tibia" and hasattr(self, "_tibia_mode") and self._tibia_mode()
        if not ok_rot and tibia_targets:
            # tibia: the torsion comes from the target table (solver) - its landmarks are only
            # needed when a torsion row is checked
            ok_rot, rotation_delta = True, 0.0
        if not ok_rot:
            self.correction_status.setText(rotation_message)
            return
        self.rotation_angle.setValue(rotation_delta)
        rotation_deg = rotation_delta if moving_is_distal else -rotation_delta

        # Torsion must rotate about the actual anatomical shaft LINE.  Use the point on that
        # line closest to the real osteotomy plane, not the cortical wedge hinge.  The old code
        # forced torsion through hinge_point and therefore introduced a spurious translation.
        anatomical_origin, shaft_axis_check, axis_error = self._get_anatomical_axis(side, bone)
        if axis_error:
            self.correction_status.setText(axis_error)
            return
        # the MASTER plane is the fixed one and (OWO) the only real cut
        master_origin, master_normal = ((proximal_origin, proximal_normal) if moving_is_distal
                                        else (distal_origin, distal_normal))
        shaft_point_at_cut = osteotomy_logic.closest_point_on_line(
            master_origin, anatomical_origin, shaft_axis_check)

        alignment_report = ""
        if is_cwo and self._cwo_alignment_is_congruent():
            fixed_origin, fixed_normal = ((proximal_origin, proximal_normal) if moving_is_distal
                                          else (distal_origin, distal_normal))
            desired_rotation, _desired_info = self._desired_correction_rotation(
                side, bone, moving_is_distal, fixed_origin, fixed_normal)
            if desired_rotation is None and tibia_rows:
                # tibia: the torsion comes only from the solver - without it the preview would silently
                # drop the torsion
                self.correction_status.setStyleSheet(STATUS_WARN_STYLE)
                self.correction_status.setText(str(_desired_info))
                return
            matrix, alignment_report = self._contact_congruent_matrix(
                proximal_origin, proximal_normal, distal_origin, distal_normal, hinge_point,
                moving_is_distal, shaft_axis_check, anatomical_origin, rotation_deg,
                ap_axis, ml_axis, abs(wedge_delta),
                side=side, bone=bone, desired_rotation=desired_rotation)
        else:
            matrix = osteotomy_logic.anatomical_correction_matrix(
                shaft_point_at_cut, shaft_axis_check, rotation_deg,
                hinge_point, ap_axis, wedge_deg, frontal_first=moving_is_distal)
            # OWO (CWO always takes the contact-congruent branch above): the SAME verified correction
            # as the SECOND-plane derivation (_desired_correction_rotation: signs checked on the
            # landmarks, angles solved, derotation about the moving fragment's own axis) - so the moved
            # cut face and the virtual SECOND plane always coincide. Wedge about the hinge, torsion
            # about the true shaft line.
            _R, info = self._desired_correction_rotation(side, bone, moving_is_distal, master_origin, master_normal)
            if not isinstance(info, dict):
                self.correction_status.setStyleSheet(STATUS_WARN_STYLE)
                self.correction_status.setText(str(info))
                return
            wedge_deg = info["frontal_signed"]
            rotation_deg = info["torsion_signed"]
            matrix = osteotomy_logic.anatomical_correction_matrix(
                shaft_point_at_cut, shaft_axis_check, rotation_deg, hinge_point, ap_axis, wedge_deg,
                frontal_first=moving_is_distal, sagittal_axis=ml_axis,
                sagittal_angle_deg=info.get("sagittal_signed", 0.0))
            # the values really applied (solver), not the raw measurement-vs-target estimate
            wedge_magnitude = abs(float(wedge_deg))
            self.rotation_angle.setValue(abs(float(rotation_deg)))
            alignment_report = (f"Correction verified on the landmarks: wedge {abs(wedge_deg):.1f}\N{DEGREE SIGN}"
                                + (f", sagittal {abs(info.get('sagittal_signed', 0.0)):.1f}\N{DEGREE SIGN}"
                                   if abs(info.get("sagittal_signed", 0.0)) >= 0.05 else "")
                                + f", torsion {abs(rotation_deg):.1f}\N{DEGREE SIGN}"
                                + (": predicted " + self._format_prediction(bone, info["predicted"])
                                   if info.get("predicted") else "") + "."
                                + (("\n" + info["warning"]) if info.get("warning") else ""))

        if not is_cwo:
            # OWO: the moved cut face must rest on the fixed one AT THE HINGE - no overlap of the
            # fragments. A torsion about the shaft line (not perpendicular to an oblique cut) tilts
            # the face slightly, and a wedge about a hinge inside the bone would push one side into
            # the fixed fragment: the fragment is moved along the cut normal until the deepest point
            # just touches (opening = wedge gap, hinge cortex in contact).
            fixed_o, fixed_n = ((proximal_origin, proximal_normal) if moving_is_distal
                                else (distal_origin, distal_normal))
            matrix, hinge_report = self._owo_hinge_contact(matrix, fixed_o, fixed_n, bone_polydata,
                                                           moving_is_distal, side, bone)
            if hinge_report:
                alignment_report = (alignment_report + "\n" if alignment_report else "") + hinge_report

        # Reduction gap: the faces are now in contact (OWO: at the hinge, CWO: congruent). A small,
        # adjustable gap avoids pre-load / tension when the fragments are reduced on the guide -
        # the moved fragment is translated along the fixed cut normal, away from the fixed one.
        # The reduction-guide pins sit on this pose, so their fragment rows end up exactly this
        # much farther apart than in the cutting guide.
        gap_mm = float(self.reduction_gap.value) if hasattr(self, "reduction_gap") else 0.0
        if gap_mm > 0.0:
            fixed_o, fixed_n = ((proximal_origin, proximal_normal) if moving_is_distal
                                else (distal_origin, distal_normal))
            sign = self._moving_side_sign(fixed_o, fixed_n, moving_is_distal, side, bone)
            if sign is not None:
                n_fix = osteotomy_logic.normalize(np.asarray(fixed_n, float))
                M = (osteotomy_logic.matrix_to_numpy(matrix) if not isinstance(matrix, np.ndarray)
                     else np.asarray(matrix, float))
                T = np.eye(4)
                T[:3, 3] = sign * gap_mm * n_fix
                matrix = osteotomy_logic.numpy_to_vtk_matrix(T @ M)
                alignment_report = ((alignment_report + "\n" if alignment_report else "")
                                    + f"Reduction gap: {gap_mm:.2f} mm between the cut faces "
                                      "(reduction-guide pins correspondingly farther apart).")
            else:
                alignment_report = ((alignment_report + "\n" if alignment_report else "")
                                    + "Reduction gap not applied: proximal landmark missing.")

        # Optional clinical fine adjustment: translate the already-corrected fragment only
        # within the MASTER cutting plane.  X/Y are taken from the actual Markups plane frame,
        # so this is independent of the shaft/world axes.  No normal (Z) translation and no
        # additional rotation are introduced.
        # The shift itself lives in a separate, lightweight PARENT transform ("fine shift (X/Y)")
        # so it can be changed live - via the X/Y fields or the X/Y gizmo - without re-clipping
        # the bone.  world = T_shift @ matrix: the shift is applied in world coordinates after
        # the planned angular correction, along the fixed MASTER plane's local X/Y directions.
        master_plane_node = (proximal_plane_node if self.reference_plane_selector.currentText == "Proximal"
                             else distal_plane_node)
        master_origin, master_x, master_y, master_normal = self._get_plane_frame(master_plane_node, side, bone)

        wedge_polydata = None
        if is_cwo:
            # Explicit three-region split.  The removed wedge is ONLY the bone between
            # the proximal and distal cutting planes.  Plane-normal signs cannot invert
            # this because clip_fragments classifies each half anatomically using shaft_axis.
            proximal_polydata, wedge_polydata, distal_polydata = \
                osteotomy_logic.split_bone_three_regions(
                    bone_polydata,
                    proximal_origin, proximal_normal,
                    distal_origin, distal_normal,
                    shaft_axis)
            technique_note = ""
        else:
            # OWO: one real cut; the second plane remains a virtual target.
            # the one real cut is the MASTER plane (proximal or distal, whichever is selected)
            proximal_polydata, distal_polydata = osteotomy_logic.clip_fragments(
                bone_polydata, master_origin, master_normal, shaft_axis)
            technique_note = (" (OWO - one cut at the MASTER plane; the "
                              + ("distal" if moving_is_distal else "proximal")
                              + " plane is a virtual target only, no bone removed)")

        wedge_description = (
            f"{wedge_magnitude:.1f}\N{DEGREE SIGN} ('{wedge_measurement_name}': "
            f"{wedge_value:.1f}\N{DEGREE SIGN} vs. target {self.target_wedge_value.value:.1f}"
            f"\N{DEGREE SIGN}; direction from the two planes, reference: "
            f"{self.reference_plane_selector.currentText}){technique_note}")

        proximal_name, distal_name, transform_name, combined_name, wedge_name = self._preview_node_names(side, bone)

        proximal_node = slicer.mrmlScene.GetFirstNodeByName(proximal_name)
        if proximal_node is None:
            proximal_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", proximal_name)
            proximal_node.CreateDefaultDisplayNodes()
            proximal_node.GetDisplayNode().SetColor(0.75, 0.75, 0.2)
            proximal_node.GetDisplayNode().SetOpacity(0.9)

        distal_node = slicer.mrmlScene.GetFirstNodeByName(distal_name)
        if distal_node is None:
            distal_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", distal_name)
            distal_node.CreateDefaultDisplayNodes()
            distal_node.GetDisplayNode().SetColor(0.2, 0.75, 0.85)
            distal_node.GetDisplayNode().SetOpacity(0.9)

        # Keep the resected CWO wedge visible at its ORIGINAL location for validation.
        # Red + semi-transparent makes it immediately obvious whether the Boolean split
        # selected exactly the bone between the two cutting planes.
        wedge_node = slicer.mrmlScene.GetFirstNodeByName(wedge_name)
        if is_cwo and wedge_polydata is not None:
            if wedge_node is None:
                wedge_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", wedge_name)
                wedge_node.CreateDefaultDisplayNodes()
            wedge_node.SetAndObservePolyData(wedge_polydata)
            wedge_node.GetDisplayNode().SetColor(1.0, 0.1, 0.1)
            wedge_node.GetDisplayNode().SetOpacity(0.55)
            # The wedge is still computed and kept in the scene (e.g. for export / checks) but no
            # longer shown by default - show it via the Data module if needed.
            wedge_node.GetDisplayNode().SetVisibility(False)
            if wedge_node.GetTransformNodeID():
                wedge_node.SetAndObserveTransformNodeID(None)
        elif wedge_node is not None:
            slicer.mrmlScene.RemoveNode(wedge_node)

        transform_node = slicer.mrmlScene.GetFirstNodeByName(transform_name)
        if transform_node is None:
            transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", transform_name)

        moving_node = distal_node if moving_is_distal else proximal_node
        fixed_node = proximal_node if moving_is_distal else distal_node
        if moving_node.GetTransformNodeID() != transform_node.GetID():
            moving_node.SetAndObserveTransformNodeID(transform_node.GetID())
        if fixed_node.GetTransformNodeID():
            fixed_node.SetAndObserveTransformNodeID(None)

        proximal_node.SetAndObservePolyData(proximal_polydata)
        distal_node.SetAndObservePolyData(distal_polydata)
        transform_node.SetMatrixTransformToParent(matrix)

        # Live X/Y fine shift (parent transform + X/Y-only gizmo on the master cut) and the
        # single-piece "as if healed" combined model, which bakes (fine shift @ correction) into
        # the mesh.
        self._fine_shift_state = {
            "side": side,
            "bone": bone,
            "origin": np.asarray(master_origin, dtype=float),
            "x": np.asarray(master_x, dtype=float),
            "y": np.asarray(master_y, dtype=float),
            "normal": np.asarray(master_normal, dtype=float),
            "base": osteotomy_logic.matrix_to_numpy(matrix),
            "moving_is_distal": moving_is_distal,
        }
        self._ensure_fine_shift_nodes(side, bone, transform_node)
        self._apply_fine_shift(update_combined=True, move_gizmo=True)

        # the original segmentation gives way to the corrected preview in 3D (see _sync_original_segmentation)
        self._sync_original_segmentation()

        self.correction_status.setText(
            f"Preview active ({self.osteotomy_technique.currentText}): wedge {wedge_description}, "
            f"rotation {self.rotation_angle.value:.1f}\N{DEGREE SIGN}. Yellow = proximal fragment, "
            "turquoise = distal fragment - the fragment that is moved shows the corrected position "
            "(the removed wedge is kept hidden in the scene). If the direction looks wrong, tilt the "
            "distal plane differently or change the reference plane.")
        if alignment_report:
            self.correction_status.setText(self.correction_status.text + "\n" + alignment_report)
        advice = self._cwo_wedge_advice(side, bone)
        if advice:
            self.correction_status.setText(self.correction_status.text + "\n" + advice)
        self._update_corrected_landmarks()
        self.onHideAnnotationsToggled(True)            # cutting planes hidden while the preview is shown
        self._apply_landmark_label_visibility()         # landmark names: 'Hide all landmark descriptions'

        # After the cut only the X/Y gizmo on the contact (master) face stays interactive; the
        # cutting planes' own handles are switched off so they cannot be moved by accident.
        # "Remove preview" switches them back on.
        self._set_cutting_plane_handles(side, bone, False)

    def _on_fragment_shift_changed(self, _value=None):
        # Update only when a preview already exists; otherwise retain the entered values for the
        # next explicit preview creation.  This avoids unexpectedly creating preview nodes while
        # the user is still planning the cuts.
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        state = getattr(self, "_fine_shift_state", None)
        if state and state["side"] == side and state["bone"] == bone:
            # Fast path: only the fine-shift transform, gizmo and combined model change - no
            # re-clipping of the bone.
            if self._apply_fine_shift(update_combined=True, move_gizmo=True):
                return
        names = self._preview_node_names(side, bone)
        if any(slicer.mrmlScene.GetFirstNodeByName(name) is not None for name in names):
            self.onPreviewCorrection()


    # ------------------------------------------------------------------
    # CWO contact-congruent alignment
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Planned correction, swing-twist plane derivation, bisecting principle
    # ------------------------------------------------------------------
    @staticmethod
    def _signed_rotation_value(bone, value, description):
        if bone == "Femur":
            return value if description == "Antetorsion" else -value
        return value if description.startswith("Internal") else -value

    # A cut BETWEEN the two shaft-axis landmarks: warn when the post-op value measured through both
    # midpoints differs from the planned value (axis of the fragment that carries the shaft) by more than this.
    SPLIT_AXIS_WARNING_DEG = 4.0

    def _split_shaft_axis(self, bone, positions, origin, normal, distal_sign):
        """(names, axis_on_distal_fragment) if the cut lies BETWEEN the two shaft-axis landmarks, else
        (set(), None). The shaft axis then belongs as a whole to the fragment that carries the larger
        part of the segment between the two landmarks: a cut close to the distal landmark (distal
        femoral osteotomy) leaves the axis on the proximal fragment, a cut close to the proximal
        landmark (proximal tibial osteotomy) puts it on the distal fragment."""
        names = [name for name in self.SHAFT_AXIS_LANDMARKS.get(bone, ()) if name in positions]
        if len(names) != 2:
            return set(), None
        depth = {name: float(np.dot(np.asarray(positions[name], dtype=float) - origin, normal)) * distal_sign
                 for name in names}                    # > 0: on the distal side of the cut
        if (depth[names[0]] > 0.0) == (depth[names[1]] > 0.0):
            return set(), None
        distal_part = max(depth.values()) / (max(depth.values()) - min(depth.values()))
        return set(names), distal_part >= 0.5

    def _split_shaft_axis_names(self, bone, positions, origin, normal, distal_sign):
        return self._split_shaft_axis(bone, positions, origin, normal, distal_sign)[0]

    def _predict_measurements(self, side, bone, H, plane_origin, plane_normal, moving_is_distal,
                              keep_split_axis=True):
        '''Move the landmarks lying on the MOVING side of the given (fixed) cut plane by the 4x4
        matrix H, read the wedge and torsion measurements, and restore the landmarks exactly.
        If the cut lies between the two shaft-axis landmarks (keep_split_axis), both go with the
        fragment that carries the larger part of the shaft (see _split_shaft_axis): the joints are then
        measured against that fragment's axis instead of a line across the osteotomy.
        keep_split_axis=False gives the value that would be measured through both midpoints after
        healing (used only for the split-axis warning).
        Returns {"wedge": value or None, "torsion": signed value or None, "axis_split": bool} or None.'''
        panel = self.left_panel if side == "left" else self.right_panel
        fiducials = slicer.mrmlScene.GetNodeByID(panel.markup_node_id) if getattr(panel, "markup_node_id", None) else None
        shaft, _, _, frame_error = self._get_anatomical_frame(side, bone)
        if fiducials is None or frame_error:
            return None
        shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))
        n = osteotomy_logic.normalize(np.asarray(plane_normal, dtype=float))
        o = np.asarray(plane_origin, dtype=float)
        distal_sign = 1.0 if float(np.dot(n, shaft)) >= 0.0 else -1.0
        H = np.asarray(H, dtype=float)
        originals = [np.array(fiducials.GetNthControlPointPositionWorld(i), dtype=float)
                     for i in range(fiducials.GetNumberOfControlPoints())]
        index_by_name = {l.name: l._id for l in getattr(panel, "landmarks", [])
                         if getattr(l, "placed", False) and getattr(l, "_id", None) is not None
                         and 0 <= l._id < len(originals)}
        split_names, axis_on_distal = self._split_shaft_axis(
            bone, {name: originals[i] for name, i in index_by_name.items()}, o, n, distal_sign)
        axis_ids = {index_by_name[name] for name in split_names} if keep_split_axis else set()
        axis_moves = axis_on_distal == moving_is_distal
        result = {"wedge": None, "torsion": None, "axis_split": bool(split_names)}
        try:
            for i, point in enumerate(originals):
                on_distal_side = float(np.dot(point - o, n)) * distal_sign > 0.0
                if (axis_moves if i in axis_ids else on_distal_side == moving_is_distal):
                    moved = (H @ np.append(point, 1.0))[:3]
                    fiducials.SetNthControlPointPositionWorld(i, *moved)
            wedge_measurement = self._get_measurement_by_name(panel, self.WEDGE_MEASUREMENT_NAME[bone])
            if wedge_measurement is not None:
                ready, value, _ = wedge_measurement()
                if ready:
                    result["wedge"] = float(value)
            rotation_measurement = self._get_measurement_by_name(panel, self.ROTATION_MEASUREMENT_NAME[bone])
            if rotation_measurement is not None:
                ready, value, description = rotation_measurement()
                if ready:
                    result["torsion"] = float(self._signed_rotation_value(bone, value, description))
            if bone == "Tibia" and hasattr(self, "_tibia_metrics"):
                result["tibia"] = self._tibia_metrics(panel)
        finally:
            for i, point in enumerate(originals):
                fiducials.SetNthControlPointPositionWorld(i, *point)
        return result

    def _split_axis_warning(self, side, bone, H, plane_origin, plane_normal, moving_is_distal, predicted):
        """Warning text if the cut lies between the shaft-axis landmarks AND the value measured
        through both midpoints after healing would differ from the plan by more than
        SPLIT_AXIS_WARNING_DEG; otherwise ''."""
        if not isinstance(predicted, dict) or not predicted.get("axis_split"):
            return ""
        chord = self._predict_measurements(side, bone, H, plane_origin, plane_normal, moving_is_distal,
                                           keep_split_axis=False)
        if not isinstance(chord, dict):
            return ""
        pairs = []
        if bone == "Tibia" and isinstance(predicted.get("tibia"), dict) and isinstance(chord.get("tibia"), dict):
            names = {"tpa": "TPA", "ampta": "aMPTA", "amdta": "aMDTA", "torsion": "torsion", "tma": "tibia/metatarsus"}
            pairs = [(names.get(k, k), predicted["tibia"][k], chord["tibia"][k])
                     for k in predicted["tibia"] if k in chord["tibia"]]
        else:
            pairs = [(self.WEDGE_MEASUREMENT_NAME[bone], predicted.get("wedge"), chord.get("wedge")),
                     (self.ROTATION_MEASUREMENT_NAME[bone], predicted.get("torsion"), chord.get("torsion"))]
        deg = "\N{DEGREE SIGN}"
        big = [f"{label} {planned:.1f}{deg} -> {through:.1f}{deg}" for label, planned, through in pairs
               if planned is not None and through is not None
               and np.isfinite(planned) and np.isfinite(through)
               and abs(float(through) - float(planned)) > self.SPLIT_AXIS_WARNING_DEG]
        if not big:
            return ""
        proximal_name, distal_name = self.SHAFT_AXIS_LANDMARKS["Femur" if bone == "Femur" else "Tibia"]
        return (f"WARNING: the cut lies between '{proximal_name}' and '{distal_name}'. The plan uses the "
                f"axis of the fragment that carries most of the shaft; measured through both midpoints after healing the values "
                f"would differ by more than {self.SPLIT_AXIS_WARNING_DEG:.0f}{deg}: " + ", ".join(big)
                + ". Consider moving the cut out of the shaft segment between these landmarks.")

    def _solve_correction_angles(self, make_h, wedge, torsion, side, bone, fixed_o, fixed_n, moving_is_distal,
                                 iterations=10):
        """Refine (wedge, torsion) so that the measurements on the moved landmarks hit the targets.
        make_h(wedge, torsion) -> 4x4. Rotating by the 'target - measured' difference does not
        change a 3D angle 1:1 (axes are not perpendicular to the rotation axis), so the angles are
        solved numerically (damped Newton with a finite-difference Jacobian).
        Returns (wedge, torsion, predicted)."""
        target = np.array([float(self.target_wedge_value.value), float(self.target_rotation_value.value)])

        def residual(x):
            predicted = self._predict_measurements(side, bone, make_h(x[0], x[1]), fixed_o, fixed_n, moving_is_distal)
            if predicted is None or predicted["wedge"] is None:
                return None, predicted
            r = [predicted["wedge"] - target[0],
                 (predicted["torsion"] - target[1]) if predicted["torsion"] is not None else 0.0]
            return np.array(r, dtype=float), predicted

        self._correction_warning = ""
        x0 = np.array([float(wedge), float(torsion)])
        x = x0.copy()
        r, predicted = residual(x)
        if r is None:
            return wedge, torsion, predicted
        predicted0 = predicted
        h = 0.5
        for _ in range(iterations):
            if np.max(np.abs(r)) < 0.05:
                break
            J = np.zeros((2, 2))
            for j in range(2):
                dx = np.zeros(2)
                dx[j] = h
                rj, _ = residual(x + dx)
                if rj is None:
                    return x[0], x[1], predicted
                J[:, j] = (rj - r) / h
            if abs(J[1, 1]) < 1e-9 and abs(J[1, 0]) < 1e-9:      # no torsion measurement: 1D
                J[1, 1] = 1.0
            try:
                step = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                break
            step = np.clip(step, -10.0, 10.0)
            for damping in (1.0, 0.5, 0.25):
                r_new, p_new = residual(x + damping * step)
                if r_new is not None and np.max(np.abs(r_new)) < np.max(np.abs(r)):
                    x, r, predicted = x + damping * step, r_new, p_new
                    break
            else:
                break
        deviation = np.abs(x - x0)
        # Large corrections legitimately need some extra angle (3D angles do not change 1:1);
        # a deviation that is BOTH > 5 deg and > 35 % of the nominal angle means a split axis.
        if np.any((deviation > 5.0) & (deviation > 0.35 * np.maximum(np.abs(x0), 1.0))):
            # A large deviation means the cut splits a measurement axis (e.g. the shaft-axis
            # landmarks lie on different fragments) - never turn that silently into a bigger wedge.
            moved, fixed = self._landmarks_by_fragment(side, fixed_o, fixed_n, moving_is_distal)
            self._correction_warning = (
                f"WARNING: hitting the targets would need wedge {abs(x[0]):.1f}\N{DEGREE SIGN} / torsion "
                f"{abs(x[1]):.1f}\N{DEGREE SIGN} instead of {abs(x0[0]):.1f}\N{DEGREE SIGN} / "
                f"{abs(x0[1]):.1f}\N{DEGREE SIGN} - refinement NOT applied. The cut probably splits a "
                f"measurement axis. Moved with the fragment: {', '.join(moved) or '-'}; fixed: "
                f"{', '.join(fixed) or '-'}.")
            return float(x0[0]), float(x0[1]), predicted0
        return float(x[0]), float(x[1]), predicted

    def _landmarks_by_fragment(self, side, fixed_o, fixed_n, moving_is_distal):
        panel = self.left_panel if side == "left" else self.right_panel
        shaft, _, _, error = self._get_anatomical_frame(side, self.osteotomy_bone.currentText)
        if error or panel is None:
            return [], []
        shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))
        n = osteotomy_logic.normalize(np.asarray(fixed_n, dtype=float))
        o = np.asarray(fixed_o, dtype=float)
        distal_sign = 1.0 if float(np.dot(n, shaft)) >= 0.0 else -1.0
        moved, fixed = [], []
        for landmark in getattr(panel, "landmarks", []):
            if not getattr(landmark, "placed", False):
                continue
            on_distal = float(np.dot(np.asarray(landmark.get_position(), dtype=float) - o, n)) * distal_sign > 0.0
            (moved if on_distal == moving_is_distal else fixed).append(landmark.name)
        return moved, fixed

    def _format_prediction(self, bone, predicted):
        deg = "\N{DEGREE SIGN}"
        parts = []
        tibia = predicted.get("tibia") if isinstance(predicted, dict) else None
        if bone == "Tibia" and tibia and getattr(self, "tibia_target_rows", None):
            names = {"tpa": "TPA", "ampta": "aMPTA", "amdta": "aMDTA", "torsion": "torsion", "tma": "tibia/metatarsus"}
            for key, row in self.tibia_target_rows.items():
                if key in tibia:
                    parts.append(f"{names[key]} {tibia[key]:.1f}{deg} (target {row['target'].value:.1f}{deg}"
                                 + ("" if row["use"].checked else ", not used") + ")")
            return ", ".join(parts)
        if predicted.get("wedge") is not None:
            parts.append(f"{self.WEDGE_MEASUREMENT_NAME[bone]} {predicted['wedge']:.1f}{deg} "
                         f"(target {self.target_wedge_value.value:.1f}{deg})")
        if predicted.get("torsion") is not None:
            parts.append(f"{self.ROTATION_MEASUREMENT_NAME[bone]} {predicted['torsion']:+.1f}{deg} "
                         f"(target {self.target_rotation_value.value:+.1f}{deg})")
        return ", ".join(parts) if parts else "n/a"

    def _desired_correction_rotation(self, side, bone, moving_is_distal, plane_origin, plane_normal):
        '''Planned rotation (3x3) of the MOVING fragment: frontal correction about the anatomical
        AP axis, then torsion about the shaft axis. Magnitudes come from measurement vs. target;
        the DIRECTIONS are verified, not assumed: each sign is chosen by moving the landmarks
        with the candidate rotation and checking which one brings the measurement to its target
        (no dependence on left/right mirroring or dataset orientation).
        Returns (R or None, info dict or error text).'''
        if bone == "Tibia" and hasattr(self, "_tibia_mode") and self._tibia_mode():
            # tibia: TPA, aMPTA, aMDTA, torsion(s) - frontal, sagittal and torsion solved together
            return self._solve_tibia_correction(side, bone, moving_is_distal, plane_origin, plane_normal)
        shaft, ml, ap, frame_error = self._get_anatomical_frame(side, bone)
        if frame_error:
            return None, frame_error
        shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))
        ap = osteotomy_logic.normalize(np.asarray(ap, dtype=float))
        panel = self.left_panel if side == "left" else self.right_panel
        wedge_measurement = self._get_measurement_by_name(panel, self.WEDGE_MEASUREMENT_NAME[bone])
        if wedge_measurement is None:
            return None, "Wedge measurement not found."
        ready, current_wedge, _ = wedge_measurement()
        if not ready:
            return None, f"Place the landmarks for '{self.WEDGE_MEASUREMENT_NAME[bone]}' ({side}) first."
        ok_rot, torsion_delta, rotation_message = self._calculated_rotation_delta(side, bone)
        if not ok_rot:
            return None, rotation_message
        frontal = abs(float(self.target_wedge_value.value) - float(current_wedge))
        torsion = abs(float(torsion_delta))
        pivot = np.asarray(plane_origin, dtype=float)

        def rot3(axis, angle):
            return self._rotation_about_line(axis, angle, np.zeros(3))[:3, :3]

        # femur: manual sagittal correction about the ML axis (+ = procurvatum corrected: the far end of
        # the moving fragment goes cranially). Signed about the frame's ml axis like the tibia's.
        ml_unit = osteotomy_logic.normalize(np.asarray(ml, dtype=float))
        sagittal_signed = self._femur_sagittal_signed(side, bone, shaft, ml_unit, moving_is_distal)
        R_sagittal = rot3(ml_unit, sagittal_signed) if abs(sagittal_signed) > 1e-6 else np.eye(3)

        def front(angle):
            return R_sagittal @ rot3(ap, angle)

        def compose(R_torsion, R_frontal_part):
            # derotation about the MOVING fragment's own long axis (see anatomical_correction_matrix)
            return R_torsion @ R_frontal_part if moving_is_distal else R_frontal_part @ R_torsion

        def as_h(R):
            H = np.eye(4, dtype=float)
            H[:3, :3] = R
            H[:3, 3] = pivot - R @ pivot
            return H

        R_frontal = R_sagittal.copy()
        if frontal > 0.01:
            best = None
            for sign in (1.0, -1.0):
                candidate = front(sign * frontal)
                predicted = self._predict_measurements(side, bone, as_h(candidate), plane_origin, plane_normal, moving_is_distal)
                if predicted is None or predicted["wedge"] is None:
                    continue
                error = abs(predicted["wedge"] - float(self.target_wedge_value.value))
                if best is None or error < best[0]:
                    best = (error, candidate)
            if best is not None:
                R_frontal = best[1]
        R = R_frontal
        if torsion > 0.01:
            best = None
            for sign in (1.0, -1.0):
                candidate = compose(rot3(shaft, sign * torsion), R_frontal)
                predicted = self._predict_measurements(side, bone, as_h(candidate), plane_origin, plane_normal, moving_is_distal)
                if predicted is None or predicted["torsion"] is None:
                    continue
                error = abs(predicted["torsion"] - float(self.target_rotation_value.value))
                if best is None or error < best[0]:
                    best = (error, candidate)
            if best is not None:
                R = best[1]
        # signs as chosen above, then solve the magnitudes on the landmarks
        frontal_signed = frontal
        if frontal > 0.01:
            frontal_signed = frontal if np.allclose(R_frontal, front(frontal)) else -frontal
        torsion_signed = torsion
        if torsion > 0.01:
            torsion_signed = torsion if np.allclose(R, compose(rot3(shaft, torsion), R_frontal)) else -torsion
        frontal_signed, torsion_signed, _ = self._solve_correction_angles(
            lambda f, t: as_h(compose(rot3(shaft, t), front(f))), frontal_signed, torsion_signed,
            side, bone, plane_origin, plane_normal, moving_is_distal)
        R = compose(rot3(shaft, torsion_signed), front(frontal_signed))
        frontal, torsion = abs(frontal_signed), abs(torsion_signed)
        predicted = self._predict_measurements(side, bone, as_h(R), plane_origin, plane_normal, moving_is_distal)
        warning = "\n".join(text for text in (
            getattr(self, "_correction_warning", ""),
            self._split_axis_warning(side, bone, as_h(R), plane_origin, plane_normal, moving_is_distal, predicted))
            if text)
        return R, {"frontal": frontal, "torsion": torsion, "predicted": predicted,
                   "frontal_signed": float(frontal_signed), "torsion_signed": float(torsion_signed),
                   "sagittal_signed": float(sagittal_signed), "warning": warning}

    def _femur_sagittal_signed(self, side, bone, shaft, ml_unit, moving_is_distal):
        """Manual femur sagittal correction as a signed angle about ml_unit: positive input = the far
        end of the MOVING fragment turns cranially (corrects a procurvatum)."""
        box = getattr(self, "femur_sagittal_value", None)
        if bone != "Femur" or box is None or abs(float(box.value)) < 1e-6:
            return 0.0
        shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))          # proximal -> distal
        # cranial as in the aligned slice views (screen-up of the red view): (ML, patient-mirrored) x proximal
        cranial = osteotomy_logic.normalize(np.cross(ml_unit if side == "left" else -ml_unit, -shaft))
        far_end = shaft if moving_is_distal else -shaft
        turns_to = np.cross(ml_unit, far_end)                  # where a positive turn about ml moves the far end
        sign = 1.0 if float(np.dot(turns_to, cranial)) >= 0.0 else -1.0
        return sign * float(box.value)

    def _section_extent(self, origin, normal, direction):
        '''Extent (+direction, -direction) of the bone cross-section in the given plane, for the
        connected contour piece nearest to origin. (0, 0) if unavailable.'''
        bone_polydata = self._get_bone_polydata()
        if bone_polydata is None:
            return 0.0, 0.0
        plane = vtk.vtkPlane()
        plane.SetOrigin(*[float(v) for v in origin])
        plane.SetNormal(*[float(v) for v in normal])
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(bone_polydata)
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputConnection(cutter.GetOutputPort())
        connectivity.SetExtractionModeToClosestPointRegion()
        connectivity.SetClosestPoint(*[float(v) for v in origin])
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(connectivity.GetOutputPort())
        cleaner.Update()
        points = cleaner.GetOutput().GetPoints()
        if points is None or points.GetNumberOfPoints() == 0:
            return 0.0, 0.0
        from vtk.util.numpy_support import vtk_to_numpy
        projections = (vtk_to_numpy(points.GetData()).astype(float) - np.asarray(origin, dtype=float)) @ np.asarray(direction, dtype=float)
        return max(0.0, float(projections.max())), max(0.0, -float(projections.min()))

    def _sync_other_plane_swing_twist(self, side, bone, master_is_distal, master_origin, master_x,
                                      master_y, master_n, other_node, label):
        '''Derive the SECOND plane so that closing the wedge (hinge = swing) followed by the
        twist about the contact normal reproduces the planned correction EXACTLY, for any MASTER
        orientation. desired = twist(n_contact) @ swing;  second-plane normal m = R^T n_contact;
        hinge axis = m x n_contact (lies in both planes); hinge placed at the cortex.'''
        moving_is_distal = not master_is_distal
        R, info = self._desired_correction_rotation(side, bone, moving_is_distal, master_origin, master_n)
        if R is None:
            return False, info
        deg = "\N{DEGREE SIGN}"
        n_f = osteotomy_logic.normalize(np.asarray(master_n, dtype=float))
        master_origin = np.asarray(master_origin, dtype=float)
        m = osteotomy_logic.normalize(R.T @ n_f)
        swing = float(np.degrees(np.arccos(float(np.clip(np.dot(m, n_f), -1.0, 1.0)))))
        shaft, _, _, _ = self._get_anatomical_frame(side, bone)
        shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))

        hinge_point = master_origin
        hinge_note = "hinge through the plane centre"
        axis = np.cross(m, n_f)
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            d = osteotomy_logic.normalize(np.cross(n_f, axis))
            extent_pos, extent_neg = self._section_extent(master_origin, n_f, d)
            move_sign = 1.0 if moving_is_distal else -1.0
            moving_dir_master = n_f * (1.0 if float(np.dot(n_f, shaft)) >= 0 else -1.0) * move_sign
            moving_dir_second = m * (1.0 if float(np.dot(m, shaft)) >= 0 else -1.0) * move_sign
            probe = master_origin + 0.5 * moving_dir_master
            best = None
            for sigma, extent in ((1.0, extent_pos), (-1.0, extent_neg)):
                candidate = master_origin + sigma * extent * d
                score = float(np.dot(probe - candidate, moving_dir_second))  # < 0: probe inside wedge
                if best is None or score < best[0]:
                    best = (score, candidate, extent)
            hinge_point = best[1]
            hinge_note = f"hinge at the cortex ({best[2]:.1f} mm from the centre)"

        other_origin, _, _, _ = self._get_plane_frame(other_node, side, bone)
        other_origin = np.asarray(other_origin, dtype=float)
        other_origin = other_origin - float(np.dot(other_origin - hinge_point, m)) * m
        self._set_plane_frame(other_node, other_origin, R.T @ np.asarray(master_x, dtype=float),
                              R.T @ np.asarray(master_y, dtype=float), m)
        self._update_ml_markers(other_node, side, bone)
        self._update_actual_plane_angle(master_n, m)

        # twist = rotation about the contact normal left after the swing (hinge closure)
        swing_matrix = self._rotation_about_line(axis, swing, np.zeros(3))[:3, :3] if np.linalg.norm(np.cross(m, n_f)) > 1e-6 else np.eye(3)
        if float(np.dot(swing_matrix @ m, n_f)) < 0.999:
            swing_matrix = swing_matrix.T
        T = R @ swing_matrix.T
        u = osteotomy_logic.normalize(np.asarray(master_x, dtype=float))
        twist = self._signed_angle_about(u, T @ u, n_f)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(n_f, shaft)))))))
        predicted = info.get("predicted") if isinstance(info, dict) else None
        text = (f"{label} plane derived (swing-twist): wedge {swing:.1f}{deg} "
                f"(frontal correction {info['frontal']:.1f}{deg}, torsion {info['torsion']:.1f}{deg}), "
                f"{hinge_note}; after closing, twist {twist:+.1f}{deg} about the contact normal. "
                f"MASTER plane is {tilt:.0f}{deg} oblique to the shaft.")
        if predicted:
            text += " Predicted: " + self._format_prediction(bone, predicted)
        if isinstance(info, dict) and info.get("warning"):
            text += "\n" + info["warning"]
        return True, text

    def onAlignMasterBisecting(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        master_is_distal = self.reference_plane_selector.currentText.startswith("Distal")
        master_name = self._distal_plane_node_name(side, bone) if master_is_distal else self._plane_node_name(side, bone)
        master = slicer.mrmlScene.GetFirstNodeByName(master_name)
        if master is None:
            self.osteotomy_status.setText("Insert the MASTER plane first ('Insert/update MASTER plane').")
            return
        anatomical_origin, shaft_axis, axis_error = self._get_anatomical_axis(side, bone)
        if axis_error:
            self.osteotomy_status.setText(axis_error)
            return
        _, _, _, frame_error = self._get_anatomical_frame(side, bone)
        if frame_error:
            self.osteotomy_status.setText(frame_error)
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            origin, x, y, n = self._get_plane_frame(master, side, bone)
            moving_is_distal = not master_is_distal
            R, info = self._desired_correction_rotation(side, bone, moving_is_distal, origin, n)
            if R is None:
                self.osteotomy_status.setText(str(info))
                return
            shaft, _, _, _ = self._get_anatomical_frame(side, bone)
            shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))
            # Contact face perpendicular to the CORRECTED axis of the moving fragment: that is the
            # shaft itself when the distal fragment moves, R*shaft when the proximal one moves.
            new_n = shaft if moving_is_distal else osteotomy_logic.normalize(R @ shaft)
            if float(np.dot(new_n, n)) < 0.0:
                new_n = -new_n
            new_origin = osteotomy_logic.closest_point_on_line(origin, anatomical_origin, shaft_axis)
            new_x = np.asarray(x, dtype=float) - float(np.dot(x, new_n)) * new_n
            if np.linalg.norm(new_x) < 1e-6:
                new_x = np.asarray(y, dtype=float) - float(np.dot(y, new_n)) * new_n
            new_x = osteotomy_logic.normalize(new_x)
            new_y = osteotomy_logic.normalize(np.cross(new_n, new_x))
            self._set_plane_frame(master, new_origin, new_x, new_y, new_n)
            self._update_ml_markers(master, side, bone)
            other_name = self._plane_node_name(side, bone) if master_is_distal else self._distal_plane_node_name(side, bone)
            if slicer.mrmlScene.GetFirstNodeByName(other_name) is None:
                self.osteotomy_status.setText(
                    "MASTER plane aligned (bisecting principle). Now create the SECOND plane.")
                return
            ok, message = self._sync_other_plane(side, bone)
            self.osteotomy_status.setText(("MASTER plane aligned (bisecting principle). " if ok else "") + message)
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onResetPlanningAndGuides(self):
        """Planning reset + both guides deleted (saw and reduction guide) - after asking."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        answer = qt.QMessageBox.question(
            slicer.util.mainWindow(), "Reset planning and guides",
            f"Delete the osteotomy planning ({side} {bone.lower()}: preview, cutting planes) and BOTH guides "
            "(saw guide and reduction guide: footprints, bodies, pins, sleeves, saw boxes)?\n\n"
            "Landmarks, measurements, segmentation and target values are kept. This cannot be undone.",
            qt.QMessageBox.Yes | qt.QMessageBox.No, qt.QMessageBox.No)
        if answer != qt.QMessageBox.Yes:
            return
        # guide editing tools off, hidden things back
        for name in ("_stop_guide_edit", "_restore_guide_hidden", "_remove_pin_gizmo", "_remove_pin_preview"):
            method = getattr(self, name, None)
            if method is not None:
                try:
                    method()
                except Exception:
                    logging.exception("Doggybone: reset step failed (continuing)")
        removed = 0
        prefixes = ("Doggybone guide", "Doggybone pin ", "Doggybone saw box", "Doggybone saw guide pin preview")
        tibia = bone == "Tibia"
        for node in list(slicer.util.getNodesByClass("vtkMRMLNode")):
            name = node.GetName() or ""
            if not name.startswith(prefixes) or node.IsA("vtkMRMLDisplayNode") or node.IsA("vtkMRMLStorageNode"):
                continue
            # only THIS bone's guide objects (tibia objects carry ' tibia' in the name); the shared
            # bone surfaces for painting are rebuilt anyway
            if (" tibia" in name) != tibia and not name.startswith("Doggybone guide bone surface"):
                continue
            slicer.mrmlScene.RemoveNode(node)
            removed += 1
        # guide state in the module
        for attribute, value in (
                                 ("_selected_pin_id", None), ("_pins_dirty", False), ("_pin_rows_active", False),
                                 ("_saw_box_gizmo_tags", {}), ("_guide_saved_opacity", []),
                                 ("_guide_locked_planes", []), ("_guide_hidden_preview", [])):
            if hasattr(self, attribute):
                setattr(self, attribute, value)
        self.onResetOsteotomyPlanning()
        self._ensure_segmentation_visible()
        message = (f"Planning and guides reset ({side} {bone.lower()}): preview, cutting planes and both guides "
                   f"removed ({removed} guide objects). Landmarks, segmentation and target values kept.")
        self.osteotomy_status.setText(message)
        if hasattr(self, "guide_status"):
            self.guide_status.setStyleSheet("")
            self.guide_status.setText("No guide - planning was reset.")

    def onResetOsteotomyPlanning(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText

        # 1) preview, fine-shift transform and gizmo
        self.onClearPreview()
        self._remove_fine_shift_observers()
        self._fine_shift_state = None
        for name in self._fine_shift_node_names(side, bone):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)

        # 2) both cutting planes with their L/M markers
        removed_planes = 0
        for plane_name in (self._plane_node_name(side, bone), self._distal_plane_node_name(side, bone)):
            for name in (plane_name, f"{plane_name} L marker", f"{plane_name} M marker"):
                node = slicer.mrmlScene.GetFirstNodeByName(name)
                while node is not None:
                    slicer.mrmlScene.RemoveNode(node)
                    removed_planes += 1
                    node = slicer.mrmlScene.GetFirstNodeByName(name)

        # 3) numeric fields back to zero (targets are clinical choices and are kept)
        for box in (self.fragment_shift_x, self.fragment_shift_y):
            was_blocked = box.blockSignals(True)
            box.setValue(0.0)
            box.blockSignals(was_blocked)
        self.actual_plane_angle_label.setText("Master/SECOND angle: --")

        # 4) re-read the CURRENT landmarks: measured values + calculated corrections
        self._on_target_values_changed()

        self._ensure_segmentation_visible()

        message = (f"Planning reset ({side} {bone.lower()}): preview, gizmo and cutting planes "
                   f"removed; values re-read from the current landmarks - "
                   f"{self.current_values_label.text}")
        self.osteotomy_status.setText(message)
        self.correction_status.setText("No preview.")
        self._remove_orphan_display_nodes()

    @staticmethod
    def _remove_orphan_display_nodes():
        """Display nodes whose plane / point list was deleted stay in the scene and are still DRAWN
        (old cutting planes in the 3D view after a reset) - remove them, then redraw the views."""
        browsers = list(slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode"))
        for node in list(slicer.util.getNodesByClass("vtkMRMLDisplayNode")):
            if not (node.IsA("vtkMRMLMarkupsDisplayNode") or node.IsA("vtkMRMLModelDisplayNode")):
                continue
            if node.GetDisplayableNode() is not None or node.GetScene() is None:
                continue
            if any(b.GetSequenceNode(node) is not None for b in browsers):
                # kept alive by a saved Scene View (snapshot): deleting it would make Slicer re-create
                # it and re-apply the snapshot (camera, visibilities) - only switch it off
                node.SetVisibility(False)
                continue
            slicer.mrmlScene.RemoveNode(node)
        slicer.util.forceRenderAllViews()

    def _set_cutting_plane_handles(self, side, bone, interactive):
        for name in (self._plane_node_name(side, bone), self._distal_plane_node_name(side, bone)):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetHandlesInteractive(bool(interactive))

    def _cwo_alignment_is_congruent(self):
        return True       # CWO: always contact-congruent (measurement-based mode removed)

    @staticmethod
    def _rotation_about_line(axis, angle_deg, pivot):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        a = np.radians(float(angle_deg))
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        R = np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)
        H = np.eye(4, dtype=float)
        H[:3, :3] = R
        H[:3, 3] = np.asarray(pivot, dtype=float) - R @ np.asarray(pivot, dtype=float)
        return H

    @staticmethod
    def _signed_angle_about(v1, v2, axis):
        axis = np.asarray(axis, dtype=float); axis = axis / np.linalg.norm(axis)
        a = np.asarray(v1, dtype=float) - float(np.dot(v1, axis)) * axis
        b = np.asarray(v2, dtype=float) - float(np.dot(v2, axis)) * axis
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
            return 0.0
        return float(np.degrees(np.arctan2(float(np.dot(np.cross(a, b), axis)), float(np.dot(a, b)))))

    def _contact_congruent_matrix(self, proximal_origin, proximal_normal, distal_origin, distal_normal,
                                  hinge_point, moving_is_distal, shaft_axis, shaft_origin,
                                  rotation_deg, ap_axis, ml_axis, wedge_needed_deg,
                                  side=None, bone=None, desired_rotation=None):
        '''CWO, contact-congruent alignment of the moving fragment. Returns (vtkMatrix4x4, report).

        1) Hinge closure: rotate about the intersection line of both planes (the hinge) by their
           dihedral angle. Both planes contain that line, so the moving cut face lands EXACTLY on
           the fixed cut face - congruent contact, hinge cortex stays continuous.
        2) Torsion about the NORMAL of that contact face, through the point where the shaft axis
           pierces it - the only rotation still possible once the faces are compressed flush.
           If the face is oblique to the shaft this also changes angulation; that is reported.
        3) No automatic translation; in-plane fine positioning is the separate X/Y shift.
        '''
        n_p = np.asarray(proximal_normal, dtype=float); n_p = n_p / np.linalg.norm(n_p)
        n_d = np.asarray(distal_normal, dtype=float); n_d = n_d / np.linalg.norm(n_d)
        if moving_is_distal:
            moving_n, fixed_n, fixed_o = n_d, n_p, np.asarray(proximal_origin, dtype=float)
        else:
            moving_n, fixed_n, fixed_o = n_p, n_d, np.asarray(distal_origin, dtype=float)

        hinge_axis = np.cross(n_p, n_d)
        if np.linalg.norm(hinge_axis) < 1e-6:
            # parallel cuts: no hinge - the closure is a pure translation that sets the moving face
            # onto the fixed one (removes the slab between the cuts)
            moving_o = np.asarray(distal_origin if moving_is_distal else proximal_origin, dtype=float)
            closure_deg = 0.0
            H_close = np.eye(4)
            H_close[:3, 3] = -float(np.dot(moving_o - fixed_o, fixed_n)) * fixed_n
        else:
            hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)
            target_n = fixed_n if float(np.dot(moving_n, fixed_n)) >= 0.0 else -fixed_n
            closure_deg = self._signed_angle_about(moving_n, target_n, hinge_axis)
            H_close = self._rotation_about_line(hinge_axis, closure_deg, hinge_point)

        shaft = np.asarray(shaft_axis, dtype=float); shaft = shaft / np.linalg.norm(shaft)
        contact_n = fixed_n if float(np.dot(fixed_n, shaft)) >= 0.0 else -fixed_n
        denom = float(np.dot(shaft, contact_n))
        shaft_o = np.asarray(shaft_origin, dtype=float)
        if abs(denom) > 1e-6:
            pierce = shaft_o + shaft * float(np.dot(fixed_o - shaft_o, contact_n)) / denom
        else:
            pierce = fixed_o
        # Twist about the contact normal. With a planned correction (desired_rotation) it is the
        # twist part of desired = twist(contact normal) @ swing(hinge closure); plane deviations
        # from that plan show up as a non-twist remainder that is reported, not hidden.
        plan_deviation = None
        twist_deg = rotation_deg
        if desired_rotation is not None:
            T = np.asarray(desired_rotation, dtype=float) @ H_close[:3, :3].T
            u = np.cross(contact_n, [1.0, 0.0, 0.0])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(contact_n, [0.0, 1.0, 0.0])
            u = u / np.linalg.norm(u)
            twist_deg = self._signed_angle_about(u, T @ u, contact_n)
            plan_deviation = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(T @ contact_n, contact_n)))))))
        H_torsion = self._rotation_about_line(contact_n, twist_deg, pierce)
        H = H_torsion @ H_close

        ap = np.asarray(ap_axis, dtype=float); ml = np.asarray(ml_axis, dtype=float)
        s_closed = H_close[:3, :3] @ shaft
        s_final = H[:3, :3] @ shaft
        frontal_close = self._signed_angle_about(shaft, s_closed, ap)
        sagittal_close = self._signed_angle_about(shaft, s_closed, ml)
        induced = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(s_closed, s_final)))))))
        frontal_total = self._signed_angle_about(shaft, s_final, ap)
        sagittal_total = self._signed_angle_about(shaft, s_final, ml)
        face_tilt = float(np.degrees(np.arccos(min(1.0, abs(denom)))))
        residual = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(H[:3, :3] @ moving_n, fixed_n)))))))

        deg = "\N{DEGREE SIGN}"
        report = (
            f"Contact-congruent: hinge closure {abs(closure_deg):.1f}{deg} -> frontal "
            f"{abs(frontal_close):.1f}{deg} (needed per measurement {wedge_needed_deg:.1f}{deg}, "
            f"difference {abs(frontal_close) - wedge_needed_deg:+.1f}{deg}), sagittal "
            f"{abs(sagittal_close):.1f}{deg}. Torsion {twist_deg:+.1f}{deg} about the contact "
            f"normal (face {face_tilt:.0f}{deg} oblique to the shaft) adds {induced:.1f}{deg} "
            f"angulation. Total: frontal {abs(frontal_total):.1f}{deg}, sagittal "
            f"{abs(sagittal_total):.1f}{deg}. Cut faces {residual:.2f}{deg} apart (congruent).")
        if plan_deviation is not None and plan_deviation > 0.5:
            report += (f" WARNING: the planes deviate {plan_deviation:.1f}{deg} from the planned "
                       f"correction - re-derive the SECOND plane.")
        if side is not None and bone is not None:
            predicted = self._predict_measurements(side, bone, H, fixed_o, fixed_n, moving_is_distal)
            if predicted is not None:
                report += "\nPredicted after correction: " + self._format_prediction(bone, predicted)
        return osteotomy_logic.numpy_to_vtk_matrix(H), report

    # ------------------------------------------------------------------
    # Live X/Y fine shift: parent transform + gizmo restricted to the master plane's X/Y axes
    # ------------------------------------------------------------------
    def _fine_shift_node_names(self, side, bone):
        return (f"Doggybone {side} {bone.lower()} fine shift (X/Y)",
                f"Doggybone {side} {bone.lower()} fine shift gizmo")

    def _remove_fine_shift_observers(self):
        for node, tag in getattr(self, "_fine_shift_observers", []):
            try:
                node.RemoveObserver(tag)
            except Exception:
                logging.exception("Doggybone: non-critical step failed (continuing)")
        self._fine_shift_observers = []

    def _ensure_fine_shift_nodes(self, side, bone, correction_transform_node):
        shift_name, gizmo_name = self._fine_shift_node_names(side, bone)
        state = self._fine_shift_state

        shift_node = slicer.mrmlScene.GetFirstNodeByName(shift_name)
        if shift_node is None:
            shift_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", shift_name)
        if correction_transform_node.GetTransformNodeID() != shift_node.GetID():
            correction_transform_node.SetAndObserveTransformNodeID(shift_node.GetID())

        gizmo = slicer.mrmlScene.GetFirstNodeByName(gizmo_name)
        if gizmo is None:
            gizmo = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", gizmo_name)
            gizmo.CreateDefaultDisplayNodes()
        self._remove_fine_shift_observers()

        self._fine_shift_updating = True
        try:
            gizmo.SetAxesWorld(state["x"], state["y"], state["normal"])
            gizmo.SetSize(20.0, 20.0)
            display = gizmo.GetDisplayNode()
            if display is not None:
                display.SetSelectedColor(0.2, 0.85, 1.0)
                display.SetColor(0.2, 0.85, 1.0)
                display.SetFillOpacity(0.15)
                display.SetPropertiesLabelVisibility(False)
                display.SetHandlesInteractive(True)
                display.SetTranslationHandleVisibility(True)
                display.SetRotationHandleVisibility(False)
                display.SetScaleHandleVisibility(False)
                # Only the in-plane X and Y arrows: no normal (Z) arrow, no free view-plane drag.
                display.SetTranslationHandleComponentVisibility([True, True, False, False])
                display.SetInteractionHandleScale(6.0)
                # The gizmo sits on the contact face, i.e. inside the bone surface: keep it
                # visible through the fragments.
                display.SetOccludedVisibility(True)
                display.SetOccludedOpacity(0.6)
        finally:
            self._fine_shift_updating = False

        self._fine_shift_observers = [
            (gizmo, gizmo.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_fine_shift_gizmo_moved)),
            (gizmo, gizmo.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._on_fine_shift_gizmo_moved)),
            (gizmo, gizmo.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self._on_fine_shift_gizmo_released)),
        ]
        return shift_node

    def _place_fine_shift_gizmo(self, position):
        state = getattr(self, "_fine_shift_state", None)
        if not state:
            return
        gizmo = slicer.mrmlScene.GetFirstNodeByName(self._fine_shift_node_names(state["side"], state["bone"])[1])
        if gizmo is None:
            return
        self._fine_shift_updating = True
        try:
            gizmo.SetAxesWorld(state["x"], state["y"], state["normal"])
            gizmo.SetOriginWorld(float(position[0]), float(position[1]), float(position[2]))
        finally:
            self._fine_shift_updating = False

    def _apply_fine_shift(self, update_combined=True, move_gizmo=True):
        state = getattr(self, "_fine_shift_state", None)
        if not state:
            return False
        side, bone = state["side"], state["bone"]
        shift_node = slicer.mrmlScene.GetFirstNodeByName(self._fine_shift_node_names(side, bone)[0])
        if shift_node is None:
            return False

        shift = (float(self.fragment_shift_x.value) * state["x"] +
                 float(self.fragment_shift_y.value) * state["y"])
        T = np.eye(4, dtype=float)
        T[:3, 3] = shift
        shift_node.SetMatrixTransformToParent(osteotomy_logic.numpy_to_vtk_matrix(T))

        if move_gizmo:
            self._place_fine_shift_gizmo(state["origin"] + shift)

        if update_combined:
            proximal_name, distal_name, _, combined_name, _ = self._preview_node_names(side, bone)
            proximal_node = slicer.mrmlScene.GetFirstNodeByName(proximal_name)
            distal_node = slicer.mrmlScene.GetFirstNodeByName(distal_name)
            if proximal_node is None or distal_node is None:
                return True
            if state["moving_is_distal"]:
                fixed_polydata, moving_polydata = proximal_node.GetPolyData(), distal_node.GetPolyData()
            else:
                fixed_polydata, moving_polydata = distal_node.GetPolyData(), proximal_node.GetPolyData()
            if fixed_polydata is None or moving_polydata is None:
                return True
            combined_matrix = osteotomy_logic.numpy_to_vtk_matrix(T @ state["base"])
            combined_polydata = osteotomy_logic.combine_fragments(fixed_polydata, moving_polydata, combined_matrix)
            combined_node = slicer.mrmlScene.GetFirstNodeByName(combined_name)
            if combined_node is None:
                combined_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", combined_name)
                combined_node.CreateDefaultDisplayNodes()
                combined_node.GetDisplayNode().SetColor(0.85, 0.8, 0.65)
                combined_node.GetDisplayNode().SetOpacity(1.0)
                combined_node.GetDisplayNode().SetVisibility(False)    # osteotomy view: the coloured fragments
            combined_node.SetAndObservePolyData(combined_polydata)
        if update_combined:
            self._update_corrected_landmarks()
        return True

    def _on_fine_shift_gizmo_moved(self, caller, event=None):
        if getattr(self, "_fine_shift_updating", False):
            return
        state = getattr(self, "_fine_shift_state", None)
        if not state:
            return
        origin = [0.0, 0.0, 0.0]
        caller.GetOriginWorld(origin)
        delta = np.asarray(origin, dtype=float) - state["origin"]
        sx = float(np.dot(delta, state["x"]))
        sy = float(np.dot(delta, state["y"]))
        clamped_x = min(max(sx, self.fragment_shift_x.minimum), self.fragment_shift_x.maximum)
        clamped_y = min(max(sy, self.fragment_shift_y.minimum), self.fragment_shift_y.maximum)

        for box, value in ((self.fragment_shift_x, clamped_x), (self.fragment_shift_y, clamped_y)):
            was_blocked = box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(was_blocked)

        # Use the exact (unrounded) gizmo position for the live shift; the fields show 0.1 mm.
        shift_node = slicer.mrmlScene.GetFirstNodeByName(self._fine_shift_node_names(state["side"], state["bone"])[0])
        if shift_node is not None:
            T = np.eye(4, dtype=float)
            T[:3, 3] = clamped_x * state["x"] + clamped_y * state["y"]
            shift_node.SetMatrixTransformToParent(osteotomy_logic.numpy_to_vtk_matrix(T))

        # Keep the gizmo on the master plane (no normal drift) and inside the +/-20 mm limit.
        off_plane = abs(float(np.dot(delta, state["normal"]))) > 1e-6
        if off_plane or clamped_x != sx or clamped_y != sy:
            self._place_fine_shift_gizmo(state["origin"] + clamped_x * state["x"] + clamped_y * state["y"])

    def _on_fine_shift_gizmo_released(self, caller=None, event=None):
        if getattr(self, "_fine_shift_updating", False):
            return
        # Snap to the 0.1 mm the fields display, then rebuild the combined model once.
        self._apply_fine_shift(update_combined=True, move_gizmo=True)

    CORRECTED_PREFIX = "Doggybone {side} corrected"
    CORRECTED_STATUS_MARK = "\nCorrected position (measured on the moved landmarks): "

    def _on_osteotomy_panel_opened(self):
        """Back in the osteotomy planning (e.g. from the guide tools): coloured fragments, original /
        compare views as the osteotomy planning shows them."""
        self._show_preview_fragments()
        self._sync_original_segmentation()

    def _show_preview_fragments(self):
        """Osteotomy view: the two fragments in their own colours (fixed yellow, moved turquoise) and NOT
        the combined bone - it is the same surface, both visible at once fight for the same pixels and
        the turquoise only flickers through the beige. The combined bone stays in the scene (reduction
        guide, export) and is shown by the reduction guide mode."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        proximal_name, distal_name, _t, combined_name, _w = self._preview_node_names(side, bone)
        combined = slicer.mrmlScene.GetFirstNodeByName(combined_name)
        if combined is None or combined.GetDisplayNode() is None:
            return
        combined.GetDisplayNode().SetVisibility(False)
        for name in (proximal_name, distal_name):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(True)

    def _sync_original_segmentation(self):
        """While a correction preview of the current side / bone exists, the ORIGINAL segmentation
        is hidden in 3D (it stays in the slice views) - whatever showed it again (bone switch, guide
        tools, opacity). Exception: the saw guide is being planned (guide panel open, saw mode) -
        that guide sits on the uncut bone."""
        if getattr(self, "_guide_shown_surface", None):
            return      # a guide tool shows its own bone surface - the segmentation stays hidden meanwhile
        guide_panel = getattr(self, "_guide_collapsible", None)
        if guide_panel is None or guide_panel.collapsed:
            self._show_preview_fragments()    # no guide tool open: fragments in colour, not the combined bone
        if self._compare_mode():
            self._apply_compare_views()       # original segmentation in the left view, corrected bone right
            return
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetDisplayNode() is None:
            return
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
        if combined is None or combined.GetPolyData() is None or combined.GetPolyData().GetNumberOfPoints() == 0:
            return
        guide_panel = getattr(self, "_guide_collapsible", None)
        saw_guide_view = (getattr(self, "_guide_mode", "saw") == "saw" and guide_panel is not None
                          and not guide_panel.collapsed)
        display = segmentation_node.GetDisplayNode()
        if hasattr(display, "SetVisibility3D"):
            display.SetVisibility(True)
            display.SetVisibility3D(saw_guide_view)
        else:
            display.SetVisibility(saw_guide_view)

    def _apply_segment_opacity(self):
        """Opacity of the preview segments (fixed + moved fragment, combined bone)."""
        slider = getattr(self, "segment_opacity_slider", None)
        if slider is None:
            return
        opacity = slider.value / 100.0
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        proximal_name, distal_name, _t, combined_name, _w = self._preview_node_names(side, bone)
        for name in (proximal_name, distal_name, combined_name):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetOpacity(opacity)
        self._sync_original_segmentation()

    def _original_overlay_nodes(self, side):
        panel = self.left_panel if side == "left" else self.right_panel
        if panel is None:
            return []
        nodes = [slicer.mrmlScene.GetNodeByID(panel.markup_node_id)]
        nodes += list(getattr(panel, "sphere_nodes", {}).values())
        nodes += list(getattr(panel, "axis_nodes", {}).values())
        # also by name - after a module reload the panel's own lists are empty, the nodes still exist
        for label in self._all_measurement_labels(panel):
            for suffix in (" axis", " sphere"):
                node = slicer.mrmlScene.GetFirstNodeByName(f"Doggybone {side} {label}{suffix}")
                if node is not None and node not in nodes:
                    nodes.append(node)
        return [n for n in nodes if n is not None and n.GetScene() is not None and n.GetDisplayNode() is not None]

    def _panel_measurements(self, panel):
        stack = getattr(panel, "measurement_stack", None)
        return [stack.widget(i).measurement for i in range(stack.count)] if stack is not None else []

    def _all_measurement_labels(self, panel):
        labels = set()
        for measurement in self._panel_measurements(panel):
            labels.update(getattr(measurement, "AXIS_LABELS", []) or [])
            labels.update(getattr(measurement, "SPHERE_LABELS", []) or [])
        return labels

    # --- compare mode: original (left 3D view) | corrected (right 3D view) ------------------------
    def _compare_mode(self):
        box = getattr(self, "compare_views_checkbox", None)
        return bool(box.checked) if box is not None else False

    def _compare_view_ids(self):
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None or layout_manager.threeDViewCount < 2:
            return None, None, []
        slice_ids = [layout_manager.sliceWidget(name).mrmlSliceNode().GetID()
                     for name in layout_manager.sliceViewNames() if layout_manager.sliceWidget(name) is not None]
        return (layout_manager.threeDWidget(0).mrmlViewNode().GetID(),
                layout_manager.threeDWidget(1).mrmlViewNode().GetID(), slice_ids)

    def _compare_node_sets(self, side, bone):
        """(original nodes, corrected nodes) of this side / bone for the two 3D views."""
        original = [slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)]
        original += list(self._original_overlay_nodes(side))
        if hasattr(self, "_cutting_plane_nodes"):
            original += list(self._cutting_plane_nodes())
        names = self._preview_node_names(side, bone)
        original.append(slicer.mrmlScene.GetFirstNodeByName(names[4]))          # the wedge to remove
        corrected = [slicer.mrmlScene.GetFirstNodeByName(names[i]) for i in (0, 1, 3)]
        prefix = self.CORRECTED_PREFIX.format(side=side)
        for class_name in ("vtkMRMLModelNode", "vtkMRMLMarkupsFiducialNode"):
            corrected += [n for n in slicer.util.getNodesByClass(class_name) if (n.GetName() or "").startswith(prefix)]
        keep = lambda nodes: [n for n in nodes if n is not None and n.GetDisplayNode() is not None]
        return keep(original), keep(corrected)

    def _compare_guide_sets(self):
        """(reduction guide nodes, saw guide nodes) - both hidden in the compare mode."""
        if not hasattr(self, "_guide_nodes_of_mode"):
            return [], []
        get = lambda names: [n for n in (slicer.mrmlScene.GetFirstNodeByName(name) for name in names)
                             if n is not None and n.GetDisplayNode() is not None]
        reduction = get(self._guide_nodes_of_mode("reduction"))
        saw = get(self._guide_nodes_of_mode("saw") + [getattr(self, "SAW_PINS_NAME", "")])
        saw = [n for n in saw if n not in reduction]
        return reduction, saw

    def _apply_compare_views(self):
        """Compare mode on: originals only in the left 3D view (+ slice views) and visible, corrected
        objects only in the right 3D view. Called again whenever the preview is redrawn."""
        if not self._compare_mode():
            return
        left_id, right_id, slice_ids = self._compare_view_ids()
        if left_id is None:
            return
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        original, corrected = self._compare_node_sets(side, bone)
        for node in original:
            display = node.GetDisplayNode()
            display.RemoveAllViewNodeIDs()
            for view_id in [left_id] + slice_ids:
                display.AddViewNodeID(view_id)
        for node in corrected:
            display = node.GetDisplayNode()
            display.RemoveAllViewNodeIDs()
            display.AddViewNodeID(right_id)
        # no guide in the comparison (neither saw nor reduction guide, in neither view); their own
        # visibility is untouched and applies again when compare mode is switched off
        reduction_guide, saw_guide = self._compare_guide_sets()
        for node in reduction_guide + saw_guide:
            display = node.GetDisplayNode()
            display.RemoveAllViewNodeIDs()
            display.AddViewNodeID("Doggybone compare - not shown")
        # the originals are shown again (in the left view), whatever the preview had hidden
        saved = self.__dict__.get("_hidden_original_overlay", {}).pop(side, None)
        for node_id, visible in (saved or {}).items():
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(visible)
        segmentation = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation is not None and segmentation.GetDisplayNode() is not None:
            segmentation.GetDisplayNode().SetVisibility(True)
            segmentation.GetDisplayNode().SetVisibility3D(True)
        slicer.util.forceRenderAllViews()

    def _clear_compare_views(self):
        """Every object back in all views."""
        for class_name in ("vtkMRMLModelNode", "vtkMRMLMarkupsNode", "vtkMRMLSegmentationNode"):
            for node in slicer.util.getNodesByClass(class_name):
                name = node.GetName() or ""
                if not name.startswith("Doggybone") and name != self.SEGMENTATION_NODE_NAME:
                    continue
                for i in range(node.GetNumberOfDisplayNodes()):
                    display = node.GetNthDisplayNode(i)
                    if display is not None:
                        display.RemoveAllViewNodeIDs()

    COMPARE_LAYOUT_ID = 1501
    COMPARE_LAYOUT_XML = (
        '<layout type="horizontal" split="true">'
        ' <item><view class="vtkMRMLViewNode" singletontag="1">'
        '<property name="viewlabel" action="default">1</property></view></item>'
        ' <item><view class="vtkMRMLViewNode" singletontag="2">'
        '<property name="viewlabel" action="default">2</property></view></item>'
        '</layout>')

    def _compare_layout(self):
        """Two large 3D views side by side, no slice views (registered once)."""
        layout_node = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if not layout_node.IsLayoutDescription(self.COMPARE_LAYOUT_ID):
            layout_node.AddLayoutDescription(self.COMPARE_LAYOUT_ID, self.COMPARE_LAYOUT_XML)
        return self.COMPARE_LAYOUT_ID

    @staticmethod
    def _two_3d_views_shown():
        """True if the current layout SHOWS two 3D views (Slicer keeps hidden view widgets around)."""
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None or layout_manager.threeDViewCount < 2:
            return False
        return all(layout_manager.threeDWidget(i).isVisible() for i in range(2))

    @staticmethod
    def _equalize_compare_split():
        """Both 3D views exactly half of the 3D area (the splitter otherwise keeps its last position)."""
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None or layout_manager.threeDViewCount < 2:
            return
        splitter = layout_manager.threeDWidget(0).parentWidget()
        if not isinstance(splitter, qt.QSplitter) or splitter.count() < 2:
            return
        half = max(1, splitter.width // splitter.count())
        splitter.setSizes([half] * splitter.count())

    def _compare_layout_changed(self):
        """Layout switched by hand while comparing: with only ONE 3D view left the comparison cannot be
        shown - compare mode off (normal preview display) instead of a view with only the originals."""
        layout_manager = slicer.app.layoutManager()
        box = getattr(self, "compare_views_checkbox", None)
        if box is None or not box.checked or getattr(self, "_compare_switching", False):
            return
        if not self._two_3d_views_shown():
            self._layout_before_compare = None          # keep the layout just chosen
            box.setChecked(False)
        else:
            if layout_manager.layout == self.COMPARE_LAYOUT_ID:
                self._equalize_compare_split()
            self._apply_compare_views()

    def onCompareViewsToggled(self, on):
        layout_manager = slicer.app.layoutManager()
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        if on:
            if layout_manager.layout != self.COMPARE_LAYOUT_ID:      # never remember the compare layout itself
                self._layout_before_compare = layout_manager.layout
            self._compare_switching = True
            try:
                layout_manager.setLayout(self._compare_layout())
                slicer.app.processEvents()
            finally:
                self._compare_switching = False
            self._equalize_compare_split()
            qt.QTimer.singleShot(100, self._equalize_compare_split)     # once the window has its final size
            view_nodes = [layout_manager.threeDWidget(i).mrmlViewNode() for i in range(2)]
            self._linked_before_compare = [v.GetLinkedControl() for v in view_nodes]
            for view in view_nodes:
                view.SetLinkedControl(True)              # both views turn / zoom together
            # same camera in the right view as in the left one
            cameras = [layout_manager.threeDWidget(i).threeDView().renderWindow().GetRenderers()
                       .GetFirstRenderer().GetActiveCamera() for i in range(2)]
            cameras[1].DeepCopy(cameras[0])
            self._apply_compare_views()
            combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
            if combined is None:
                self.correction_status.setText(
                    "Compare mode: left the original bone. The right view shows the corrected bone once "
                    "'Show/update preview' has run.")
            return
        self._clear_compare_views()
        if hasattr(self, "_guide_nodes_of_mode"):        # guides as the guide panel's mode says again
            mode = getattr(self, "_guide_mode", "saw")
            other = "saw" if mode == "reduction" else "reduction"
            for visible, names in ((False, self._guide_nodes_of_mode(other)), (True, self._guide_nodes_of_mode(mode))):
                for name in names:
                    node = slicer.mrmlScene.GetFirstNodeByName(name)
                    if node is not None and node.GetDisplayNode() is not None:
                        node.GetDisplayNode().SetVisibility(visible)
        for i, linked in enumerate(getattr(self, "_linked_before_compare", []) or []):
            if i < layout_manager.threeDViewCount:
                layout_manager.threeDWidget(i).mrmlViewNode().SetLinkedControl(bool(linked))
        before = getattr(self, "_layout_before_compare", None)
        if layout_manager.layout == self.COMPARE_LAYOUT_ID and (before is None or before == self.COMPARE_LAYOUT_ID):
            before = slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView      # never stay in the compare layout
        if before is not None and before != self.COMPARE_LAYOUT_ID:
            self._compare_switching = True
            try:
                layout_manager.setLayout(before)
            finally:
                self._compare_switching = False
        self._layout_before_compare = None
        # back to the normal preview display: originals hidden while a preview exists
        combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
        if combined is not None:
            self._hide_original_overlay(side)
        self._sync_original_segmentation()
        slicer.util.forceRenderAllViews()

    def _hide_original_overlay(self, side):
        """While the corrected landmarks are shown, hide the ORIGINAL (pre-correction) landmarks,
        spheres and axes of that side; their visibility is remembered and restored later."""
        if self._compare_mode():
            self._apply_compare_views()
            return
        store = self.__dict__.setdefault("_hidden_original_overlay", {})
        saved = store.setdefault(side, {})
        for node in self._original_overlay_nodes(side):
            display = node.GetDisplayNode()
            if node.GetID() not in saved:
                saved[node.GetID()] = bool(display.GetVisibility())
            display.SetVisibility(False)
        self._set_sphere_edit_visibility(False)

    def _set_sphere_edit_visibility(self, visible):
        """The circle / sphere and the radius handle of a landmark sphere being edited belong to the
        ORIGINAL landmarks - hidden together with them in the corrected (reduced) view."""
        for name in ("Doggybone sphere preview", "Doggybone sphere radius"):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None or node.GetDisplayNode() is None:
                continue
            show = bool(visible)
            if show and node.IsA("vtkMRMLMarkupsNode"):
                show = node.GetNumberOfControlPoints() > 0
            node.GetDisplayNode().SetVisibility(show)

    def _restore_original_overlay(self, side):
        saved = self.__dict__.get("_hidden_original_overlay", {}).pop(side, None)
        self._set_sphere_edit_visibility(True)
        for node_id, visible in (saved or {}).items():
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(visible)
        if saved is None:
            # nothing remembered (e.g. after a module reload): draw the original landmarks and the
            # axes / spheres of the active measurements again from the original landmark positions
            panel = self.left_panel if side == "left" else self.right_panel
            markups = slicer.mrmlScene.GetNodeByID(panel.markup_node_id) if panel is not None else None
            if markups is not None and markups.GetDisplayNode() is not None:
                markups.GetDisplayNode().SetVisibility(True)
            stack = getattr(panel, "measurement_stack", None)
            for i in range(stack.count if stack is not None else 0):
                widget = stack.widget(i)
                if getattr(widget, "enabled", False):
                    try:
                        widget.update_measurement()
                    except Exception:
                        logging.exception("Doggybone: redrawing a measurement failed (continuing)")

    def _remove_corrected_landmarks(self, side):
        self._restore_original_overlay(side)
        prefix = self.CORRECTED_PREFIX.format(side=side)
        for class_name in ("vtkMRMLModelNode", "vtkMRMLMarkupsFiducialNode"):
            for node in list(slicer.util.getNodesByClass(class_name)):
                if node.GetName().startswith(prefix):
                    slicer.mrmlScene.RemoveNode(node)
        status = self.correction_status.text
        if self.CORRECTED_STATUS_MARK in status:
            self.correction_status.setText(status.split(self.CORRECTED_STATUS_MARK)[0])

    def _corrected_model(self, name, color, opacity):
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetVisibility2D(False)
        node.GetDisplayNode().SetColor(*color)
        node.GetDisplayNode().SetOpacity(opacity)
        node.GetDisplayNode().SetVisibility(True)
        return node

    def _update_corrected_landmarks(self):
        """Landmarks, spheres and axes in the CORRECTED position: the landmarks on the moving side
        of the master cut are carried along with the correction (incl. fine shift), the others stay.
        The measurements are evaluated on these points (the original landmarks are never touched)."""
        import copy
        state = getattr(self, "_fine_shift_state", None)
        side = (state or {}).get("side") or self.osteotomy_side.currentText.lower()
        self._remove_corrected_landmarks(side)
        if not state:
            return
        bone = state["bone"]
        names = self._preview_node_names(side, bone)
        transform = slicer.mrmlScene.GetFirstNodeByName(names[2])
        panel = self.left_panel if side == "left" else self.right_panel
        if transform is None or panel is None:
            return
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)          # correction incl. fine shift
        H = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        shaft, _, _, error = self._get_anatomical_frame(side, bone)
        if error:
            return
        shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))
        n = osteotomy_logic.normalize(np.asarray(state["normal"], dtype=float))
        o = np.asarray(state["origin"], dtype=float)
        distal_sign = 1.0 if float(np.dot(n, shaft)) >= 0.0 else -1.0

        placed = {l.name: np.asarray(l.get_position(), dtype=float)
                  for l in getattr(panel, "landmarks", []) if getattr(l, "placed", False)}
        # cut between the shaft-axis landmarks: the values use the axis of the fragment that carries
        # the shaft (as the plan, see _split_shaft_axis); the drawing shows every point where it is
        split_names, axis_on_distal = self._split_shaft_axis(bone, placed, o, n, distal_sign)
        axis_moves = axis_on_distal == state["moving_is_distal"]
        points, measure_points, radii, moved_names = {}, {}, {}, []
        for landmark in getattr(panel, "landmarks", []):
            if not getattr(landmark, "placed", False):
                continue
            p = placed[landmark.name]
            moved = (H @ np.append(p, 1.0))[:3]
            on_distal_side = float(np.dot(p - o, n)) * distal_sign > 0.0
            if landmark.name in split_names:
                measure_points[landmark.name] = moved if axis_moves else p
            else:
                measure_points[landmark.name] = moved if on_distal_side == state["moving_is_distal"] else p
            if on_distal_side == state["moving_is_distal"]:
                p = moved
                moved_names.append(landmark.name)
            points[landmark.name] = p
            radii[landmark.name] = getattr(landmark, "radius", None)
        if not points:
            return

        prefix = self.CORRECTED_PREFIX.format(side=side)
        fiducials = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{prefix} landmarks")
        fiducials.SetLocked(True)
        fiducials.SetMarkupLabelFormat("")
        for name, p in points.items():
            index = fiducials.AddControlPoint(float(p[0]), float(p[1]), float(p[2]))
            fiducials.SetNthControlPointLabel(index, name)
            fiducials.SetNthControlPointLocked(index, True)
        display = fiducials.GetDisplayNode()
        display.SetSelectedColor(0.2, 1.0, 0.4)
        display.SetColor(0.2, 1.0, 0.4)
        display.SetTextScale(2.5)
        display.SetPointLabelsVisibility(not self._landmark_labels_hidden())
        display.SetGlyphScale(2.0)

        # which measurements to show in the corrected position: the planner's wedge / rotation
        # measurements plus every measurement whose ORIGINAL axes are visible right now (e.g. TPA with
        # its functional axis and plateau line) - otherwise those stay behind in the original position
        planner_names = [name for name in (self.WEDGE_MEASUREMENT_NAME.get(bone),
                                           self.ROTATION_MEASUREMENT_NAME.get(bone)) if name]
        shown_names = list(planner_names)
        for measurement in self._panel_measurements(panel):
            if measurement.name in shown_names:
                continue
            for label in getattr(measurement, "AXIS_LABELS", []) or []:
                node = slicer.mrmlScene.GetFirstNodeByName(f"Doggybone {side} {label} axis")
                if node is not None and node.GetDisplayNode() is not None and node.GetDisplayNode().GetVisibility():
                    shown_names.append(measurement.name)
                    break
        results, spheres, axes = [], {}, {}
        for bone_name in shown_names:
            measurement = self._get_measurement_by_name(panel, bone_name) if bone_name else None
            if measurement is None:
                continue
            needed = [l.name for l in measurement.get_landmarks()]
            if any(name not in measure_points for name in needed):
                continue
            corrected = copy.copy(measurement)          # own spheres/axes - the original stays as is
            try:
                value, description = corrected._measure(measure_points, radii)
            except Exception as exc:                     # a measurement must never break the preview
                results.append(f"{measurement.name}: not measurable ({exc})")
                continue
            if bone_name not in planner_names:
                results.append(f"{getattr(measurement, 'display_name', None) or measurement.name}: "
                               f"{value:.1f}\N{DEGREE SIGN} ({description})")
            elif bone_name == self.ROTATION_MEASUREMENT_NAME.get(bone):
                value = self._signed_rotation_value(bone, value, description)
                target = float(self.target_rotation_value.value)
                results.append(f"{measurement.name} {value:+.1f}\N{DEGREE SIGN} (target {target:+.1f}\N{DEGREE SIGN})")
            else:
                target = float(self.target_wedge_value.value)
                results.append(f"{description} {value:.1f}\N{DEGREE SIGN} (target {target:.1f}\N{DEGREE SIGN})")
            # the DRAWN axes / spheres follow the bone: every landmark where it really is after the
            # correction (also a shaft-axis landmark on the moved fragment); only the values above
            # use the axis of the fragment that carries the shaft when the cut splits it
            shown = corrected
            if measure_points is not points:
                shown = copy.copy(measurement)
                try:
                    shown._measure(points, radii)
                except Exception:
                    shown = corrected
            spheres.update(shown.get_spheres())
            for label, ends in shown.get_axes().items():
                axes.setdefault(label, ends)

        palette = [(0.2, 1.0, 0.4), (1.0, 0.85, 0.1), (1.0, 0.4, 0.2), (0.4, 0.7, 1.0)]
        for label, (center, radius) in spheres.items():
            if radius is None:
                continue
            source = vtk.vtkSphereSource()
            source.SetCenter(*[float(v) for v in center])
            source.SetRadius(max(float(radius), 0.01))
            source.SetThetaResolution(32)
            source.SetPhiResolution(32)
            source.Update()
            node = self._corrected_model(f"{prefix} {label} sphere", (0.2, 1.0, 0.4), 0.25)
            node.SetAndObservePolyData(source.GetOutput())
        for index, (label, (start, end)) in enumerate(axes.items()):
            line = vtk.vtkLineSource()
            line.SetPoint1(*[float(v) for v in start])
            line.SetPoint2(*[float(v) for v in end])
            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(1.0)
            tube.SetNumberOfSides(12)
            tube.CappingOn()
            tube.Update()
            node = self._corrected_model(f"{prefix} {label} axis", palette[index % len(palette)], 1.0)
            node.SetAndObservePolyData(tube.GetOutput())

        if results:
            self.correction_status.setText(self.correction_status.text + self.CORRECTED_STATUS_MARK +
                                           "; ".join(results) + f" [{len(moved_names)} landmarks moved]")

        self._hide_original_overlay(side)
        self._apply_segment_opacity()

    def _ensure_segmentation_visible(self):
        """Bone segmentation visible in 2D and 3D, the segment itself switched on, and a usable 3D
        opacity: the 'Opacity' slider value, but at least 10 % (at 0 % it is set back to 100 %)."""
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetDisplayNode() is None:
            return
        display_node = segmentation_node.GetDisplayNode()
        display_node.SetVisibility(True)
        if hasattr(display_node, "SetVisibility3D"):
            display_node.SetVisibility3D(True)
        if hasattr(display_node, "SetVisibility2D"):
            display_node.SetVisibility2D(True)
        segmentation = segmentation_node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(i)
            display_node.SetSegmentVisibility(segment_id, True)
            if display_node.GetSegmentOpacity3D(segment_id) < 0.1:
                display_node.SetSegmentOpacity3D(segment_id, 1.0)
        slider = getattr(self, "opacity_slider", None)
        if slider is not None and slider.value < 10:
            slider.setValue(100)                       # also applies the opacity (slider signal)
        opacity = (slider.value / 100.0) if slider is not None else 1.0
        if hasattr(display_node, "SetOpacity3D"):
            display_node.SetOpacity3D(max(opacity, 0.1))
        else:
            display_node.SetOpacity(max(opacity, 0.1))
        if hasattr(self, "_guide_saved_opacity"):
            self._guide_saved_opacity = []             # nothing left to restore to a hidden state
        self._sync_original_segmentation()            # a preview of this bone still hides it in 3D

    def onClearPreview(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        removed_any = False
        self._remove_fine_shift_observers()
        self._fine_shift_state = None
        self._remove_corrected_landmarks(side)
        self._set_cutting_plane_handles(side, bone, True)
        for name in self._preview_node_names(side, bone) + self._fine_shift_node_names(side, bone):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
                removed_any = True
        # without a preview the original bone must always be visible again - whatever hid it
        # (preview, guide tools, a module reload that lost the remembered state)
        self._ensure_segmentation_visible()
        self.onHideAnnotationsToggled(False)           # labels and cutting planes back
        self.correction_status.setText("Preview removed." if removed_any else "No preview to remove.")
