# Doggybone 3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""TibiaTargetsMixin: tibial osteotomy with several current / target values.

Rows: TPA, aMPTA (proximal varus/valgus), aMDTA (distal varus/valgus), tibial torsion,
tibia/metatarsus torsion. Each row shows the measured value, an editable target, a 'use'
checkbox and the value predicted after the planned correction.

The correction of the moving fragment has three rotations - frontal (about the anatomical AP
axis), sagittal (about the ML axis) and torsion (about the shaft axis). They are solved together
(damped Gauss-Newton, least squares) so that the measurements on the MOVED landmarks hit the
checked targets as well as possible; unchecked values are only predicted. A small regularisation
keeps rotations that no checked target asks for at ~0.
"""

from DoggyboneUI.common import *


TIBIA_TARGET_ROWS = [
    # key, label, default target, used by default, tooltip
    ("tpa", "TPA", 6.0, False,
     "Tibial plateau angle (medial plateau vs. perpendicular to the functional axis). + = slopes caudally."),
    ("ampta", "aMPTA (proximal)", measurement_logic.TIBIA_MPTA_REF_DEG, True,
     "Proximal joint line vs. the anatomical (shaft) axis - proximal varus (<) / valgus (>)."),
    ("amdta", "aMDTA (distal)", measurement_logic.TIBIA_MDTA_REF_DEG, True,
     "Distal joint line vs. the anatomical (shaft) axis - distal varus (<) / valgus (>)."),
    ("torsion", "Torsion tibia", 0.0, True,
     "Tibial torsion (proximal condylar axis vs. cochlea axis). + = internal rotation."),
    ("tma", "Torsion tibia/metatarsus", 0.0, False,
     "Tibial metatarsal angle: foot axis vs. proximal condylar axis. + = inward."),
]


class TibiaTargetsMixin:

    # ------------------------------------------------------------------ UI
    def _build_tibia_targets(self, form_layout):
        self.tibia_targets_widget = qt.QWidget()
        grid = qt.QGridLayout(self.tibia_targets_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        for column, title in enumerate(["", "Current", "Target", "Use", "Planned"]):
            label = qt.QLabel(f"<b>{title}</b>")
            grid.addWidget(label, 0, column)
        self.tibia_target_rows = {}
        for row, (key, label, default, used, tip) in enumerate(TIBIA_TARGET_ROWS, start=1):
            name = qt.QLabel(label)
            name.setToolTip(tip)
            current = qt.QLabel("--")
            target = qt.QDoubleSpinBox()
            target.setRange(-90.0, 180.0)
            target.setDecimals(1)
            target.setSingleStep(0.5)
            target.setSuffix(" °")
            target.setValue(float(default))
            use = qt.QCheckBox()
            use.setChecked(bool(used))
            use.setToolTip("Checked: the correction is solved for this target. Unchecked: only predicted.")
            planned = qt.QLabel("--")
            grid.addWidget(name, row, 0)
            grid.addWidget(current, row, 1)
            grid.addWidget(target, row, 2)
            grid.addWidget(use, row, 3)
            grid.addWidget(planned, row, 4)
            target.valueChanged.connect(lambda _v: self._on_tibia_targets_changed())
            use.toggled.connect(lambda _c: self._on_tibia_targets_changed())
            self.tibia_target_rows[key] = {"current": current, "target": target, "use": use, "planned": planned}
        refresh = qt.QPushButton("Update current values")
        refresh.clicked.connect(self._refresh_tibia_current)
        grid.addWidget(refresh, len(TIBIA_TARGET_ROWS) + 1, 0, 1, 5)
        form_layout.addRow(self.tibia_targets_widget)
        self._restore_tibia_targets()
        self.calculated_sagittal = qt.QDoubleSpinBox()
        self.calculated_sagittal.setRange(-180.0, 180.0)
        self.calculated_sagittal.setSuffix(" °")
        self.calculated_sagittal.setReadOnly(True)
        self.calculated_sagittal.setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
        self.calculated_sagittal.setToolTip("Sagittal part of the correction (about the medio-lateral axis).")
        return self.calculated_sagittal

    # targets are kept in the scene (survive a module reload and are saved with the scene)
    TIBIA_TARGETS_NODE = "Doggybone tibia targets"

    def _tibia_targets_node(self, create=False):
        node = slicer.mrmlScene.GetFirstNodeByName(self.TIBIA_TARGETS_NODE)
        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", self.TIBIA_TARGETS_NODE)
            node.SetHideFromEditors(True)
        return node

    def _save_tibia_targets(self):
        if getattr(self, "_restoring_tibia_targets", False):
            return
        node = self._tibia_targets_node(create=True)
        for key, row in self.tibia_target_rows.items():
            node.SetParameter(f"{key}.target", str(float(row["target"].value)))
            node.SetParameter(f"{key}.use", "1" if row["use"].checked else "0")

    def _restore_tibia_targets(self):
        node = self._tibia_targets_node()
        if node is None:
            return
        self._restoring_tibia_targets = True
        try:
            for key, row in self.tibia_target_rows.items():
                value, use = node.GetParameter(f"{key}.target"), node.GetParameter(f"{key}.use")
                for widget, setter, arg in ((row["target"], row["target"].setValue, float(value) if value else None),
                                            (row["use"], row["use"].setChecked, use == "1" if use else None)):
                    if arg is None:
                        continue
                    was = widget.blockSignals(True)
                    setter(arg)
                    widget.blockSignals(was)
        finally:
            self._restoring_tibia_targets = False


    def _tibia_mode(self):
        return (getattr(self, "tibia_target_rows", None) is not None
                and self.osteotomy_bone.currentText == "Tibia")

    def _tibia_targets(self):
        """{key: (target, used)}"""
        return {key: (float(row["target"].value), bool(row["use"].checked))
                for key, row in self.tibia_target_rows.items()}

    def _on_tibia_targets_changed(self):
        if not self._tibia_mode():
            return
        self._save_tibia_targets()
        targets = self._tibia_targets()
        self._tibia_solution_fresh = False            # the shown sagittal value is an estimate until solved
        # keep the classic two target fields consistent (used for the start tilt of the second
        # plane and the status texts): frontal = aMPTA + aMDTA - 180, torsion = tibial torsion
        for widget, value in ((self.target_wedge_value, targets["ampta"][0] + targets["amdta"][0] - 180.0),
                              (self.target_rotation_value, targets["torsion"][0])):
            was = widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(was)
        # recalculate the shown corrections (frontal / rotation / sagittal) from the new targets
        if hasattr(self, "_on_target_values_changed"):
            self._on_target_values_changed()
        else:
            self._refresh_tibia_current()
        # a correction whose target is not checked ('Use') is not planned - show 0
        if not (targets["ampta"][1] or targets["amdta"][1]) and hasattr(self, "calculated_wedge"):
            self.calculated_wedge.setValue(0.0)
        if not (targets["torsion"][1] or targets.get("tma", (0, False))[1]) and hasattr(self, "rotation_angle"):
            self.rotation_angle.setValue(0.0)

    def _set_tibia_mode_visible(self, tibia):
        widget = getattr(self, "tibia_targets_widget", None)
        if widget is None:
            return
        widget.setVisible(tibia)
        layout = getattr(self, "_osteotomy_form_layout", None)
        for field, show in ((self.target_wedge_value, not tibia), (self.target_rotation_value, not tibia),
                            (self.calculated_sagittal, tibia)):
            field.setVisible(show)
            if layout is not None:
                label = layout.labelForField(field)
                if label is not None:
                    label.setVisible(show)
        if tibia:
            self._on_tibia_targets_changed()

    # ------------------------------------------------------------------ values
    def _tibia_metrics(self, panel):
        """Measured tibial values {key: signed value} from the current landmark positions."""
        values = {}
        vv = self._get_measurement_by_name(panel, "Varus Valgus Tibia")
        if vv is not None:
            ready, _value, _description = vv()
            angles = getattr(vv, "_angles", None)
            if ready and angles is not None:
                values["ampta"], values["amdta"] = float(angles[2]), float(angles[3])
        tpa = self._get_measurement_by_name(panel, "Tibial Plateau Angle")
        if tpa is not None:
            ready, value, description = tpa()
            if ready:
                values["tpa"] = float(value) if "caudally" in description else -float(value)
        torsion = self._get_measurement_by_name(panel, "Tibia Torsion")
        if torsion is not None:
            ready, value, description = torsion()
            if ready:
                values["torsion"] = float(self._signed_rotation_value("Tibia", value, description))
        tma = self._get_measurement_by_name(panel, "Tibial Metatarsal Angle")
        if tma is not None:
            ready, value, description = tma()
            if ready:
                values["tma"] = float(value) if description.startswith("Inward") else -float(value)
        return values

    def _refresh_tibia_current(self):
        if getattr(self, "tibia_target_rows", None) is None or not hasattr(self, "left_panel"):
            return
        side = self.osteotomy_side.currentText.lower()
        panel = self.left_panel if side == "left" else self.right_panel
        values = self._tibia_metrics(panel)
        deg = "\N{DEGREE SIGN}"
        for key, row in self.tibia_target_rows.items():
            if key in values:
                row["current"].setText(f"{values[key]:+.1f}{deg}" if key in ("tpa", "torsion", "tma")
                                       else f"{values[key]:.1f}{deg}")
                row["current"].setStyleSheet(OSTEOTOMY_VALUE_STYLE)
            else:
                row["current"].setText("landmarks?")
                row["current"].setStyleSheet(STATUS_WARN_STYLE)
        # Sagittal (TPA): before the correction is solved (master plane, 'Update planning') show the
        # estimate = current TPA - target TPA (a sagittal wedge changes the TPA by about its angle)
        field = getattr(self, "calculated_sagittal", None)
        if field is not None and not getattr(self, "_tibia_solution_fresh", False):
            target, used = self._tibia_targets().get("tpa", (0.0, False))
            if used and "tpa" in values:
                field.setValue(float(values["tpa"]) - float(target))
                field.setSuffix(" \N{DEGREE SIGN} (est.)")
                field.setToolTip("Estimate: current TPA - target TPA. The exact value is solved when the "
                                 "correction is planned ('Update planning' with the master plane).")
            else:
                field.setValue(0.0)
                field.setSuffix(" \N{DEGREE SIGN}")

    def _show_tibia_planned(self, predicted):
        if getattr(self, "tibia_target_rows", None) is None:
            return
        values = (predicted or {}).get("tibia") or {}
        deg = "\N{DEGREE SIGN}"
        for key, row in self.tibia_target_rows.items():
            if key not in values:
                row["planned"].setText("--")
                row["planned"].setStyleSheet("")
                continue
            target, used = float(row["target"].value), bool(row["use"].checked)
            off = abs(values[key] - target)
            row["planned"].setText(f"{values[key]:+.1f}{deg}" if key in ("tpa", "torsion", "tma")
                                   else f"{values[key]:.1f}{deg}")
            row["planned"].setStyleSheet(STATUS_OK_STYLE if (not used or off < 1.0) else STATUS_WARN_STYLE)

    # ------------------------------------------------------------------ solver
    def _solve_tibia_correction(self, side, bone, moving_is_distal, plane_origin, plane_normal):
        """(R 3x3, info) for the tibia: frontal, sagittal and torsion solved together on the
        landmarks so the checked targets are met (least squares)."""
        shaft, ml, ap, frame_error = self._get_anatomical_frame(side, bone)
        if frame_error:
            return None, frame_error
        shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))
        ap = osteotomy_logic.normalize(np.asarray(ap, dtype=float))
        ml = osteotomy_logic.normalize(np.asarray(ml, dtype=float))
        targets = self._tibia_targets()
        side_panel = self.left_panel if side == "left" else self.right_panel
        current = self._tibia_metrics(side_panel)
        used = [key for key, (_t, use) in targets.items() if use]
        missing = [key for key in used if key not in current]
        if not used:
            return None, "Tibia: check at least one target ('Use')."
        if missing:
            labels = {key: label for key, label, *_ in TIBIA_TARGET_ROWS}
            return None, ("Tibia: landmarks missing for " + ", ".join(labels[k] for k in missing)
                          + " - place them or uncheck 'Use'.")
        pivot = np.asarray(plane_origin, dtype=float)

        def rot3(axis, angle):
            return self._rotation_about_line(axis, angle, np.zeros(3))[:3, :3]

        def rotation(x):
            swing = rot3(ml, x[1]) @ rot3(ap, x[0])
            twist = rot3(shaft, x[2])
            return twist @ swing if moving_is_distal else swing @ twist

        def as_h(R):
            H = np.eye(4, dtype=float)
            H[:3, :3] = R
            H[:3, 3] = pivot - R @ pivot
            return H

        regularisation = 0.05
        # a rotation is only free when a checked target asks for it: frontal <- aMPTA / aMDTA,
        # sagittal <- TPA, torsion <- torsion tibia / tibia-metatarsus. Otherwise it stays 0, so
        # e.g. the TPA is never changed as a side effect of a varus correction.
        free = np.array([("ampta" in used) or ("amdta" in used), "tpa" in used,
                         ("torsion" in used) or ("tma" in used)])

        def residual(x):
            predicted = self._predict_measurements(side, bone, as_h(rotation(x)), plane_origin, plane_normal,
                                                   moving_is_distal)
            values = (predicted or {}).get("tibia") or {}
            if any(key not in values for key in used):
                return None, predicted
            r = [values[key] - targets[key][0] for key in used]
            r += [regularisation * float(v) for v in x]
            return np.array(r, dtype=float), predicted

        x = np.zeros(3)
        r, predicted = residual(x)
        columns = [j for j in range(3) if free[j]]
        if r is None:
            return None, "Tibia: the measurements could not be evaluated."
        h = 0.5
        for _ in range(20):
            if np.max(np.abs(r[:len(used)])) < 0.05:
                break
            J = np.zeros((len(r), 3))
            for j in columns:
                dx = np.zeros(3)
                dx[j] = h
                rj, _p = residual(x + dx)
                if rj is None:
                    break
                J[:, j] = (rj - r) / h
            step = np.zeros(3)
            solved, *_ = np.linalg.lstsq(J[:, columns], -r, rcond=None)
            step[columns] = solved
            step = np.clip(step, -15.0, 15.0)
            for damping in (1.0, 0.5, 0.25, 0.1):
                r_new, p_new = residual(x + damping * step)
                if r_new is not None and float(r_new @ r_new) < float(r @ r):
                    x, r, predicted = x + damping * step, r_new, p_new
                    break
            else:
                break
        R = rotation(x)
        predicted = self._predict_measurements(side, bone, as_h(R), plane_origin, plane_normal, moving_is_distal)
        off = {key: (predicted or {}).get("tibia", {}).get(key, float("nan")) - targets[key][0] for key in used}
        warning = ""
        misses = [f"{key} {value:+.1f}\N{DEGREE SIGN}" for key, value in off.items() if abs(value) >= 1.0]
        if misses:
            warning = ("Not all checked targets can be reached with one osteotomy (remaining deviation: "
                       + ", ".join(misses) + ") - the plan is the best compromise; uncheck the less important target.")
        self._tibia_last_solution = x.copy()
        self._show_tibia_planned(predicted)
        if hasattr(self, "calculated_sagittal"):
            # shown as the TPA change the plan achieves (current - planned), sign as the estimate
            tpa_now = current.get("tpa")
            tpa_planned = (predicted or {}).get("tibia", {}).get("tpa")
            shown = (float(tpa_now) - float(tpa_planned)) if (tpa_now is not None and tpa_planned is not None) \
                else float(x[1])
            self._tibia_solution_fresh = True
            self.calculated_sagittal.setSuffix(" \N{DEGREE SIGN}")
            self.calculated_sagittal.setToolTip("TPA change of the planned correction (current - planned TPA); "
                                                "positive = TPA reduced.")
            self.calculated_sagittal.setValue(shown)
        return R, {"frontal": abs(float(x[0])), "torsion": abs(float(x[2])), "predicted": predicted,
                   "frontal_signed": float(x[0]), "sagittal_signed": float(x[1]), "torsion_signed": float(x[2]),
                   "warning": warning}
