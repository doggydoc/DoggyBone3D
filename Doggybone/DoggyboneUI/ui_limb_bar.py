# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""LimbBarMixin: one bar at the top of the module - side and level (each bone of the limb, Overview).

It replaces the separate side selectors of the measurements and the osteotomy and the bone
selector of the osteotomy: those combo boxes still exist (all planning code reads them), but are
hidden and set from the bar. A bone level shows only that bone's measurements and plans its
osteotomy and guides; Overview shows every measurement, a table of all values and the export of
all guides.

Uses attributes created in DoggyboneWidget.setup() and methods of the other mixins."""

from DoggyboneUI.common import *


OVERVIEW = "Overview"


class LimbBarMixin:

    # ------------------------------------------------------------------ build
    def _build_limb_bar(self):
        """Create the bar; call at the very start of setup()."""
        bar = qt.QWidget()
        form = qt.QFormLayout(bar)
        form.setContentsMargins(0, 0, 0, 4)
        self.limb_side = qt.QComboBox()
        self.limb_side.addItems(["Right", "Left"])
        self.limb_side.setToolTip("Side of the limb - measurements, osteotomy and guides all follow it.")
        form.addRow("Side:", self.limb_side)

        row = qt.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.limb_level_buttons = {}
        self._limb_level_group = qt.QButtonGroup()
        self._limb_level_group.setExclusive(True)
        for level in list(self.BONES.names) + [OVERVIEW]:
            button = qt.QPushButton(level)
            button.setCheckable(True)
            button.setToolTip("Overview: all measurements, all values, export of all guides." if level == OVERVIEW
                              else f"Measure and plan the {level.lower()}.")
            self._limb_level_group.addButton(button)
            row.addWidget(button)
            self.limb_level_buttons[level] = button
        level_widget = qt.QWidget()
        level_widget.setLayout(row)
        form.addRow("Level:", level_widget)
        self.limb_bar = bar
        self.layout.addWidget(bar)

    def _finish_limb_bar(self):
        """Hide the old selectors, add the overview box and wire everything; call at the end of setup()."""
        # the former selectors stay as hidden state holders
        for box, layout in ((getattr(self, "osteotomy_side", None), getattr(self, "_osteotomy_form_layout", None)),
                            (getattr(self, "osteotomy_bone", None), getattr(self, "_osteotomy_form_layout", None))):
            if box is None:
                continue
            box.setVisible(False)
            label = layout.labelForField(box) if layout is not None else None
            if label is not None:
                label.setVisible(False)
        if getattr(self, "side_selector", None) is not None:
            self.side_selector.setVisible(False)
        if getattr(self, "_measurement_side_label", None) is not None:
            self._measurement_side_label.setVisible(False)

        # overview box (only shown on the Overview level)
        box = ctk.ctkCollapsibleButton()
        box.text = "Overview"
        box.setStyleSheet("ctkCollapsibleButton { background-color: #2f855a; color: white; font-weight: bold; }")
        layout = qt.QVBoxLayout(box)
        self.overview_values = qt.QLabel("")
        self.overview_values.setWordWrap(True)
        self.overview_values.setTextFormat(qt.Qt.RichText)
        self.overview_values.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        layout.addWidget(self.overview_values)
        refresh = qt.QPushButton("Refresh values")
        refresh.clicked.connect(lambda: self._refresh_overview())
        export = qt.QPushButton("Export all guides")
        export.setToolTip("Save every finished guide of this patient (all bones, osteotomy and reduction guide).")
        export.clicked.connect(self._on_overview_export)
        buttons = qt.QHBoxLayout()
        buttons.addWidget(refresh)
        buttons.addWidget(export)
        layout.addLayout(buttons)
        self.overview_status = StatusLabel("")
        self.overview_status.setWordWrap(True)
        layout.addWidget(self.overview_status)
        self.overview_box = box
        self.layout.addWidget(box)

        # start from the current state of the (hidden) selectors
        side = self.osteotomy_side.currentText if getattr(self, "osteotomy_side", None) is not None else "Right"
        blocked = self.limb_side.blockSignals(True)
        self.limb_side.setCurrentText(side)
        self.limb_side.blockSignals(blocked)
        bone = self.osteotomy_bone.currentText if getattr(self, "osteotomy_bone", None) is not None else None
        self.limb_level_buttons.get(bone, self.limb_level_buttons[self.BONES.default.name]).setChecked(True)

        self.limb_side.currentIndexChanged.connect(lambda _i: self._on_limb_side_changed())
        for button in self.limb_level_buttons.values():
            button.connect("clicked()", self._on_limb_level_changed)
        # code that still sets the hidden selectors directly (tests, scene restore): the bar follows
        if getattr(self, "osteotomy_side", None) is not None:
            self.osteotomy_side.currentIndexChanged.connect(lambda _i: self._sync_bar_from_selectors())
        if getattr(self, "osteotomy_bone", None) is not None:
            self.osteotomy_bone.currentIndexChanged.connect(lambda _i: self._sync_bar_from_selectors())
        self._on_limb_side_changed()
        self._on_limb_level_changed()

    # ------------------------------------------------------------------ state
    def current_limb_level(self):
        for level, button in getattr(self, "limb_level_buttons", {}).items():
            if button.isChecked():
                return level
        return self.BONES.default.name

    def _on_limb_side_changed(self):
        side = self.limb_side.currentText
        for name in ("osteotomy_side", "side_selector"):
            box = getattr(self, name, None)
            if box is not None and box.currentText != side:
                box.setCurrentText(side)
        if self.current_limb_level() == OVERVIEW:
            self._refresh_overview()

    def _on_limb_level_changed(self, *_args):
        level = self.current_limb_level()
        overview = level == OVERVIEW
        if not overview and getattr(self, "osteotomy_bone", None) is not None \
                and self.osteotomy_bone.currentText != level:
            self.osteotomy_bone.setCurrentText(level)          # -> _on_osteotomy_bone_changed
        for name in ("_osteotomy_collapsible", "_guide_collapsible"):
            box = getattr(self, name, None)
            if box is not None:
                box.setVisible(not overview)
        if getattr(self, "overview_box", None) is not None:
            self.overview_box.setVisible(overview)
        names = None if overview else self.BONES.get(level).measurements
        for panel in (getattr(self, "left_panel", None), getattr(self, "right_panel", None)):
            if panel is not None and hasattr(panel, "set_visible_measurements"):
                panel.set_visible_measurements(names, active=panel is self._current_panel())
        self._show_level_tools(names)
        if overview:
            self._refresh_overview()

    def _show_level_tools(self, names):
        """Tools that belong to some measurements only: shown for the level whose measurements use them
        (e.g. tools that only one bone's measurements use)."""
        panel = self._current_panel() if hasattr(self, "_current_panel") else None
        if panel is None:
            return
        stack = panel.measurement_stack
        measurements = [stack.widget(i).measurement for i in range(stack.count)]
        if names is not None:
            measurements = [m for m in measurements if m.name in names]
        used = {landmark.name for m in measurements for landmark in getattr(m, "landmarks", [])}
        box = getattr(self, "condyle_circle_box", None)
        if box is not None:
            show = names is not None and any(n in used for n in self.CONDYLE_CIRCLE_LANDMARKS)
            box.setVisible(show)
            if hasattr(self, "_schedule_condyle_circles"):
                self._schedule_condyle_circles()
        with_spheres = any(not str(label).startswith("CORA")          # CORA markers are no fitted spheres
                           for m in measurements for label in (getattr(m, "SPHERE_LABELS", None) or []))
        for each in (getattr(self, "left_panel", None), getattr(self, "right_panel", None)):
            checkbox = getattr(each, "show_spheres_checkbox", None)
            if checkbox is not None:
                checkbox.setVisible(with_spheres)

    def _sync_bar_from_selectors(self):
        side = self.osteotomy_side.currentText
        if self.limb_side.currentText != side:
            self.limb_side.setCurrentText(side)               # -> _on_limb_side_changed (no-op for equal)
        bone = self.osteotomy_bone.currentText
        level = self.current_limb_level()
        if level != OVERVIEW and level != bone and bone in self.limb_level_buttons:
            self.limb_level_buttons[bone].setChecked(True)
            self._on_limb_level_changed()

    # ------------------------------------------------------------------ overview
    def _refresh_overview(self):
        if getattr(self, "overview_values", None) is None:
            return
        side = self.limb_side.currentText
        panel = self._panel_for_side(side)
        deg = "\N{DEGREE SIGN}"
        rows, seen = [], set()
        for bone in self.BONES:
            rows.append(f"<tr><td colspan='2'><b>{bone.name}</b></td></tr>")
            for name in bone.measurements:
                if name in seen:
                    continue                                  # shared measurement: listed once
                seen.add(name)
                measurement = self._get_measurement_by_name(panel, name)
                if measurement is None:
                    continue
                try:
                    ready, value, description = measurement()
                except Exception:
                    ready, value, description = False, None, ""
                label = getattr(measurement, "display_name", name)
                text = f"{value:.1f}{deg} {description}" if ready else "<i>landmarks missing</i>"
                rows.append(f"<tr><td>{label}</td><td>&nbsp;{text}</td></tr>")
        self.overview_values.setText(f"<p><b>{side} {self.LIMB_LABEL}</b></p><table>" + "".join(rows) + "</table>")

    def _on_overview_export(self):
        self.onExportAllGuides()
        status = getattr(self, "guide_status", None)
        if status is not None:
            self.overview_status.setText(status.text)
