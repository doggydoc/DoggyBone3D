# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""ReportMixin: one PDF that sums up the planning of a limb - measurements, targets and calculated
corrections, the correction preview and the guides (state, key values, pins) of every bone of the
chosen side, with a picture of the 3D view."""

import datetime
import html
import tempfile

from DoggyboneUI.common import *


class ReportMixin:

    # ------------------------------------------------------------------ collecting
    @staticmethod
    def _report_plain(text):
        return re.sub(r"<[^>]+>", "", str(text or "")).replace("&nbsp;", " ").strip()

    def _report_widget_value(self, widget):
        """Display value of a field (None for buttons, sliders, pictures)."""
        if isinstance(widget, (qt.QDoubleSpinBox, qt.QSpinBox)):
            return widget.text
        if isinstance(widget, qt.QComboBox):
            return widget.currentText
        if isinstance(widget, qt.QLineEdit):
            return widget.text
        if isinstance(widget, qt.QCheckBox):
            return "yes" if widget.checked else "no"
        if isinstance(widget, qt.QLabel):
            return self._report_plain(widget.text)
        return None

    def _report_container_rows(self, widget):
        """Rows of a container widget (e.g. the tibia target table): [(label, value)] in layout order."""
        layout = widget.layout() if widget is not None else None
        rows = []
        if isinstance(layout, qt.QGridLayout):
            for r in range(layout.rowCount()):
                cells = []
                for c in range(layout.columnCount()):
                    item = layout.itemAtPosition(r, c)
                    child = item.widget() if item is not None else None
                    if child is None or not child.isVisible():
                        continue
                    value = self._report_widget_value(child)
                    if value:
                        cells.append(value)
                if cells:
                    rows.append((cells[0], " | ".join(cells[1:])))
        elif layout is not None:
            for i in range(layout.count()):
                child = layout.itemAt(i).widget()
                if child is not None and child.isVisible():
                    rows += self._report_container_rows(child) if child.layout() is not None else (
                        [(self._report_widget_value(child), "")] if self._report_widget_value(child) else [])
        return rows

    def _report_planning_rows(self):
        """The osteotomy section as shown: [(kind, label, value)] - kind 'head' for the section titles."""
        form = getattr(self, "_osteotomy_form_layout", None)
        rows = []
        if form is None:
            return rows
        for r in range(form.rowCount()):
            label_item = form.itemAt(r, qt.QFormLayout.LabelRole)
            field_item = form.itemAt(r, qt.QFormLayout.FieldRole) or form.itemAt(r, qt.QFormLayout.SpanningRole)
            widget = field_item.widget() if field_item is not None else None
            if widget is None or widget.isHidden():
                continue
            label = label_item.widget().text if label_item is not None and label_item.widget() else ""
            if label in ("Status", "Bone opacity"):
                continue
            if isinstance(widget, qt.QLabel):
                text = widget.text
                plain = self._report_plain(text)
                if not plain:
                    continue
                rows.append(("head", plain, "") if text.strip().startswith("<b>") else ("text", label, plain))
                continue
            value = self._report_widget_value(widget)
            if value is not None and label:
                rows.append(("value", label, value))
            elif value is None and widget.layout() is not None and not isinstance(widget, qt.QPushButton):
                for row_label, row_value in self._report_container_rows(widget):
                    if row_label and row_value and "Export" not in row_label and "planning" not in row_label.lower():
                        rows.append(("value", row_label, row_value))
        return rows

    def _report_preview(self, side, bone):
        """Movement of the fragment in the correction preview: (rotation deg, translation mm) or None."""
        transform = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side.lower(), bone)[2])
        if transform is None:
            return None
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)
        M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        angle = float(np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
        # shift AT THE OSTEOTOMY: how far the centre of the moved fragment's cut moves (the translation
        # part of the matrix alone depends on the distance to the coordinate origin and means nothing)
        shift = None
        for name in (self._distal_plane_node_name(side.lower(), bone), self._plane_node_name(side.lower(), bone)):
            plane = slicer.mrmlScene.GetFirstNodeByName(name)
            if plane is not None:
                origin = np.asarray(self._get_plane_origin_normal(plane)[0], float)
                shift = float(np.linalg.norm(M[:3, :3] @ origin + M[:3, 3] - origin))
                break
        return angle, shift

    def _report_measurements(self, side, bone):
        panel = self._panel_for_side(side)
        rows, seen = [], set()
        for name in bone.measurements:
            if name in seen:
                continue
            seen.add(name)
            measurement = self._get_measurement_by_name(panel, name)
            if measurement is None:
                continue
            try:
                ready, value, description = measurement()
            except Exception:
                ready, value, description = False, None, ""
            label = getattr(measurement, "display_name", name)
            rows.append((label, f"{value:.1f}\N{DEGREE SIGN}" if ready else "landmarks missing",
                         self._report_plain(description) if ready else ""))
        return rows

    def _report_guides(self, side, bone):
        guides = []
        for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if (node.GetAttribute("Doggybone.GuideBone") != bone or node.GetAttribute("Doggybone.GuideSide") != side
                    or node.GetPolyData() is None or node.GetPolyData().GetNumberOfPoints() == 0):
                continue
            mode = node.GetAttribute("Doggybone.GuideMode") or "saw"
            try:
                changed = self._guide_changed_parts(node, side, bone, mode)
            except Exception:
                changed = None
            state = ("up to date" if changed == [] else
                     f"OUTDATED ({', '.join(changed)} changed)" if changed else "built")
            problems = [p for p in (node.GetAttribute("Doggybone.Problems") or "").split("\n") if p.strip()]
            bounds = node.GetPolyData().GetBounds()
            guides.append(dict(name="Reduction guide" if mode == "reduction" else "Osteotomy (saw) guide",
                               state=state, summary=node.GetAttribute("Doggybone.Summary") or "",
                               problems=problems,
                               size=f"{bounds[1] - bounds[0]:.0f} x {bounds[3] - bounds[2]:.0f} x {bounds[5] - bounds[4]:.0f} mm"))
        guides.sort(key=lambda g: g["name"])
        pins = []
        try:
            for pin in (self._load_planned_pins() or []):
                if "error" in pin:
                    continue
                length = float(np.linalg.norm(np.asarray(pin["far"], float) - np.asarray(pin["entry"], float)))
                pins.append((pin.get("label", ""), pin.get("segment", ""), f"{float(pin['diameter']):.1f} mm",
                             f"{length:.1f} mm"))
        except Exception:
            pass
        try:                                     # sulcus prosthesis pin (femur, saw guide)
            sulcus = self._sulcus_pin_plan(float(self.guide_gap.value)) if hasattr(self, "_sulcus_pin_plan") else None
            info = getattr(self, "_sulcus_report", None) or {}
            if sulcus is not None:
                tilt = info.get("tilt")
                lean = "" if tilt is None else (
                    ", perpendicular to the shaft" if abs(tilt) < 0.05 else
                    f", entry end tilted {abs(tilt):.1f}\N{DEGREE SIGN} {'distally' if tilt > 0 else 'proximally'}")
                depth = info.get("cut_depth")
                cut = f", cut plane {depth:.1f} mm below the entry" if depth is not None else ""
                pins.append(("S", "MaMaMia trochlear prosthesis (_|_ condylar axis" + lean + cut + ")",
                             f"{float(sulcus['diameter']):.1f} mm", f"{float(sulcus['bone_length']):.1f} mm"))
        except Exception:
            pass
        return guides, pins

    def _report_guide_settings(self):
        rows = []
        for label, (_text, field) in getattr(self, "_guide_field_labels", {}).items():
            if field is None or field.isHidden():
                continue
            value = self._report_widget_value(field)
            if value:
                rows.append((label, value))
        return rows

    def _report_picture(self):
        """PNG of the 3D view (temporary file) or None."""
        try:
            view = slicer.app.layoutManager().threeDWidget(0).threeDView()
            view.forceRender()
            image = ctk.ctkWidgetsUtils.grabWidget(view)
            path = os.path.join(tempfile.gettempdir(), "doggybone_report_view.png")
            image.save(path)
            return path
        except Exception:
            return None

    def _report_bone_has_data(self, side, bone):
        panel = self._panel_for_side(side)
        for name in bone.measurements:
            measurement = self._get_measurement_by_name(panel, name)
            try:
                if measurement is not None and measurement()[0]:
                    return True
            except Exception:
                pass
        if self._report_preview(side, bone.name) is not None:
            return True
        return any(n.GetAttribute("Doggybone.GuideBone") == bone.name and n.GetAttribute("Doggybone.GuideSide") == side
                   for n in slicer.util.getNodesByClass("vtkMRMLModelNode"))

    # ------------------------------------------------------------------ document
    def _report_html(self, side, sections, settings, picture):
        e = html.escape
        name = self.patient_name.text.strip() if getattr(self, "patient_name", None) else ""
        number = self.patient_number.text.strip() if getattr(self, "patient_number", None) else ""
        out = ["<html><body style='font-family: Helvetica, Arial; font-size: 9.5pt;'>",
               "<h2 style='margin-bottom:2px'>DoggyBone3D - planning report</h2>",
               f"<p>Patient: <b>{e(name) or '-'}</b> &nbsp; No.: <b>{e(number) or '-'}</b> &nbsp; "
               f"Side: <b>{e(side)}</b> &nbsp; Date: {datetime.date.today().strftime('%d.%m.%Y')}</p>",
               "<p style='color:#7a2e00'><i>Planning document generated from the CT - all values have to be checked "
               "by the surgeon. Not a diagnosis.</i></p>"]
        table = "<table border='1' cellspacing='0' cellpadding='3' style='border-collapse: collapse; width:100%'>"
        for bone, data in sections:
            out.append(f"<h3 style='background:#e3eaf2; padding:3px'>{e(bone)}</h3>")
            out.append("<h4>Measurements</h4>" + table)
            out.append("<tr><th align='left'>Measurement</th><th align='left'>Value</th><th align='left'>Note</th></tr>")
            for label, value, note in data["measurements"]:
                out.append(f"<tr><td>{e(label)}</td><td><b>{e(value)}</b></td><td>{e(note)}</td></tr>")
            out.append("</table>")
            out.append("<h4>Correction planning</h4>" + table)
            for kind, label, value in data["planning"]:
                if kind == "head":
                    out.append(f"<tr><td colspan='2' style='background:#f2f2f2'><b>{e(label)}</b></td></tr>")
                elif kind == "text":
                    out.append(f"<tr><td colspan='2'>{e(value).replace(chr(10), '<br>')}</td></tr>")
                else:
                    out.append(f"<tr><td width='40%'>{e(label)}</td><td><b>{e(value)}</b></td></tr>")
            preview = data["preview"]
            if preview is not None:
                shift = f", the cut centre moves {preview[1]:.1f} mm" if preview[1] is not None else ""
                out.append(f"<tr><td>Preview: moved fragment</td><td><b>{preview[0]:.1f}\N{DEGREE SIGN} rotation in total"
                           f"{shift}</b></td></tr>")
            if data.get("status"):
                out.append(f"<tr><td colspan='2'><i>{e(data['status']).replace(chr(10), '<br>')}</i></td></tr>")
            out.append("</table>")
            out.append("<h4>Guides</h4>")
            if not data["guides"]:
                out.append("<p>No guide built.</p>")
            else:
                out.append(table + "<tr><th align='left'>Guide</th><th align='left'>State</th>"
                                   "<th align='left'>Key values</th><th align='left'>Size</th></tr>")
                for g in data["guides"]:
                    extra = "".join(f"<br><span style='color:#9b1c1c'>{e(p)}</span>" for p in g["problems"])
                    out.append(f"<tr><td>{e(g['name'])}</td><td>{e(g['state'])}</td><td>{e(g['summary'])}{extra}</td>"
                               f"<td>{e(g['size'])}</td></tr>")
                out.append("</table>")
            if data["pins"]:
                out.append("<h4>Pins (saw guide)</h4>" + table +
                           "<tr><th align='left'>Pin</th><th align='left'>Fragment</th><th align='left'>Ø</th>"
                           "<th align='left'>In bone</th></tr>")
                for row in data["pins"]:
                    out.append("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in row) + "</tr>")
                out.append("</table>")
        if settings:
            out.append("<h3 style='background:#e3eaf2; padding:3px'>Guide settings</h3>" + table)
            for label, value in settings:
                out.append(f"<tr><td width='40%'>{e(label)}</td><td>{e(value)}</td></tr>")
            out.append("</table>")
        if picture:
            out.append("<p style='background:#e3eaf2; padding:3px; font-size:13pt'><b>3D view</b><br>"
                       "<img src='doggybone-view' width='460'></p>")
        out.append("</body></html>")
        return "".join(out)

    # ------------------------------------------------------------------ export
    def onExportPlanningReport(self, status=None):
        """PDF with all measurements, corrections and guides of the current side (every bone with data)."""
        if not hasattr(status, "setText"):
            status = getattr(self, "guide_status", None) or self.correction_status
        side = self.limb_side.currentText if getattr(self, "limb_side", None) is not None else \
            self.osteotomy_side.currentText
        default_name = os.path.splitext(self._export_file_name("sx" if side == "Left" else "dx", "report"))[0] + ".pdf"
        start_dir = getattr(self, "_last_stl_dir", None) or qt.QDir.homePath()
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Export planning report",
                                              os.path.join(start_dir, default_name), "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        picture = self._report_picture()
        original_bone = self.osteotomy_bone.currentText
        sections = []
        with slicer.util.tryWithErrorDisplay("The planning report could not be written.", waitCursor=True):
            try:
                for bone in self.BONES:
                    if not self._report_bone_has_data(side, bone):
                        continue
                    if self.osteotomy_bone.currentText != bone.name:
                        self.osteotomy_bone.setCurrentText(bone.name)       # its targets / corrections
                        slicer.app.processEvents()
                    guides, pins = self._report_guides(side, bone.name)
                    status_text = getattr(self, "correction_status", None)
                    preview = self._report_preview(side, bone.name)
                    sections.append((bone.name, dict(
                        measurements=self._report_measurements(side, bone),
                        planning=self._report_planning_rows(),
                        preview=preview,
                        status=(self._report_plain(getattr(status_text, "full_text", status_text.text))
                                if status_text is not None and preview is not None else ""),
                        guides=guides, pins=pins)))
            finally:
                if self.osteotomy_bone.currentText != original_bone:
                    self.osteotomy_bone.setCurrentText(original_bone)
                    slicer.app.processEvents()
            document = qt.QTextDocument()
            if picture:
                document.addResource(qt.QTextDocument.ImageResource, qt.QUrl("doggybone-view"), qt.QImage(picture))
            document.setHtml(self._report_html(side, sections, self._report_guide_settings(), picture))
            printer = qt.QPrinter(qt.QPrinter.HighResolution)
            printer.setOutputFormat(qt.QPrinter.PdfFormat)
            printer.setPageSize(qt.QPageSize(qt.QPageSize.A4))
            printer.setPageMargins(qt.QMarginsF(15, 15, 15, 15), qt.QPageLayout.Millimeter)
            printer.setOutputFileName(path)
            document.print_(printer) if hasattr(document, "print_") else document.print(printer)
            self._last_stl_dir = os.path.dirname(path)
            status.setText(f"Planning report saved: {path}\n({len(sections)} bone(s): "
                           f"{', '.join(name for name, _d in sections) or 'no data'})")
