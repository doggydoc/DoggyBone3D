# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""SulcusPinMixin: guide pin for a sulcus (trochlear groove) prosthesis, built as one more sleeve into
the femoral saw guide.

The pin is always perpendicular to the condylar axis (medial -> lateral femur condyle) - in the frontal
plane it stays coupled to that axis. A plane gizmo carries it: the two arrows shift it in the frontal
plane (along the condylar axis and proximal / distal), the ring tilts it in the sagittal plane (rotation
about the condylar axis). Default: entry in the middle between the condyles on the cranial bone surface,
drilled cranial -> caudal."""

from DoggyboneUI.common import *


class SulcusPinMixin:
    SULCUS_DIAMETERS = ("Off", "1.6 mm", "2.0 mm", "2.4 mm", "3.2 mm")
    SULCUS_BONE = "Femur"

    def _sulcus_names(self, side):
        side = side.lower()
        return f"Doggybone {side} sulcus pin gizmo", f"Doggybone {side} sulcus pin"

    def _sulcus_diameter(self):
        box = getattr(self, "guide_sulcus_pin", None)
        if box is None or box.currentText == "Off":
            return None
        return float(box.currentText.split()[0])

    def _sulcus_side(self):
        return self.osteotomy_side.currentText

    def _sulcus_frame(self, side):
        """(condylar axis c, shaft y _|_ c, cranial z _|_ c, condyle midpoint) or an error text."""
        bone = self.SULCUS_BONE
        shaft, _ml, _ap, error = self._get_anatomical_frame(side.lower(), bone)
        if error:
            return None, error
        medial_name, lateral_name = self.ML_LANDMARKS[bone]
        medial = self._named_position(side.lower(), medial_name)
        lateral = self._named_position(side.lower(), lateral_name)
        if medial is None or lateral is None:
            return None, f"Place '{medial_name}' and '{lateral_name}' first."
        c = guide_logic.unit(np.asarray(lateral, float) - np.asarray(medial, float))
        y = np.asarray(shaft, float) - float(np.dot(shaft, c)) * c
        y = guide_logic.unit(y)
        cranial = np.cross(shaft, c) if side.lower() == "left" else np.cross(c, shaft)
        cranial = cranial - float(np.dot(cranial, c)) * c
        z = guide_logic.unit(cranial)
        y = guide_logic.unit(np.cross(z, c))                       # right-handed (c, y, z)
        return (c, y, z, 0.5 * (np.asarray(medial, float) + np.asarray(lateral, float))), None

    def _sulcus_bone(self, side):
        """Femur surface alone (no patella in front of the trochlea), else the whole segmentation."""
        polydata = None
        if hasattr(self, "_single_bone_polydata"):
            try:
                polydata, _note = self._single_bone_polydata(self.SULCUS_BONE, side)
            except Exception:
                polydata = None
        return polydata if polydata is not None else self._get_bone_polydata()

    def _on_sulcus_pin_changed(self, *_args):
        side = self._sulcus_side()
        gizmo_name, pin_name = self._sulcus_names(side)
        diameter = self._sulcus_diameter()
        if diameter is None:
            for name in (gizmo_name, pin_name):
                node = slicer.mrmlScene.GetFirstNodeByName(name)
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
            self.guide_status.setText("Sulcus pin removed.")
            return
        frame, error = self._sulcus_frame(side)
        if error:
            self.guide_status.setText(f"Sulcus pin: {error}")
            return
        gizmo = slicer.mrmlScene.GetFirstNodeByName(gizmo_name)
        if gizmo is None:
            c, y, z, mid = frame
            bone = self._sulcus_bone(side)
            obb = self._pin_obb_tree(bone)
            # start in the trochlea: from the level of the condyles move proximally until the pin runs
            # through solid bone (at the condyle level the ray falls into the intercondylar fossa)
            shaft = self._get_anatomical_frame(side.lower(), self.SULCUS_BONE)[0]
            proximal = guide_logic.unit(-(np.asarray(shaft, float) - float(np.dot(shaft, c)) * c))
            candidates = []
            for s in np.arange(0.0, 30.5, 1.0):
                p = mid + s * proximal
                hits = guide_logic.line_hits(obb, p + 60.0 * z, p - 60.0 * z)
                if len(hits):
                    candidates.append((float(np.dot(hits[0] - hits[-1], z)), np.asarray(hits[0], float)))
            if candidates:
                best = max(length for length, _e in candidates)
                entry = next(e for length, e in candidates if length >= 0.9 * best)
            else:
                entry = mid
            gizmo = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", gizmo_name)
            gizmo.CreateDefaultDisplayNodes()
            display = gizmo.GetDisplayNode()
            display.SetColor(0.85, 0.2, 0.85)
            display.SetSelectedColor(0.85, 0.2, 0.85)
            display.SetFillOpacity(0.1)
            display.SetPropertiesLabelVisibility(False)
            display.SetPointLabelsVisibility(False)
            display.SetHandlesInteractive(True)
            display.SetTranslationHandleVisibility(True)
            display.SetRotationHandleVisibility(True)
            display.SetScaleHandleVisibility(False)
            display.SetTranslationHandleComponentVisibility([True, True, False, False])   # frontal shift
            display.SetRotationHandleComponentVisibility([True, False, False, False])     # about the condylar axis
            display.SetInteractionHandleScale(3.0)
            self._sulcus_updating = True
            try:
                gizmo.SetAxesWorld([float(v) for v in c], [float(v) for v in y], [float(v) for v in z])
                gizmo.SetOriginWorld(*[float(v) for v in entry + 4.0 * z])
                gizmo.SetSize(8.0, 8.0)
            finally:
                self._sulcus_updating = False
            gizmo.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                              lambda caller, event: self._on_sulcus_gizmo_moved(final=False))
            gizmo.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                              lambda caller, event: self._on_sulcus_gizmo_moved(final=True))
        self._on_sulcus_gizmo_moved(final=True)

    def _sulcus_axes(self, gizmo):
        x, y, z = [0.0] * 3, [0.0] * 3, [0.0] * 3
        gizmo.GetAxesWorld(x, y, z)
        origin = [0.0] * 3
        gizmo.GetOriginWorld(origin)
        return np.asarray(origin, float), np.asarray(z, float)

    def _on_sulcus_gizmo_moved(self, final=False):
        """Keep the pin perpendicular to the condylar axis (only the sagittal tilt is free), rebuild the
        pin model and report its angles."""
        if getattr(self, "_sulcus_updating", False):
            return
        side = self._sulcus_side()
        gizmo_name, pin_name = self._sulcus_names(side)
        gizmo = slicer.mrmlScene.GetFirstNodeByName(gizmo_name)
        diameter = self._sulcus_diameter()
        if gizmo is None or diameter is None:
            return
        frame, error = self._sulcus_frame(side)
        if error:
            self.guide_status.setText(f"Sulcus pin: {error}")
            return
        c, y0, z0, _mid = frame
        origin, z = self._sulcus_axes(gizmo)
        z = z - float(np.dot(z, c)) * c                        # coupled to the condylar axis
        if np.linalg.norm(z) < 1e-6:
            z = z0
        z = guide_logic.unit(z)
        if float(np.dot(z, z0)) < 0.0:
            z = -z                                              # always drilled from cranial
        y = guide_logic.unit(np.cross(z, c))
        self._sulcus_updating = True
        try:
            gizmo.SetAxesWorld([float(v) for v in c], [float(v) for v in y], [float(v) for v in z])
        finally:
            self._sulcus_updating = False
        if not final:
            return
        pin = self._sulcus_pin_plan(float(self.guide_gap.value))
        node = slicer.mrmlScene.GetFirstNodeByName(pin_name)
        if pin is None:
            self.guide_status.setText("Sulcus pin: the pin axis misses the femur - move the gizmo onto the trochlea.")
            return
        a = np.asarray(pin["axis"], float)
        line = vtk.vtkLineSource()
        line.SetPoint1(*[float(v) for v in np.asarray(pin["far"], float) - 3.0 * a])
        line.SetPoint2(*[float(v) for v in np.asarray(pin["top"], float) + 8.0 * a])
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(0.5 * diameter)
        tube.SetNumberOfSides(20)
        tube.CappingOn()
        tube.Update()
        if node is None:
            node = slicer.modules.models.logic().AddModel(tube.GetOutput())
            node.SetName(pin_name)
            node.GetDisplayNode().SetColor(0.85, 0.2, 0.85)
        else:
            node.SetAndObservePolyData(tube.GetOutput())
        node.GetDisplayNode().SetVisibility(True)
        shaft = self._get_anatomical_frame(side.lower(), self.SULCUS_BONE)[0]
        to_shaft = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, shaft))), 0.0, 1.0))))
        # sagittal tilt against 'perpendicular to the shaft': + = the cranial (entry) end lies further distal
        tilt = float(np.degrees(np.arcsin(np.clip(float(np.dot(a, shaft)), -1.0, 1.0))))
        lean = ("perpendicular to the shaft" if abs(tilt) < 0.05 else
                f"entry end tilted {abs(tilt):.1f}° {'distally' if tilt > 0 else 'proximally'}")
        self._sulcus_report = dict(diameter=diameter, to_shaft=to_shaft, tilt=tilt, bone_length=pin["bone_length"])
        self.guide_status.setText(
            f"Sulcus pin Ø {diameter:.1f} mm: perpendicular to the condylar axis; "
            f"sagittal: {lean} (angle to the femoral shaft axis {to_shaft:.1f}°), "
            f"bone along the pin {pin['bone_length']:.1f} mm. "
            "Arrows: shift in the frontal plane, ring: tilt in the sagittal plane. Built into the saw guide "
            "with 'Extrude guide'.")

    def _sulcus_pin_plan(self, gap):
        """The sulcus pin as a guide pin dict (sleeve + bore in the saw guide), or None."""
        diameter = self._sulcus_diameter()
        if diameter is None or self.osteotomy_bone.currentText != self.SULCUS_BONE:
            return None
        side = self._sulcus_side()
        gizmo = slicer.mrmlScene.GetFirstNodeByName(self._sulcus_names(side)[0])
        if gizmo is None:
            return None
        origin, z = self._sulcus_axes(gizmo)
        obb = self._pin_obb_tree(self._sulcus_bone(side))
        pin = guide_logic.plan_pin(obb, origin, guide_logic.unit(z), diameter, gap)
        if "error" in pin:
            return None
        pin["label"], pin["segment"] = "S", "sulcus"
        return pin
