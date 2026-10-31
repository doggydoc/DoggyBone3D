# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Licensed under the MIT License, see LICENSE in the repository root.

"""CombinedCorrectionMixin: both corrections of a limb in ONE picture (Overview level).

Every bone's correction preview splits the whole limb at its own cut. Combined (proximal bone first,
both moving their DISTAL fragment):
  * above the upper cut              - fixed (yellow)
  * between the upper and lower cut  - moved by the upper correction (turquoise)
  * below the lower cut              - moved by the upper AND the lower correction (orange)
The landmarks are moved the same way and all measurements of both bones are evaluated on them.
Nothing of the single previews is changed; their objects are only hidden while the combined view is on.
"""

import copy

from DoggyboneUI.common import *


class CombinedCorrectionMixin:
    COMBINED_PREFIX = "Doggybone {side} combined correction"
    COMBINED_COLORS = ((0.95, 0.85, 0.35), (0.2, 0.8, 0.85), (1.0, 0.6, 0.25))

    def _combined_names(self, side):
        prefix = self.COMBINED_PREFIX.format(side=side)
        return (f"{prefix} fixed", f"{prefix} middle", f"{prefix} distal", f"{prefix} landmarks")

    def _correction_matrix(self, side, bone):
        transform = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[2])
        if transform is None:
            return None
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)          # correction incl. fine shift
        return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])

    def _cut_of(self, side, bone):
        """(origin, normal, distal_sign, moving_is_distal) of the bone's master cut, or None."""
        state = self._state_for(side, bone) if hasattr(self, "_state_for") else None
        if not state:
            return None
        shaft, _ml, _ap, error = self._get_anatomical_frame(side, bone)
        if error:
            return None
        shaft = osteotomy_logic.normalize(np.asarray(shaft, float))
        n = osteotomy_logic.normalize(np.asarray(state["normal"], float))
        o = np.asarray(state["origin"], float)
        return o, n, (1.0 if float(np.dot(n, shaft)) >= 0.0 else -1.0), bool(state["moving_is_distal"])

    @staticmethod
    def _transformed(polydata, H):
        matrix = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                matrix.SetElement(r, c, float(H[r][c]))
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        filt = vtk.vtkTransformPolyDataFilter()
        filt.SetInputData(polydata)
        filt.SetTransform(transform)
        filt.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(filt.GetOutput())
        return out

    @staticmethod
    def _proximal_part(polydata, origin, normal, distal_sign):
        """The part of polydata on the PROXIMAL side of the cut (origin, normal)."""
        plane = vtk.vtkPlane()
        plane.SetOrigin(*[float(v) for v in origin])
        plane.SetNormal(*[float(v) * distal_sign for v in normal])     # + = distal
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(polydata)
        clip.SetClipFunction(plane)
        clip.InsideOutOn()                                             # keep the negative (proximal) side
        clip.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(clip.GetOutput())
        return out

    def _combined_model(self, name, polydata, color):
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetVisibility2D(False)
        node.SetAndObservePolyData(polydata)
        display = node.GetDisplayNode()
        display.SetColor(*color)
        slider = getattr(self, "segment_opacity_slider", None)
        display.SetOpacity(slider.value / 100.0 if slider is not None else 1.0)
        display.SetVisibility(True)
        return node

    def show_combined_correction(self, side):
        """Build and show both corrections of the side in one picture. Returns (ok, text)."""
        names = list(self.BONES.names)
        if len(names) != 2:
            return False, "The combined view needs exactly two bones."
        upper, lower = names
        missing = [b for b in names if slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, b)[3]) is None]
        if missing:
            return False, (f"No correction preview of the {' and the '.join(b.lower() for b in missing)} ({side}) - "
                           "plan it in the osteotomy planning first ('Show/update preview').")
        H_u, H_l = self._correction_matrix(side, upper), self._correction_matrix(side, lower)
        cut_u, cut_l = self._cut_of(side, upper), self._cut_of(side, lower)
        if H_u is None or H_l is None or cut_u is None or cut_l is None:
            return False, "A correction (transform or cutting plane) is missing - update the preview of both bones."
        if not (cut_u[3] and cut_l[3]):
            return False, ("The combined view needs the DISTAL fragment moved in both bones "
                           "(reference plane 'Proximal').")
        p_u, d_u, _t, _c, _w = self._preview_node_names(side, upper)
        p_l, d_l, _t2, _c2, _w2 = self._preview_node_names(side, lower)
        fixed_node = slicer.mrmlScene.GetFirstNodeByName(p_u)
        middle_node = slicer.mrmlScene.GetFirstNodeByName(d_u)
        distal_node = slicer.mrmlScene.GetFirstNodeByName(d_l)
        if fixed_node is None or middle_node is None or distal_node is None:
            return False, "A fragment of the previews is missing - update the preview of both bones."
        H_both = H_u @ H_l
        fixed = vtk.vtkPolyData()
        fixed.DeepCopy(fixed_node.GetPolyData())
        middle = self._transformed(self._proximal_part(middle_node.GetPolyData(), cut_l[0], cut_l[1], cut_l[2]), H_u)
        distal = self._transformed(distal_node.GetPolyData(), H_both)
        model_names = self._combined_names(side)
        self._hide_for_combined(side)
        for name, polydata, color in zip(model_names[:3], (fixed, middle, distal), self.COMBINED_COLORS):
            self._combined_model(name, polydata, color)

        # landmarks moved like the bone they sit on
        panel = self.left_panel if side == "left" else self.right_panel
        points, radii, moved = {}, {}, {"upper": 0, "both": 0}
        for landmark in getattr(panel, "landmarks", []):
            if not getattr(landmark, "placed", False):
                continue
            p = np.asarray(landmark.get_position(), float)
            below_u = float(np.dot(p - cut_u[0], cut_u[1])) * cut_u[2] > 0.0
            below_l = float(np.dot(p - cut_l[0], cut_l[1])) * cut_l[2] > 0.0
            if below_u and below_l:
                p, key = (H_both @ np.append(p, 1.0))[:3], "both"
            elif below_u:
                p, key = (H_u @ np.append(p, 1.0))[:3], "upper"
            else:
                key = None
            if key:
                moved[key] += 1
            points[landmark.name] = p
            radii[landmark.name] = getattr(landmark, "radius", None)
        fiducials = slicer.mrmlScene.GetFirstNodeByName(model_names[3])
        if fiducials is not None:
            slicer.mrmlScene.RemoveNode(fiducials)
        fiducials = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", model_names[3])
        fiducials.SetLocked(True)
        fiducials.SetMarkupLabelFormat("")
        for name, p in points.items():
            index = fiducials.AddControlPoint(float(p[0]), float(p[1]), float(p[2]))
            fiducials.SetNthControlPointLabel(index, name)
            fiducials.SetNthControlPointLocked(index, True)
        display = fiducials.GetDisplayNode()
        display.SetColor(0.2, 1.0, 0.4)
        display.SetSelectedColor(0.2, 1.0, 0.4)
        display.SetGlyphScale(2.0)
        display.SetPointLabelsVisibility(False)

        # every measurement of both bones: before -> after both corrections
        lines = [f"<b>{side.capitalize()}: {upper} + {lower} corrected</b> "
                 f"({moved['upper']} landmarks moved with the {upper.lower()} correction, "
                 f"{moved['both']} with both)"]
        seen = []
        for bone in names:
            for mname in self.BONES.get(bone).measurements:
                if mname in seen:
                    continue
                seen.append(mname)
                measurement = self._get_measurement_by_name(panel, mname)
                if measurement is None:
                    continue
                needed = [l.name for l in measurement.get_landmarks()
                          if l.name not in getattr(measurement, "OPTIONAL_LANDMARKS", [])]
                if any(n not in points for n in needed):
                    continue
                label = getattr(measurement, "display_name", None) or mname
                try:
                    ok, before, desc_before = measurement()
                    after_m = copy.copy(measurement)
                    after, desc_after = after_m._measure(dict(points), radii)
                except Exception as exc:                 # one measurement must never break the view
                    lines.append(f"{label}: not measurable ({exc})")
                    continue
                before_text = f"{before:.1f}\N{DEGREE SIGN}" if ok else "-"
                lines.append(f"{label}: {before_text} \N{RIGHTWARDS ARROW} {after:.1f}\N{DEGREE SIGN} ({desc_after})")
        slicer.util.forceRenderAllViews()
        return True, "<br>".join(lines)

    def _hide_for_combined(self, side):
        """Single previews, corrected overlays, original landmarks and the bone segmentation give way."""
        if self.__dict__.get("_combined_hidden_side") == side:
            return
        nodes = []
        for bone in self.BONES.names:
            nodes += self._bone_preview_nodes(side, bone)
            nodes += self._corrected_overlay_nodes(side, bone)
        panel = self.left_panel if side == "left" else self.right_panel
        markups = slicer.mrmlScene.GetNodeByID(panel.markup_node_id) if panel is not None else None
        if markups is not None and markups.GetDisplayNode() is not None:
            nodes.append(markups)
        nodes += [n for n in slicer.util.getNodesByClass("vtkMRMLModelNode")
                  if (n.GetName() or "").startswith(f"Doggybone {side} ") and (n.GetName() or "").endswith((" axis", " sphere"))
                  and n.GetDisplayNode() is not None]
        segmentation = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation is not None and segmentation.GetDisplayNode() is not None:
            nodes.append(segmentation)
        self._stash_hide(("combined", side), nodes)
        self._combined_hidden_side = side

    def hide_combined_correction(self):
        """Remove the combined view and bring back what it hid."""
        for side in ("left", "right"):
            for name in self._combined_names(side):
                node = slicer.mrmlScene.GetFirstNodeByName(name)
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
            self._stash_restore(("combined", side))
        self._combined_hidden_side = None
        slicer.util.forceRenderAllViews()
