# Doggybone 3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""PlanesMixin: Osteotomy planes: master/derived/distal cutting planes, anatomical frame, hinge point, plane sync.

Part of DoggyboneWidget (see Doggybone.py). Methods were moved here unchanged; they rely on
attributes created in DoggyboneWidget.setup() and on methods of the other mixins.
"""

from DoggyboneUI.common import *


class PlanesMixin:

    # Every node this module creates is looked up by name in the scene rather than cached on
    # the widget instance (self.xxx = ...). This makes the whole osteotomy/correction workflow
    # safe to use with Slicer's "Reload" (slicer.util.reloadScriptedModule) - a reload creates a
    # brand-new widget object with none of the old instance state, but the scene (and therefore
    # anything looked up by name) is untouched by a module reload.
    def _plane_node_name(self, side, bone):
        return f"Doggybone {side} {bone.lower()} proximal plane"

    def _get_plane_origin_normal(self, plane_node):
        # Mirrors the Set.../SetWorld... version check in onInsertCuttingPlane, for the Get side.
        if hasattr(plane_node, "GetOriginWorld") and hasattr(plane_node, "GetNormalWorld"):
            origin = [0.0, 0.0, 0.0]
            normal = [0.0, 0.0, 0.0]
            plane_node.GetOriginWorld(origin)
            plane_node.GetNormalWorld(normal)
        else:
            origin = list(plane_node.GetOrigin())
            normal = list(plane_node.GetNormal())
        return np.array(origin), np.array(normal)

    # Purely an internal starting point for the proximal plane when it's first inserted - not a
    # user-facing setting. Both planes are meant to be positioned entirely by hand from there.
    DEFAULT_PLANE_LEVEL_FRACTION = 0.4

    def _get_plane_frame(self, plane_node, side=None, bone=None):
        """Return the plane's ACTUAL world-space (origin, x, y, normal) frame.

        Crucially, x/y are read from the Markups Plane node when the Slicer API
        provides them.  This preserves in-plane twist/torsion.  Reconstructing x/y
        from the condylar vector is only a compatibility fallback for older Slicer
        versions and is not used when the true plane axes are available.
        """
        origin, normal = self._get_plane_origin_normal(plane_node)
        normal = osteotomy_logic.normalize(normal)

        for method_name in ("GetAxesWorld", "GetAxes"):
            method = getattr(plane_node, method_name, None)
            if method is None:
                continue
            try:
                axes = vtk.vtkMatrix3x3()
                method(axes)
                x = np.array([axes.GetElement(r, 0) for r in range(3)], dtype=float)
                y = np.array([axes.GetElement(r, 1) for r in range(3)], dtype=float)
                z = np.array([axes.GetElement(r, 2) for r in range(3)], dtype=float)
                if np.linalg.norm(x) > 1e-6 and np.linalg.norm(y) > 1e-6 and np.linalg.norm(z) > 1e-6:
                    x = osteotomy_logic.normalize(x)
                    z = osteotomy_logic.normalize(z)
                    # Re-orthogonalize to suppress small numerical drift.
                    x = osteotomy_logic.normalize(x - np.dot(x, z) * z)
                    y = osteotomy_logic.normalize(np.cross(z, x))
                    if np.dot(z, normal) < 0:
                        z, y = -z, -y
                    return origin, x, y, z
            except Exception:
                logging.exception("Doggybone: non-critical step failed (continuing)")

        ml_vector = self._get_ml_vector(side, bone) if side is not None and bone is not None else None
        if ml_vector is not None:
            _, x, y = osteotomy_logic.plane_frame(normal, ml_vector)
        else:
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, normal)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            x = osteotomy_logic.normalize(helper - np.dot(helper, normal) * normal)
            y = osteotomy_logic.normalize(np.cross(normal, x))
        return origin, x, y, normal

    def _set_plane_frame(self, plane_node, origin, x_axis, y_axis, normal):
        """Set a complete plane pose, including its in-plane torsional orientation."""
        origin = np.asarray(origin, dtype=float)
        x_axis = osteotomy_logic.normalize(x_axis)
        normal = osteotomy_logic.normalize(normal)
        x_axis = osteotomy_logic.normalize(x_axis - np.dot(x_axis, normal) * normal)
        y_axis = osteotomy_logic.normalize(np.cross(normal, x_axis))
        if hasattr(plane_node, "SetOriginWorld"):
            plane_node.SetOriginWorld(*origin)
        elif hasattr(plane_node, "SetOrigin"):
            plane_node.SetOrigin(*origin)
        if hasattr(plane_node, "SetAxes"):
            plane_node.SetAxes(vtk.vtkVector3d(*x_axis), vtk.vtkVector3d(*y_axis), vtk.vtkVector3d(*normal))
        elif hasattr(plane_node, "SetNormalWorld"):
            plane_node.SetNormalWorld(*normal)
        else:
            plane_node.SetNormal(*normal)

    def _get_anatomical_axis(self, side, bone):
        '''
        The bone's true proximal->distal shaft axis and a point on it at a fixed default level -
        from the reference landmarks alone, independent of where the cutting planes actually end
        up once manually positioned. Used to define the rotation (derotation/torsion) correction
        axis, and as the proximal plane's initial position, so adapting the planes by hand
        doesn't change what "derotation" means for the case - that stays tied to the real bone
        anatomy.
        Returns (origin, shaft_axis_unit, error_message); the first two are None on failure.
        '''
        panel = self.left_panel if side == "left" else self.right_panel
        landmarks_by_name = {l.name: l for l in panel.landmarks}

        if bone == "Femur":
            proximal_name, distal_name = "Femur midpoint 30%", "Femur midpoint 50%"
        else:
            proximal_name, distal_name = "proximal tibia midpoint", "distal tibia midpoint"

        proximal_landmark = landmarks_by_name.get(proximal_name)
        distal_landmark = landmarks_by_name.get(distal_name)
        if proximal_landmark is None or distal_landmark is None or not proximal_landmark.placed or not distal_landmark.placed:
            return None, None, f"Place '{proximal_name}' and '{distal_name}' ({side}) first"

        proximal_pos = np.array(proximal_landmark.get_position())
        distal_pos = np.array(distal_landmark.get_position())
        origin = proximal_pos + self.DEFAULT_PLANE_LEVEL_FRACTION * (distal_pos - proximal_pos)
        shaft_vector = distal_pos - proximal_pos
        shaft_length = np.linalg.norm(shaft_vector)
        if shaft_length < 1e-6:
            return None, None, "Reference points are coincident - cannot define an axis"
        return origin, shaft_vector / shaft_length, None

    def _get_anatomical_frame(self, side, bone):
        '''
        The bone's true (shaft, ml, ap) anatomical frame from the reference/condyle landmarks
        alone - independent of how the patient/limb happened to be positioned in the CT scanner.
        Used by onInsertDistalPlane to pre-angle the distal plane's starting orientation.
        Returns (shaft, ml, ap, error_message) - the first three are None on failure.
        '''
        origin, shaft_axis, error = self._get_anatomical_axis(side, bone)
        if error:
            return None, None, None, error

        panel = self.left_panel if side == "left" else self.right_panel
        landmarks_by_name = {l.name: l for l in panel.landmarks}
        medial_name, lateral_name = self.ML_LANDMARKS[bone]
        medial_landmark = landmarks_by_name.get(medial_name)
        lateral_landmark = landmarks_by_name.get(lateral_name)
        if (medial_landmark is None or lateral_landmark is None
                or not medial_landmark.placed or not lateral_landmark.placed):
            return None, None, None, f"Place '{medial_name}' and '{lateral_name}' ({side}) first."

        medial_pos = np.array(medial_landmark.get_position())
        lateral_pos = np.array(lateral_landmark.get_position())
        ml_vector = lateral_pos - medial_pos

        try:
            shaft, ml, ap = osteotomy_logic.plane_frame(shaft_axis, ml_vector)
        except ValueError as e:
            return None, None, None, f"Invalid geometry: {e}"

        return shaft, ml, ap, None

    def _get_condyle_width(self, side, bone):
        '''
        Distance between the medial and lateral condyle landmarks (the same ML reference used
        for the correction frame) - a fixed, anatomy-based bone-width reference at the condylar
        level, independent of the segmentation or the cutting plane's current position. Returns
        None if those landmarks aren't placed yet.
        '''
        ml_vector = self._get_ml_vector(side, bone)
        return None if ml_vector is None else float(np.linalg.norm(ml_vector))

    def _get_ml_vector(self, side, bone):
        '''
        The (unnormalized) medial->lateral condyle vector, or None if those landmarks aren't
        placed yet. Used both for _get_condyle_width and to set a plane's initial in-plane
        twist so it starts aligned with the condylar axis (see _configure_plane_size) - without
        this, only the plane's normal is set on insert, and VTK picks an arbitrary in-plane
        rotation around that normal with no relation to the anatomy, which is why the plane used
        to need a manual twist correction in addition to any frontal/sagittal tilt.
        '''
        panel = self.left_panel if side == "left" else self.right_panel
        landmarks_by_name = {l.name: l for l in panel.landmarks}
        medial_name, lateral_name = self.ML_LANDMARKS[bone]
        medial_landmark = landmarks_by_name.get(medial_name)
        lateral_landmark = landmarks_by_name.get(lateral_name)
        if (medial_landmark is None or lateral_landmark is None
                or not medial_landmark.placed or not lateral_landmark.placed):
            return None
        medial_pos = np.array(medial_landmark.get_position())
        lateral_pos = np.array(lateral_landmark.get_position())
        return lateral_pos - medial_pos

    def _configure_plane_size(self, plane_node, side, bone, color=(1.0, 0.9, 0.0), label=None):
        '''
        Sizes a cutting-plane markup to ~2x the condylar bone width (see _get_condyle_width)
        rather than an arbitrary fixed constant, and locks its scale/resize handles so that size
        can't be dragged out of that reference by accident - translation and rotation handles
        stay interactive, since both planes are meant to be positioned/oriented entirely by hand.
        Falls back to a fixed 150mm default, with a note in the return value, if the condyle
        landmarks aren't placed yet - the plane still needs a size either way.
        If given, `label` is shown as a short on-screen text at the plane (e.g. "Proximal") -
        separate from the node's own (long, lookup-only) Name, which stays unchanged so
        GetFirstNodeByName keeps working everywhere else in this module.
        Returns True if the anatomical width was available, False if the fallback was used.
        '''
        condyle_width = self._get_condyle_width(side, bone)
        size = 2.0 * condyle_width if condyle_width else 150.0
        if hasattr(plane_node, "SetSize"):
            plane_node.SetSize(size, size)

        # Align the plane's in-plane twist (rotation about its own normal) with the condylar ML
        # axis, so its edges start parallel/perpendicular to it - only the normal was being set
        # before, and VTK then picks an arbitrary in-plane rotation with no anatomical meaning,
        # which is why the plane used to need a manual twist correction in addition to whatever
        # frontal/sagittal tilt was actually wanted. This uses the plane's OWN current normal
        # (not the idealized shaft direction), so it works the same for an already-tilted plane
        # (e.g. the distal plane's starting guess) as for a fresh perpendicular one.
        ml_vector = self._get_ml_vector(side, bone)
        if ml_vector is not None and hasattr(plane_node, "SetAxes"):
            _, plane_normal = self._get_plane_origin_normal(plane_node)
            try:
                _, ml_axis, ap_axis = osteotomy_logic.plane_frame(plane_normal, ml_vector)
                plane_node.SetAxes(
                    vtk.vtkVector3d(*ml_axis),
                    vtk.vtkVector3d(*ap_axis),
                    vtk.vtkVector3d(*plane_normal))
            except Exception:
                pass  # SetAxes not available/compatible on this Slicer version - skip silently

        if label is not None and hasattr(plane_node, "SetNthControlPointLabel") \
                and plane_node.GetNumberOfControlPoints() > 0:
            plane_node.SetNthControlPointLabel(0, label)

        display_node = plane_node.GetDisplayNode()
        if display_node is not None and hasattr(display_node, "SetScaleHandleVisibility"):
            display_node.SetScaleHandleVisibility(False)
        # Rotation/translation handle VISIBILITY (the small draggable points at the border) is
        # always on for both planes - whether they're currently INTERACTIVE (draggable at all)
        # is controlled separately and centrally by _update_plane_handle_interactivity, based on
        # the "Active plane" selector, so only one plane can be accidentally dragged at a time.
        if display_node is not None and hasattr(display_node, "SetRotationHandleVisibility"):
            display_node.SetRotationHandleVisibility(True)
        if display_node is not None and hasattr(display_node, "SetTranslationHandleVisibility"):
            display_node.SetTranslationHandleVisibility(True)
        # Prefer the plain per-point label (just the short text, e.g. "Proximal") over
        # PropertiesLabelVisibility, which tends to render a more verbose info box (often
        # including coordinates) - that extra box was the "hanging text window" cluttering the
        # view. A smaller TextScale keeps it compact rather than a large floating label.
        if display_node is not None and hasattr(display_node, "SetPropertiesLabelVisibility"):
            display_node.SetPropertiesLabelVisibility(False)
        if display_node is not None and hasattr(display_node, "SetPointLabelsVisibility"):
            display_node.SetPointLabelsVisibility(label is not None)
        if display_node is not None and hasattr(display_node, "SetTextScale"):
            display_node.SetTextScale(2.5)

        # Show the plane's intersection with the Red/Yellow/Green slice views as a thick,
        # bright line - that's where the bone cross-section and cortex are actually easy to
        # judge by eye (unlike the 3D view), and the line is directly draggable there in
        # Slicer's normal interaction mode. This is the primary way to position/align each
        # plane by hand - no extra button needed, drag it directly in whichever 2D view shows
        # the bone clearly.
        # SetSliceIntersectionVisibility is deprecated (and its old wrapper appears to hang in
        # this Slicer version) in favor of SetVisibility2D - prefer that, only falling back to
        # the deprecated name on older Slicer builds that don't have SetVisibility2D at all.
        if display_node is not None:
            if hasattr(display_node, "SetVisibility2D"):
                display_node.SetVisibility2D(True)
            elif hasattr(display_node, "SetSliceIntersectionVisibility"):
                display_node.SetSliceIntersectionVisibility(True)
            if hasattr(display_node, "SetSliceIntersectionThickness"):
                display_node.SetSliceIntersectionThickness(3)
            if hasattr(display_node, "SetColor"):
                display_node.SetColor(*color)  # e.g. yellow for proximal, orange for distal

        self._update_ml_markers(plane_node, side, bone)

        return condyle_width is not None

    def _update_ml_markers(self, plane_node, side, bone):
        '''
        Places small "M"/"L" text markers at the plane's medial/lateral edge, along its own
        local x-axis (which the block above aligns with the condylar ML direction) - without
        this, the plane's spatial orientation (which way is medial, which is lateral) isn't
        actually readable at a glance, even though the edges themselves are now anatomically
        aligned: a square looks the same rotated 180° around its own center, so the ALIGNMENT
        alone doesn't tell you which end is which - only a label does.
        Recomputed from the plane's CURRENT origin/size/orientation every time this runs (called
        from _configure_plane_size and onSyncOtherPlane) - stale if the plane is moved/resized by
        hand afterward without either of those running again.
        '''
        # M/L markers switched off: only remove any left over from earlier versions.
        for suffix in (" L marker", " M marker"):
            node = slicer.mrmlScene.GetFirstNodeByName(f"{plane_node.GetName()}{suffix}")
            while node is not None:
                slicer.mrmlScene.RemoveNode(node)
                node = slicer.mrmlScene.GetFirstNodeByName(f"{plane_node.GetName()}{suffix}")
        return
        ml_vector = self._get_ml_vector(side, bone)
        if ml_vector is None:
            return
        origin, plane_normal = self._get_plane_origin_normal(plane_node)
        try:
            _, ml_axis, _ = osteotomy_logic.plane_frame(plane_normal, ml_vector)
        except ValueError:
            return

        size = 150.0
        if hasattr(plane_node, "GetSize"):
            try:
                size = float(plane_node.GetSize()[0])
            except Exception:
                logging.exception("Doggybone: non-critical step failed (continuing)")
        half_edge = size / 2.0 * 0.9  # slightly inside the true edge so the letter doesn't clip

        node_name = plane_node.GetName()
        for suffix, offset_sign, text in ((" L marker", 1.0, "L"), (" M marker", -1.0, "M")):
            marker_name = f"{node_name}{suffix}"
            marker_node = slicer.mrmlScene.GetFirstNodeByName(marker_name)
            if marker_node is None:
                marker_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", marker_name)
                marker_node.CreateDefaultDisplayNodes()
                marker_node.SetMaximumNumberOfControlPoints(1)
                display_node = marker_node.GetDisplayNode()
                if display_node is not None:
                    display_node.SetColor(1.0, 1.0, 1.0)
                    display_node.SetSelectedColor(1.0, 1.0, 1.0)
                    display_node.SetGlyphScale(1.0)
                    if hasattr(display_node, "SetPropertiesLabelVisibility"):
                        display_node.SetPropertiesLabelVisibility(False)
                    if hasattr(display_node, "SetPointLabelsVisibility"):
                        display_node.SetPointLabelsVisibility(True)
                    if hasattr(display_node, "SetTextScale"):
                        display_node.SetTextScale(4.0)

            position = origin + ml_axis * (half_edge * offset_sign)
            if marker_node.GetNumberOfControlPoints() == 0:
                marker_node.AddControlPoint(position[0], position[1], position[2])
            else:
                marker_node.SetNthControlPointPosition(0, position[0], position[1], position[2])
            if hasattr(marker_node, "SetNthControlPointLabel"):
                marker_node.SetNthControlPointLabel(0, text)

    def _create_plane_node_at_anatomical_axis(self, side, bone, node_name, label, color=None):
        existing = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if existing is not None:
            return existing, False, f"{label} plane already exists; it was left untouched."
        origin, shaft_axis, error = self._get_anatomical_axis(side, bone)
        if error:
            return None, False, error

        # Anatomical default orientation for a newly created MASTER plane.
        # X is EXACTLY medial->lateral (the condylar axis), rather than merely a
        # projection of that axis onto an arbitrary plane.  The plane normal is
        # chosen as close as possible to the shaft direction while remaining
        # perpendicular to X.  Therefore the initial plane actually CONTAINS the
        # condylar axis and its X edge is reproducibly parallel to it.
        ml_vector = self._get_ml_vector(side, bone)
        if ml_vector is None or np.linalg.norm(ml_vector) < 1e-6:
            return None, False, "Place the medial and lateral condyle landmarks first."
        x_axis = osteotomy_logic.normalize(ml_vector)
        normal = np.asarray(shaft_axis, dtype=float) - np.dot(shaft_axis, x_axis) * x_axis
        if np.linalg.norm(normal) < 1e-6:
            return None, False, "Shaft and condylar axes are nearly parallel - cannot initialize the master plane."
        normal = osteotomy_logic.normalize(normal)
        y_axis = osteotomy_logic.normalize(np.cross(normal, x_axis))

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", node_name)
        node.CreateDefaultDisplayNodes()
        self._set_plane_frame(node, origin, x_axis, y_axis, normal)
        kwargs = {"label": label}
        if color is not None:
            kwargs["color"] = color
        self._configure_plane_size(node, side, bone, **kwargs)
        return node, True, f"{label} master plane created. Position and orient it manually, then create the second plane."

    def onInsertMasterPlane(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        distal = self.reference_plane_selector.currentText.startswith("Distal")
        node_name = self._distal_plane_node_name(side, bone) if distal else self._plane_node_name(side, bone)
        label = "Distal" if distal else "Proximal"
        color = (0.95, 0.5, 0.1) if distal else None
        try:
            node, created, message = self._create_plane_node_at_anatomical_axis(side, bone, node_name, label, color)
            self._update_plane_handle_interactivity()
            self.osteotomy_status.setText(message)
            if created:
                # bisecting principle as the fixed default; manual changes afterwards are kept
                self.onAlignMasterBisecting()
                self.osteotomy_status.setText(
                    f"{label} master plane created and aligned by the bisecting principle - move / "
                    "rotate it by hand if needed, then create the second plane. "
                    + ("" if "aligned" in self.osteotomy_status.text else self.osteotomy_status.text))
        except Exception as exc:
            self.osteotomy_status.setText(f"Could not create master plane: {exc}")

    def onCreateDerivedPlane(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        master_is_distal = self.reference_plane_selector.currentText.startswith("Distal")
        master_name = self._distal_plane_node_name(side, bone) if master_is_distal else self._plane_node_name(side, bone)
        other_name = self._plane_node_name(side, bone) if master_is_distal else self._distal_plane_node_name(side, bone)
        master = slicer.mrmlScene.GetFirstNodeByName(master_name)
        if master is None:
            self.osteotomy_status.setText("Create and position the MASTER plane first.")
            return
        other = slicer.mrmlScene.GetFirstNodeByName(other_name)
        if other is None:
            origin, x, y, n = self._get_plane_frame(master, side, bone)
            other = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", other_name)
            other.CreateDefaultDisplayNodes()
            self._set_plane_frame(other, origin, x, y, n)
            label = "Proximal" if master_is_distal else "Distal"
            color = None if master_is_distal else (0.95, 0.5, 0.1)
            kwargs = {"label": label}
            if color is not None: kwargs["color"] = color
            self._configure_plane_size(other, side, bone, **kwargs)
        ok, message = self._sync_other_plane(side, bone)
        self._update_plane_handle_interactivity()
        self.osteotomy_status.setText(message)

    def onInsertCuttingPlane(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText

        node_name = self._plane_node_name(side, bone)
        plane_node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if plane_node is not None:
            # This button only does the initial placement - once the plane exists, its position/
            # orientation is left alone here, no matter how many times this is clicked, since the
            # correction math always reads the plane's live current state anyway (see
            # onPreviewCorrection/onShowHingePoint) and re-running this used to silently snap any
            # careful manual placement (axial move, tilt, centering) back to the default start.
            self.osteotomy_status.setText(
                "Proximal plane already exists - its position/orientation was left untouched. "
                "To reset it to the default starting position, delete it first (Slicer's Data "
                "module) and click this again.")
            return

        origin, normal, error = self._get_anatomical_axis(side, bone)
        if error:
            self.osteotomy_status.setText(error)
            return

        # The very first vtkMRMLMarkupsPlaneNode created in a Slicer session can take up to
        # ~30 seconds to build its interactive 3D widget the first time - observed on
        # Rosetta-translated Slicer on Apple Silicon, where the process sits at ~0% CPU
        # while it happens (waiting on the GPU/rendering pipeline, not computing). It's a
        # one-time cost of that widget type, not something this module's code controls or
        # can speed up - but a silent 30s stall looks exactly like a crash, so make it
        # visible instead: a wait cursor + an explicit status message, flushed to screen
        # with processEvents() before the slow call actually starts.
        self.osteotomy_status.setText(
            "Creating cutting plane... (can take up to ~30s the very first time this "
            "session - not a crash, please wait)")
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()

        try:
            plane_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", node_name)
            plane_node.CreateDefaultDisplayNodes()

            # Use the plane's own origin/normal API rather than managing control points directly -
            # manually adding 2 control points hit an unexpected "maximum number of control points
            # allowed (1)" restriction, since the exact control-point representation a freshly
            # created plane node expects isn't something to guess at from outside.
            if hasattr(plane_node, "SetOriginWorld") and hasattr(plane_node, "SetNormalWorld"):
                plane_node.SetOriginWorld(origin[0], origin[1], origin[2])
                plane_node.SetNormalWorld(normal[0], normal[1], normal[2])
            elif hasattr(plane_node, "SetOrigin") and hasattr(plane_node, "SetNormal"):
                plane_node.SetOrigin(origin[0], origin[1], origin[2])
                plane_node.SetNormal(normal[0], normal[1], normal[2])
            else:
                self.osteotomy_status.setText(
                    "This Slicer version's Plane markup doesn't expose SetOrigin/SetNormal - "
                    "please tell me the Slicer version so I can use the right API for it.")
                return

            has_condyle_width = self._configure_plane_size(plane_node, side, bone, label="Proximal")
            self._update_plane_handle_interactivity()

            size_note = "" if has_condyle_width else (
                " (Size: 150mm placeholder default, since the condyle landmarks aren't placed "
                "yet - remove and re-insert after placing them to scale to 2x condyle width.)")
            self.osteotomy_status.setText(
                f"Proximal plane inserted (starting position on the {bone.lower()} shaft axis) - "
                f"now position/orient it freely in the 3D view or slice views.{size_note}")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onShowHingePoint(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText

        proximal_plane_node = slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone))
        if proximal_plane_node is None:
            self.osteotomy_status.setText(
                "Insert the proximal plane first ('Insert/update proximal plane').")
            return
        distal_plane_node = slicer.mrmlScene.GetFirstNodeByName(self._distal_plane_node_name(side, bone))
        if distal_plane_node is None:
            self.osteotomy_status.setText(
                "Insert the distal plane first ('Insert/update distal plane').")
            return

        proximal_origin, proximal_normal = self._get_plane_origin_normal(proximal_plane_node)
        distal_origin, distal_normal = self._get_plane_origin_normal(distal_plane_node)
        hinge_point, hinge_axis = osteotomy_logic.planes_intersection_line(
            proximal_origin, proximal_normal, distal_origin, distal_normal)
        if hinge_point is None:
            self.osteotomy_status.setText(
                "The proximal and distal planes are (nearly) parallel - no hinge yet. Tilt the "
                "distal plane in the 3D view or slice views, or drag its border rotation handles.")
            return

        node_name = f"Doggybone {side} {bone.lower()} hinge point"
        marker_node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if marker_node is None:
            marker_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
            marker_node.CreateDefaultDisplayNodes()
            marker_node.SetMaximumNumberOfControlPoints(1)
            display_node = marker_node.GetDisplayNode()
            if display_node is not None:
                display_node.SetColor(1.0, 0.0, 0.0)
                display_node.SetSelectedColor(1.0, 0.0, 0.0)
                if hasattr(display_node, "SetPropertiesLabelVisibility"):
                    display_node.SetPropertiesLabelVisibility(False)
                if hasattr(display_node, "SetPointLabelsVisibility"):
                    display_node.SetPointLabelsVisibility(True)
                if hasattr(display_node, "SetTextScale"):
                    display_node.SetTextScale(2.5)

        if marker_node.GetNumberOfControlPoints() == 0:
            marker_node.AddControlPoint(hinge_point[0], hinge_point[1], hinge_point[2])
        else:
            marker_node.SetNthControlPointPosition(0, hinge_point[0], hinge_point[1], hinge_point[2])
        if hasattr(marker_node, "SetNthControlPointLabel"):
            marker_node.SetNthControlPointLabel(0, "Hinge")

        self.osteotomy_status.setText(
            "Hinge point marker updated - adjust either plane and click this button again to see "
            "where it moves to.")

    def onSyncOtherPlane(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        ok, message = self._sync_other_plane(side, bone)
        self.osteotomy_status.setText(message)

    def _sync_other_plane(self, side, bone):
        """Derive SECOND from MASTER in two explicit, local steps.

        1) Apply the side-aware varus/valgus correction about the anatomical AP direction
           through the MASTER cut centre.
        2) Rotate the already angulated SECOND-plane frame by the torsion correction about
           the TRUE anatomical shaft LINE (the same line onPreviewCorrection's actual fragment
           transform uses - see anatomical_correction_matrix) - not the master plane's own
           normal. The master plane's normal only coincides with the true shaft once some
           frontal/wedge correction has already tilted it away from it, so rotating about it
           instead of the true shaft line would visibly disagree with where the fragment
           transform actually puts the corresponding cut face once wedge and torsion are both
           non-zero: this plane is meant to be a preview of that, so it uses the same axis.

        This deliberately treats the derived cutting plane as an object referenced to the
        master cut.  Its centre is not translated by either orientation step.  After creation
        the surgeon may still translate the SECOND plane for centring/osteotomy width.
        """
        proximal = slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone))
        distal = slicer.mrmlScene.GetFirstNodeByName(self._distal_plane_node_name(side, bone))
        if proximal is None or distal is None:
            return False, "Both proximal and distal planes must exist first."

        p_origin, p_x, p_y, p_n = self._get_plane_frame(proximal, side, bone)
        d_origin, d_x, d_y, d_n = self._get_plane_frame(distal, side, bone)

        master_is_distal = self.reference_plane_selector.currentText.startswith("Distal")
        if master_is_distal:
            master_origin, master_x, master_y, master_n = d_origin, d_x, d_y, d_n
            other_node = proximal
            label = "Proximal"
        else:
            master_origin, master_x, master_y, master_n = p_origin, p_x, p_y, p_n
            other_node = distal
            label = "Distal"

        # The true anatomical shaft line, at the point on it closest to the master cut - the
        # torsion axis for BOTH correction paths below. Computed once here since it doesn't
        # depend on the master plane's own (possibly wedge-tilted) orientation.
        anatomical_origin, shaft_axis, shaft_error = self._get_anatomical_axis(side, bone)
        if shaft_error:
            return False, shaft_error
        shaft_point_at_cut = osteotomy_logic.closest_point_on_line(
            master_origin, anatomical_origin, shaft_axis)

        if (self.osteotomy_technique.currentText.startswith("CWO")
                and self._cwo_alignment_is_congruent()):
            return self._sync_other_plane_swing_twist(
                side, bone, master_is_distal, master_origin, master_x, master_y, master_n,
                other_node, label)

        panel = self.left_panel if side == "left" else self.right_panel
        measurement_name = self.WEDGE_MEASUREMENT_NAME[bone]
        measurement = self._get_measurement_by_name(panel, measurement_name)
        if measurement is None:
            return False, f"Measurement '{measurement_name}' not found."
        ready, current_wedge, vv_description = measurement()
        if not ready:
            return False, f"Place the landmarks for '{measurement_name}' ({side}) first."

        shaft, ml, ap, frame_error = self._get_anatomical_frame(side, bone)
        if frame_error:
            return False, frame_error

        ok_rot, torsion_delta, rotation_message = self._calculated_rotation_delta(side, bone)
        if not ok_rot:
            return False, rotation_message
        self.rotation_angle.setValue(torsion_delta)

        # OWO: the SECOND plane is the virtual target - the master cut face of the moving fragment
        # in its corrected position. It uses exactly the rotation of the preview transform
        # (_desired_correction_rotation), so plane and moved fragment always coincide.
        moving_is_distal = not master_is_distal
        R, info = self._desired_correction_rotation(side, bone, moving_is_distal, master_origin, master_n)
        if R is None:
            return False, str(info)
        new_x = osteotomy_logic.normalize(R @ np.asarray(master_x, dtype=float))
        new_y = osteotomy_logic.normalize(R @ np.asarray(master_y, dtype=float))
        new_n = osteotomy_logic.normalize(R @ np.asarray(master_n, dtype=float))
        self.calculated_wedge.setValue(info["frontal_signed"])
        self.rotation_angle.setValue(info["torsion_signed"])

        # Keep the surgeon's SECOND-plane position.  Only its orientation is regenerated.
        other_origin, _, _, _ = self._get_plane_frame(other_node, side, bone)
        self._set_plane_frame(other_node, other_origin, new_x, new_y, new_n)
        self._update_ml_markers(other_node, side, bone)
        self._update_actual_plane_angle(master_n, new_n)
        text = (f"{label} plane updated: frontal {info['frontal_signed']:+.1f} deg, torsion "
                f"{info['torsion_signed']:+.1f} deg (verified on the landmarks)")
        if info.get("predicted"):
            text += "; predicted " + self._format_prediction(bone, info["predicted"])
        return True, text + "." + (("\n" + info["warning"]) if info.get("warning") else "")

    def _distal_plane_node_name(self, side, bone):
        return f"Doggybone {side} {bone.lower()} distal plane"

    def onInsertDistalPlane(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText

        node_name = self._distal_plane_node_name(side, bone)
        distal_plane_node = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if distal_plane_node is not None:
            # This button only does the initial placement - once the plane exists, its position/
            # orientation is left alone here, no matter how many times this is clicked, since the
            # correction math always reads both planes' live current state anyway (see
            # onPreviewCorrection/onShowHingePoint) and re-running this used to silently snap any
            # careful manual placement (axial move, tilt, centering) back to the proximal plane's
            # position.
            self.osteotomy_status.setText(
                "Distal plane already exists - its position/orientation was left untouched. To "
                "reset it to the proximal plane's current position and the current suggested "
                "tilt, delete it first (Slicer's Data module) and click this again.")
            return

        proximal_plane_node = slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone))
        if proximal_plane_node is None:
            self.osteotomy_status.setText(
                "Insert the proximal plane first ('Insert/update proximal plane').")
            return
        proximal_origin, proximal_normal = self._get_plane_origin_normal(proximal_plane_node)

        # Give the distal plane a sensible starting tilt (current measurement vs. target value)
        # instead of dropping it exactly coincident with the proximal plane - the position starts
        # at the proximal plane's origin, only the orientation is pre-angled, using the same
        # anatomical AP axis the OWO correction frame uses. Falls back to no tilt (coincident) if
        # the needed landmarks/measurement aren't available yet.
        suggested_deg = 0.0
        suggestion_note = ""
        panel = self.left_panel if side == "left" else self.right_panel
        measurement_name = self.WEDGE_MEASUREMENT_NAME[bone]
        measurement = self._get_measurement_by_name(panel, measurement_name)
        _, _, ap_axis, frame_error = self._get_anatomical_frame(side, bone)
        if measurement is not None and ap_axis is not None:
            ready, value, description = measurement()
            if ready:
                magnitude = abs(value - self.target_wedge_value.value)
                apex_is_medial = "Varus" in description
                suggested_deg = magnitude * (-1.0 if apex_is_medial else 1.0)
                suggestion_note = (
                    f" Starting tilt {abs(suggested_deg):.1f}\N{DEGREE SIGN} pre-filled from "
                    f"'{measurement_name}' vs. target value - adjust further by hand as needed.")

        if abs(suggested_deg) > 1e-6:
            # Project ap_axis to be exactly perpendicular to the PROXIMAL PLANE'S ACTUAL current
            # normal (not just the idealized landmark-derived shaft direction it started from) -
            # otherwise, if the proximal plane has been manually tilted since insertion (which
            # the rotation handles are there for), rotating about the unprojected ap_axis doesn't
            # land the distal plane exactly suggested_deg away from the proximal plane as measured
            # by the ACTUAL later intersection axis, and the wedge angle computed at preview time
            # comes out inflated/distorted instead of matching this starting guess.
            ap_in_plane = ap_axis - np.dot(ap_axis, proximal_normal) * proximal_normal
            ap_in_plane_norm = np.linalg.norm(ap_in_plane)
            if ap_in_plane_norm > 1e-6:
                ap_in_plane = ap_in_plane / ap_in_plane_norm
            else:
                ap_in_plane = ap_axis  # proximal plane essentially never tilted away from ap_axis
            rotation = vtk.vtkTransform()
            rotation.RotateWXYZ(suggested_deg, *ap_in_plane)
            distal_normal = np.array(rotation.TransformVector(
                proximal_normal[0], proximal_normal[1], proximal_normal[2]))
            distal_normal = distal_normal / np.linalg.norm(distal_normal)
        else:
            distal_normal = proximal_normal
            if frame_error:
                suggestion_note = " (Inserted coincident with the proximal plane - tilt it by hand.)"

        self.osteotomy_status.setText(
            "Creating distal plane... (can take up to ~30s the very first time this session "
            "- not a crash, please wait)")
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()

        try:
            distal_plane_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", node_name)
            distal_plane_node.CreateDefaultDisplayNodes()

            if hasattr(distal_plane_node, "SetOriginWorld") and hasattr(distal_plane_node, "SetNormalWorld"):
                distal_plane_node.SetOriginWorld(proximal_origin[0], proximal_origin[1], proximal_origin[2])
                distal_plane_node.SetNormalWorld(distal_normal[0], distal_normal[1], distal_normal[2])
            elif hasattr(distal_plane_node, "SetOrigin") and hasattr(distal_plane_node, "SetNormal"):
                distal_plane_node.SetOrigin(proximal_origin[0], proximal_origin[1], proximal_origin[2])
                distal_plane_node.SetNormal(distal_normal[0], distal_normal[1], distal_normal[2])

            # Distinct color from the proximal plane (yellow) so the two planes are easy to tell
            # apart once separated in the 3D/2D views.
            self._configure_plane_size(distal_plane_node, side, bone, color=(0.95, 0.5, 0.1), label="Distal")
            # Immediately derive its COMPLETE orientation (including in-plane torsion) from the
            # current target values. Position remains at the proximal plane until the surgeon
            # translates it in X/Y/Z.
            self._update_plane_handle_interactivity()
            ok_sync, sync_message = self._sync_other_plane(side, bone)
            self._update_plane_handle_interactivity()
            self.osteotomy_status.setText(
                sync_message if ok_sync else
                "Distal plane inserted. " + sync_message)
        finally:
            qt.QApplication.restoreOverrideCursor()
