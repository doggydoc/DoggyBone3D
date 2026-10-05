import qt, ctk, vtk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import errorDisplay

import os.path as osp
import csv
import numpy as np
from copy import deepcopy
import locale
locale.setlocale(locale.LC_ALL, '')

# Slicer's "Reload" (slicer.util.reloadScriptedModule) re-executes this file, but a plain
# `import X` further down is a no-op once X is already in sys.modules - so without an explicit
# importlib.reload() here, edits to any Resources/*.py file can silently keep running the
# version from the FIRST time this module was ever loaded in the session, even after many
# reloads. Force a fresh read of our own submodules every time this file runs, leaf modules
# (measurement_logic, landmark_logic) before the modules that do `from ... import *` on them
# (measurements, landmarks), so the re-import below picks up the refreshed content.
import importlib
from Resources import measurement_logic
importlib.reload(measurement_logic)
from Resources import landmark_logic
importlib.reload(landmark_logic)
from Resources import osteotomy_logic
from Resources import guide_logic
importlib.reload(osteotomy_logic)
importlib.reload(guide_logic)
from Resources import measurements
importlib.reload(measurements)
from Resources import landmarks
importlib.reload(landmarks)

from Resources.measurements import MEASUREMENTS
from Resources.landmarks import LANDMARKS

MODULE_PATH = osp.dirname(__file__)
#
# BoneAngleMeterModule
#

class BoneAngleMeterModule(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Bone Angle Meter"
    self.parent.categories = ["Measurements"]
    self.parent.dependencies = []
    self.parent.contributors = ["Juliette Burg"]
    self.parent.helpText = """
This is a module for measuring bone angles based on landmarks.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
Developed during Juliette's doctorate at CTK, LMU Munich.
"""


class BoneAngleMeterModuleWidget(ScriptedLoadableModuleWidget):
    """
    Base widget that manages the measurement groups
    """

    SEGMENTATION_NODE_NAME = "Automatic Bone Segmentation Node"

    # Which existing angle measurement feeds the "suggest" button for each correction and bone -
    # see MeasurementsPanel/measurement_logic.py for where these names/descriptions come from.
    WEDGE_MEASUREMENT_NAME = {"Femur": "aLDFA (Varus/Valgus Femur)", "Tibia": "Varus Valgus Tibia"}
    ROTATION_MEASUREMENT_NAME = {"Femur": "Antetorsion", "Tibia": "Tibia Torsion"}
    # Default target ("Zielwert") each measurement should read after correction, per bone.
    # Femur wedge (90°) and Tibia wedge (0°) match how those measurements are already defined in
    # measurement_logic.py (aLDFA is read relative to 90°; the tibial varus/valgus angle is
    # already a deviation-from-straight angle). The rotation targets are placeholders (0°) - this
    # module has no reliable breed/bone-specific reference for normal antetorsion/tibial torsion,
    # so these MUST be reviewed and set by the vet rather than trusted as clinical defaults.
    # Same measurement methods as Doggybone 3D: aLDFA against the canine reference (~94 deg); tibia =
    # frontal angle between the joint lines aMPTA + aMDTA - 180 (normal ~ MPTA_REF + MDTA_REF - 180).
    WEDGE_TARGET_DEFAULT = {"Femur": measurement_logic.ALDFA_REFERENCE_DEG,
                            "Tibia": round(measurement_logic.TIBIA_MPTA_REF_DEG + measurement_logic.TIBIA_MDTA_REF_DEG - 180.0, 1)}
    ROTATION_TARGET_DEFAULT = {"Femur": 0.0, "Tibia": 0.0}
    # Landmark pair used as the medial->lateral reference direction at the osteotomy level -
    # reuses landmarks already placed for the angle measurements rather than adding new ones.
    ML_LANDMARKS = {"Femur": ("medial femur condyle", "lateral femur condyle"),
                     "Tibia": ("condylus medialis tibiae", "condylus lateralis tibiae")}

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        #####################
        # GUI and connections
        #####################
        # Automatic Segmentation
        segmentation_collapsible_button = ctk.ctkCollapsibleButton()
        segmentation_collapsible_button.text = "3D Segmentation"
        segmentation_collapsible_button.setStyleSheet(
            "ctkCollapsibleButton { background-color: #2b6cb0; color: white; font-weight: bold; }")
        self.layout.addWidget(segmentation_collapsible_button)
        segmentation_form_layout = qt.QFormLayout(segmentation_collapsible_button)
        segmentation_form_layout.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
        self.apply_segmentation_button = qt.QPushButton("Apply")
        self.apply_segmentation_button.connect('clicked(bool)', self.onApplySegmentation)
        segmentation_form_layout.addRow("Segmentation", self.apply_segmentation_button)

        self.threshold_lower = qt.QSpinBox()
        self.threshold_lower.setRange(-3000, 3000)
        self.threshold_lower.setValue(300)
        self.threshold_lower.setMinimumWidth(70)
        self.threshold_upper = qt.QSpinBox()
        self.threshold_upper.setRange(-3000, 3000)
        self.threshold_upper.setValue(3000)
        self.threshold_upper.setMinimumWidth(70)
        segmentation_form_layout.addRow("Lower threshold", self.threshold_lower)
        segmentation_form_layout.addRow("Upper threshold", self.threshold_upper)

        # A separate threshold for landmark snapping, independent of the segmentation's own
        # Lower threshold above. The segmentation threshold controls how complete the visual bone
        # mesh looks (too high, and legitimate lower-density bone gets excluded, leaving gaps);
        # the snap threshold controls how strictly "bone" has to read before a landmark is
        # allowed to land there. A scan where soft tissue reads unusually high (e.g. ~340 HU)
        # needs a snap threshold well above that, without needing to raise the segmentation
        # threshold to match and degrade the mesh.
        self.snap_threshold = qt.QSpinBox()
        self.snap_threshold.setRange(-3000, 3000)
        self.snap_threshold.setValue(600)
        self.snap_threshold.setMinimumWidth(70)
        self.snap_threshold.setToolTip(
            "How strictly a landmark must land on bone density (HU) when placed or dragged. "
            "Independent of the segmentation Lower threshold above - raise this if landmarks "
            "keep landing on soft tissue, without needing to change the segmentation itself.")
        self.snap_threshold.valueChanged.connect(self.onSnapThresholdChanged)
        segmentation_form_layout.addRow("Landmark snap threshold", self.snap_threshold)

        # Keep the landmark bone-snapping threshold in sync with this control's current value
        # from the start, not just from the first time the user moves it.
        landmark_logic.set_bone_threshold(self.snap_threshold.value)

        self.smoothing = qt.QDoubleSpinBox()
        self.smoothing.setRange(0, 1)
        self.smoothing.setDecimals(1)
        self.smoothing.setSingleStep(0.1)
        self.smoothing.setValue(0.5)
        segmentation_form_layout.addRow("Surface smoothing", self.smoothing)

        self.opacity_slider = qt.QSlider(qt.Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.setToolTip(
            "3D opacity of the bone surface. Lower it to see landmarks placed inside the bone (e.g. femur head, condyle centers).")
        self.opacity_slider.valueChanged.connect(self.onOpacityChanged)
        segmentation_form_layout.addRow("3D Opacity", self.opacity_slider)

        self.segmentation_status = qt.QLabel("no segmentation")
        self.segmentation_status.setWordWrap(True)
        segmentation_form_layout.addRow("Status", self.segmentation_status)

        # Clean-up of the bone segmentation with Segment Editor effects: a free-form lasso in the
        # 3D view keeps only the encircled region (through the full depth), then small islands
        # (noise, CT table fragments) are removed.
        self.lasso_button = qt.QPushButton("Lasso: keep inside")
        self.lasso_button.setCheckable(True)
        self.lasso_button.setToolTip(
            "Switch on, then draw around the region to KEEP in the 3D view (hold the left mouse "
            "button). Everything outside the lasso is removed through the full depth. Draw "
            "several times if needed; switch off when done. Landmark placement pauses meanwhile.")
        self.lasso_button.toggled.connect(self.onLassoToggled)
        self.delete_cleanup_button = qt.QPushButton("Delete")
        self.delete_cleanup_button.setToolTip(
            "Delete all clean-up edits (lasso and islands) - back to the segmentation as it was "
            "before the first clean-up step, to start again.")
        self.delete_cleanup_button.clicked.connect(self.onDeleteSegmentationEdits)
        lasso_row = qt.QHBoxLayout()
        lasso_row.setContentsMargins(0, 0, 0, 0)
        lasso_row.addWidget(self.lasso_button, 1)
        lasso_row.addWidget(self.delete_cleanup_button)
        lasso_row_widget = qt.QWidget()
        lasso_row_widget.setLayout(lasso_row)
        segmentation_form_layout.addRow("Clean-up", lasso_row_widget)

        self.island_min_size = qt.QSpinBox()
        self.island_min_size.setRange(1, 100000000)
        self.island_min_size.setSingleStep(1000)
        self.island_min_size.setValue(1000)
        self.island_min_size.setSuffix(" voxels")
        self.island_min_size.setToolTip("Islands smaller than this are removed.")
        self.remove_islands_button = qt.QPushButton("Remove small islands")
        self.remove_islands_button.clicked.connect(self.onRemoveSmallIslands)
        islands_row = qt.QHBoxLayout()
        islands_row.setContentsMargins(0, 0, 0, 0)
        islands_row.addWidget(self.island_min_size)
        islands_row.addWidget(self.remove_islands_button, 1)
        islands_row_widget = qt.QWidget()
        islands_row_widget.setLayout(islands_row)
        segmentation_form_layout.addRow("Small islands", islands_row_widget)

        # A VISIBLE, compact Segment Editor: interactive effects such as the Scissors/lasso only
        # receive the view's mouse events when their editor widget is part of the GUI. Only the
        # two clean-up effects are shown; its own undo/redo works together with the buttons above.
        editor = slicer.qMRMLSegmentEditorWidget()
        editor.setMRMLScene(slicer.mrmlScene)
        editor.setMaximumNumberOfUndoStates(3)
        for prop in ("segmentationNodeSelectorVisible", "sourceVolumeNodeSelectorVisible",
                     "switchToSegmentationsButtonVisible",
                     "unorderedEffectsVisible"):
            try:
                setattr(editor, prop, False)
            except Exception:
                pass
        editor.setEffectNameOrder(["Scissors", "Islands"])
        editor_node = slicer.mrmlScene.GetFirstNodeByName("BoneAngleMeter segment editor")
        if editor_node is None:
            editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "BoneAngleMeter segment editor")
        editor.setMRMLSegmentEditorNode(editor_node)
        segmentation_form_layout.addRow(editor)
        editor.setVisible(False)   # used internally (Islands); the lasso is the module's own
        self._cleanup_editor_widget = editor
        if slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME) is not None:
            status_text = self.segmentation_status.text
            self._ensure_cleanup_editor()
            self.segmentation_status.setText(status_text)

        # Measurements panel, embedded directly in this module's sidebar (always visible next to
        # the 3D/slice views) instead of a separate window - no switching between windows needed.
        # Wrapped in a collapsible section (like "3D Segmentation" above) so the whole thing can
        # be shown/hidden as one unit.
        measurements_collapsible_button = ctk.ctkCollapsibleButton()
        measurements_collapsible_button.text = "Measurements"
        measurements_collapsible_button.setStyleSheet(
            "ctkCollapsibleButton { background-color: #6b46c1; color: white; font-weight: bold; }")
        self.layout.addWidget(measurements_collapsible_button)
        measurements_layout = qt.QVBoxLayout(measurements_collapsible_button)

        self.side_selector = qt.QComboBox()
        self.side_selector.addItems(["Left", "Right"])
        self.side_selector.currentIndexChanged.connect(self._on_side_changed)

        # Reuse existing landmark lists (e.g. after "Reload") instead of adding a new pair each
        # time, and re-link landmarks that are already placed in them.
        self.left_panel = MeasurementsPanel('left',
            self._get_or_create_fiducial_node("BoneAngleMeterFiducialsLeft"), self)
        self.right_panel = MeasurementsPanel('right',
            self._get_or_create_fiducial_node("BoneAngleMeterFiducialsRight"), self)
        self._relink_existing_landmarks(self.left_panel)
        self._relink_existing_landmarks(self.right_panel)

        self.side_stack = qt.QStackedWidget()
        self.side_stack.addWidget(self.left_panel)
        self.side_stack.addWidget(self.right_panel)

        side_row = qt.QHBoxLayout()
        side_row.addWidget(qt.QLabel("Side"))
        side_row.addWidget(self.side_selector)
        side_row.addStretch(1)
        side_row_widget = qt.QWidget()
        side_row_widget.setLayout(side_row)

        measurements_layout.addWidget(side_row_widget)
        measurements_layout.addWidget(self.side_stack, 1)

        # Osteotomy planning (stage 1): a cutting plane, parametrically seeded from the existing
        # bone shaft axis (reusing the same landmarks already placed for the angle measurements),
        # then handed off to Slicer's native, fully-interactive plane widget for fine adjustment.
        osteotomy_collapsible_button = ctk.ctkCollapsibleButton()
        osteotomy_collapsible_button.text = "Osteotomy Planning"
        osteotomy_collapsible_button.setStyleSheet(
            "ctkCollapsibleButton { background-color: #c05621; color: white; font-weight: bold; }")
        osteotomy_collapsible_button.collapsed = True
        # Without this, expanding this section (by far the largest/most widget-dense one) makes
        # its content's natural, unconstrained width propagate up to the whole module panel -
        # the panel then jumps to that width and the divider between the panel and the 3D view
        # can no longer be dragged narrower. Telling the collapsible button to IGNORE its
        # content's size hint for the purposes of its own minimum size lets the panel keep
        # whatever width was dragged, with the content wrapping/scrolling inside as needed,
        # exactly like the smaller Segmentation/Measurements sections already do.
        osteotomy_collapsible_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        self.layout.addWidget(osteotomy_collapsible_button)
        osteotomy_form_layout = qt.QFormLayout(osteotomy_collapsible_button)
        osteotomy_form_layout.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)

        self.osteotomy_side = qt.QComboBox()
        self.osteotomy_side.addItems(["Left", "Right"])
        self.osteotomy_side.currentIndexChanged.connect(self._update_plane_handle_interactivity)
        osteotomy_form_layout.addRow("Side", self.osteotomy_side)

        self.osteotomy_bone = qt.QComboBox()
        self.osteotomy_bone.addItems(["Femur", "Tibia"])
        self.osteotomy_bone.currentIndexChanged.connect(self._on_osteotomy_bone_changed)
        osteotomy_form_layout.addRow("Bone", self.osteotomy_bone)

        self.osteotomy_technique = qt.QComboBox()
        self.osteotomy_technique.addItems(["OWO - Opening Wedge", "CWO - Closing Wedge"])
        self.osteotomy_technique.setToolTip(
            "OWO: one cut is enough, the distal fragment is opened up, the gap stays/is bridged "
            "(e.g. plate/bone graft). CWO: a bone wedge is removed, no gap - needs a second, "
            "converging cutting plane.")
        osteotomy_form_layout.addRow("Technique", self.osteotomy_technique)

        # Planning mode: either derive correction from measured -> target anatomy, or enter
        # correction angles directly.  Direct mode deliberately does not require anatomical
        # measurements; frontal correction is applied about local Y of the MASTER plane and
        # torsion about the MASTER normal.
        self.correction_mode = qt.QComboBox()
        self.correction_mode.addItems(["Target anatomy", "Direct correction"])
        self.correction_mode.currentIndexChanged.connect(self._on_correction_mode_changed)
        osteotomy_form_layout.addRow("Planning mode", self.correction_mode)

        self.direct_frontal_value = qt.QDoubleSpinBox()
        self.direct_frontal_value.setRange(-90.0, 90.0)
        self.direct_frontal_value.setSingleStep(0.5)
        self.direct_frontal_value.setSuffix(" °")
        self.direct_frontal_value.setToolTip("Signed frontal correction applied directly to the SECOND plane.")
        osteotomy_form_layout.addRow("Frontal correction", self.direct_frontal_value)

        self.direct_torsion_value = qt.QDoubleSpinBox()
        self.direct_torsion_value.setRange(-180.0, 180.0)
        self.direct_torsion_value.setSingleStep(0.5)
        self.direct_torsion_value.setSuffix(" °")
        self.direct_torsion_value.setToolTip("Signed torsion applied directly about the MASTER-plane normal.")
        osteotomy_form_layout.addRow("Torsion correction", self.direct_torsion_value)

        self.actual_plane_angle_label = qt.QLabel("Master/SECOND angle: --")
        osteotomy_form_layout.addRow(self.actual_plane_angle_label)

        # Compact target-anatomy workflow: measured anatomy first, then postoperative targets.
        osteotomy_form_layout.addRow(qt.QLabel("<b>Measured values</b>"))
        self.current_values_label = qt.QLabel("Place the required landmarks")
        self.current_values_label.setWordWrap(True)
        osteotomy_form_layout.addRow(self.current_values_label)

        osteotomy_form_layout.addRow(qt.QLabel("<b>Target values</b>"))
        self.target_wedge_value = qt.QDoubleSpinBox()
        self.target_wedge_value.setRange(-90.0, 180.0)
        self.target_wedge_value.setSingleStep(0.5)
        self.target_wedge_value.setSuffix(" °")
        osteotomy_form_layout.addRow("Varus/valgus target", self.target_wedge_value)

        self.target_rotation_value = qt.QDoubleSpinBox()
        self.target_rotation_value.setRange(-90.0, 90.0)
        self.target_rotation_value.setSingleStep(0.5)
        self.target_rotation_value.setSuffix(" °")
        osteotomy_form_layout.addRow("Torsion target", self.target_rotation_value)

        osteotomy_form_layout.addRow(qt.QLabel("<b>Calculated correction</b>"))
        self.calculated_wedge = qt.QDoubleSpinBox()
        self.calculated_wedge.setRange(-180.0, 180.0)
        self.calculated_wedge.setSuffix(" °")
        self.calculated_wedge.setReadOnly(True)
        self.calculated_wedge.setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
        osteotomy_form_layout.addRow("Varus/valgus", self.calculated_wedge)

        self.rotation_angle = qt.QDoubleSpinBox()
        self.rotation_angle.setRange(-90.0, 90.0)
        self.rotation_angle.setSingleStep(0.5)
        self.rotation_angle.setSuffix(" °")
        self.rotation_angle.setReadOnly(True)
        self.rotation_angle.setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
        osteotomy_form_layout.addRow("Torsion", self.rotation_angle)

        self.target_wedge_value.valueChanged.connect(self._on_target_values_changed)
        self.target_rotation_value.valueChanged.connect(self._on_target_values_changed)
        self.direct_frontal_value.valueChanged.connect(self._on_direct_values_changed)
        self.direct_torsion_value.valueChanged.connect(self._on_direct_values_changed)
        self._on_osteotomy_bone_changed(self.osteotomy_bone.currentIndex)
        self._on_correction_mode_changed(self.correction_mode.currentIndex)

        self.reference_plane_selector = qt.QComboBox()
        self.reference_plane_selector.addItems(["Proximal", "Distal"])
        self.reference_plane_selector.currentIndexChanged.connect(self._on_master_plane_changed)
        osteotomy_form_layout.addRow("Master cutting plane", self.reference_plane_selector)
        self.active_plane_selector = self.reference_plane_selector

        self.cwo_alignment_selector = qt.QComboBox()
        self.cwo_alignment_selector.addItems([
            "Contact-congruent (hinge closure + torsion about contact normal)",
            "Measurement-based (legacy)"])
        self.cwo_alignment_selector.setToolTip(
            "CWO only. Contact-congruent: the moving fragment is rotated about the hinge (the "
            "intersection line of both planes) until its cut face lies exactly on the fixed cut "
            "face, then derotated about the normal of that contact face - the only motion "
            "possible once the faces are compressed flush. The frontal correction then equals "
            "the angle between the planes; the status shows it next to the measured need.\n"
            "Measurement-based: wedge size from the measurement, torsion about the shaft axis; "
            "the cut faces are generally not congruent.")
        osteotomy_form_layout.addRow("CWO alignment", self.cwo_alignment_selector)

        self.osteotomy_insert_button = qt.QPushButton("Insert/update MASTER plane")
        self.osteotomy_insert_button.clicked.connect(self.onInsertMasterPlane)
        osteotomy_form_layout.addRow(self.osteotomy_insert_button)

        self.align_bisecting_button = qt.QPushButton("Align MASTER plane (bisecting principle)")
        self.align_bisecting_button.setToolTip(
            "Sets the MASTER plane perpendicular to the corrected segment axis (each cut "
            "perpendicular to its own segment axis, wedge symmetric to the bisector), keeps its "
            "level on the bone, centres it on the shaft axis and derives the SECOND plane. Then the "
            "wedge equals the frontal correction and the derotation on the flush contact face is "
            "a pure torsion.")
        self.align_bisecting_button.clicked.connect(self.onAlignMasterBisecting)
        osteotomy_form_layout.addRow(self.align_bisecting_button)

        self.insert_distal_plane_button = qt.QPushButton("Create/update SECOND plane from MASTER")
        self.insert_distal_plane_button.clicked.connect(self.onCreateDerivedPlane)
        osteotomy_form_layout.addRow(self.insert_distal_plane_button)

        # Legacy controls remain as hidden compatibility objects because older helper methods
        # still write status text, but they no longer clutter the clinical workflow.
        self.show_hinge_button = qt.QPushButton()
        self.show_hinge_button.setVisible(False)
        self.sync_other_plane_button = qt.QPushButton()
        self.sync_other_plane_button.setVisible(False)
        self.osteotomy_status = qt.QLabel()
        self.osteotomy_status.setVisible(False)

        #############################
        # Corrective osteotomy (wedge + rotation, with live preview)
        #############################
        self.rotation_invert = qt.QCheckBox()
        self.rotation_invert.setVisible(False)
        self.rotation_suggest_button = qt.QPushButton()
        self.rotation_suggest_button.setVisible(False)

        osteotomy_form_layout.addRow(qt.QLabel("<b>Preview</b>"))

        self.preview_button = qt.QPushButton("Show/update correction preview")
        self.preview_button.setToolTip(
            "Splits the segmented bone model at the cutting plane(s) and shows the moving "
            "fragment shifted according to the wedge + rotation correction (the other fragment "
            "stays fixed).")
        self.preview_button.clicked.connect(self.onPreviewCorrection)
        osteotomy_form_layout.addRow(self.preview_button)

        # Clinical fine adjustment AFTER the calculated angular correction.  These are
        # translations in the LOCAL X/Y axes of the MASTER cutting plane only.  They do not
        # change varus/valgus, torsion, or the plane-normal (contact) position.
        self.fragment_shift_x = qt.QDoubleSpinBox()
        self.fragment_shift_x.setRange(-20.0, 20.0)
        self.fragment_shift_x.setDecimals(1)
        self.fragment_shift_x.setSingleStep(0.5)
        self.fragment_shift_x.setSuffix(" mm")
        self.fragment_shift_x.setToolTip("Fine translation of the corrected fragment along local X of the MASTER cutting plane.")
        osteotomy_form_layout.addRow("Fragment shift X", self.fragment_shift_x)

        self.fragment_shift_y = qt.QDoubleSpinBox()
        self.fragment_shift_y.setRange(-20.0, 20.0)
        self.fragment_shift_y.setDecimals(1)
        self.fragment_shift_y.setSingleStep(0.5)
        self.fragment_shift_y.setSuffix(" mm")
        self.fragment_shift_y.setToolTip("Fine translation of the corrected fragment along local Y of the MASTER cutting plane.")
        osteotomy_form_layout.addRow("Fragment shift Y", self.fragment_shift_y)

        self.reset_fragment_shift_button = qt.QPushButton("Reset X/Y shift")
        self.reset_fragment_shift_button.clicked.connect(self.onResetFragmentShift)
        osteotomy_form_layout.addRow(self.reset_fragment_shift_button)
        self.fragment_shift_x.valueChanged.connect(self._on_fragment_shift_changed)
        self.fragment_shift_y.valueChanged.connect(self._on_fragment_shift_changed)

        self.clear_preview_button = qt.QPushButton("Remove preview")
        self.clear_preview_button.clicked.connect(self.onClearPreview)
        osteotomy_form_layout.addRow(self.clear_preview_button)

        self.reset_planning_button = qt.QPushButton("Reset planning (re-read landmarks)")
        self.reset_planning_button.setToolTip(
            "Back to zero for the selected side and bone: removes the preview, the X/Y gizmo and "
            "both cutting planes, sets fragment shift and direct correction values to 0, and "
            "re-reads the measured values and calculated corrections from the CURRENT landmarks "
            "(so corrected landmarks are taken over). Target values are kept.")
        self.reset_planning_button.clicked.connect(self.onResetOsteotomyPlanning)
        osteotomy_form_layout.addRow(self.reset_planning_button)

        self.save_screenshot_button = qt.QPushButton("Save screenshot of 3D view...")
        self.save_screenshot_button.setToolTip(
            "Saves the current 3D view as a PNG image - e.g. to document the preview of the "
            "final result. Zoom/rotate the 3D view as desired first.")
        self.save_screenshot_button.clicked.connect(self.onSaveScreenshot)
        osteotomy_form_layout.addRow(self.save_screenshot_button)

        self.correction_status = qt.QLabel()
        self.correction_status.setVisible(False)

        self.hide_original_in_preview = qt.QCheckBox("Hide original segmentation in preview")
        self.hide_original_in_preview.setChecked(True)
        osteotomy_form_layout.addRow(self.hide_original_in_preview)

        # Guide planning (stage 2, upcoming): placeholder section so the entry point is already
        # visible once the guide geometry (cutting slot + drill sleeves, STL export) is built on
        # top of the plane(s)/correction above.
        guide_collapsible_button = ctk.ctkCollapsibleButton()
        guide_collapsible_button.text = "Guide Planning (work in progress)"
        guide_collapsible_button.setStyleSheet(
            "ctkCollapsibleButton { background-color: #718096; color: white; font-weight: bold; }")
        guide_collapsible_button.collapsed = True
        self.layout.addWidget(guide_collapsible_button)
        guide_layout = qt.QVBoxLayout(guide_collapsible_button)
        guide_info_label = qt.QLabel(
            "Footprint = coloured area on the bone SURFACE where the guide sits. Suggest it for "
            "the chosen side, refine it with brush / lasso in the 3D view, then extrude it into "
            "the guide body (lifted off the bone by the fit gap). Uses the side and bone selected "
            "in Osteotomy Planning.")
        guide_info_label.setWordWrap(True)
        guide_layout.addWidget(guide_info_label)

        guide_form = qt.QFormLayout()
        guide_layout.addLayout(guide_form)
        self.guide_side_selector = qt.QComboBox()
        self.guide_side_selector.addItems(["Cranial", "Caudal", "Medial", "Lateral"])
        self.guide_side_selector.setToolTip(
            "Side of the bone the guide sits on - the same choice for every bone. Directions come "
            "from the anatomical frame (shaft axis, medial/lateral landmarks, limb side).")
        guide_form.addRow("Guide side", self.guide_side_selector)

        def spin(minimum, maximum, value, step, suffix, tip, decimals=1):
            box = qt.QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setValue(value)
            box.setSuffix(suffix)
            box.setToolTip(tip)
            return box

        self.guide_footprint_width = spin(4.0, 80.0, 14.0, 1.0, " mm", "Width of the suggested contact area across the bone.")
        guide_form.addRow("Footprint width", self.guide_footprint_width)
        self.guide_footprint_margin = spin(2.0, 60.0, 12.0, 1.0, " mm",
                                           "How far the suggested area reaches beyond the cuts (room for the pins).")
        guide_form.addRow("Beyond the cuts", self.guide_footprint_margin)
        self.guide_suggest_button = qt.QPushButton("Suggest footprint")
        self.guide_suggest_button.clicked.connect(self.onSuggestGuideFootprint)
        guide_form.addRow(self.guide_suggest_button)

        # Surface editing tools (3D view): brush / lasso, add / remove, undo.
        self._guide_tool_buttons = {}
        tools_row = qt.QHBoxLayout()
        tools_row.setContentsMargins(0, 0, 0, 0)
        for key, label, tip in (("brush_add", "Brush +", "Colour the surface under the mouse (drag)."),
                                ("brush_remove", "Brush -", "Uncolour the surface under the mouse (drag)."),
                                ("lasso_add", "Lasso +", "Draw around the area to colour (visible surface only)."),
                                ("lasso_remove", "Lasso -", "Draw around the area to uncolour (visible surface only).")):
            button = qt.QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.toggled.connect(lambda checked, k=key: self.onGuideToolToggled(k, checked))
            tools_row.addWidget(button)
            self._guide_tool_buttons[key] = button
        self.guide_delete_button = qt.QPushButton("Delete")
        self.guide_delete_button.setToolTip(
            "Delete the whole footprint and the guide body - start again with 'Suggest footprint' "
            "or by painting with Brush + / Lasso +.")
        self.guide_delete_button.clicked.connect(self.onGuideDelete)
        tools_row.addWidget(self.guide_delete_button)
        tools_widget = qt.QWidget()
        tools_widget.setLayout(tools_row)
        guide_form.addRow("Edit", tools_widget)
        self.guide_brush_radius = spin(0.5, 15.0, 3.0, 0.5, " mm", "Brush radius.")
        guide_form.addRow("Brush radius", self.guide_brush_radius)

        self.guide_gap = spin(0.0, 2.0, 0.2, 0.05, " mm",
                              "Fit gap: the guide's inner surface is lifted this far off the bone.", decimals=2)
        guide_form.addRow("Fit gap", self.guide_gap)
        self.guide_thickness = spin(1.0, 10.0, 3.0, 0.5, " mm", "Wall thickness of the guide body.")
        guide_form.addRow("Wall thickness", self.guide_thickness)
        self.guide_extrude_button = qt.QPushButton("Extrude guide body")
        self.guide_extrude_button.clicked.connect(self.onExtrudeGuideBody)
        guide_form.addRow(self.guide_extrude_button)

        self.guide_status = qt.QLabel("No footprint yet.")
        self.guide_status.setWordWrap(True)
        guide_form.addRow("Status", self.guide_status)
        self._guide_observers = []
        self._guide_undo_stack = []
        self._guide_drag = False
        self._guide_active_tool = None

        # After a full UI rebuild (this whole setup() re-runs on every module Reload), Qt/macOS
        # occasionally doesn't repaint the panel immediately - it looks frozen (though it isn't;
        # switching to another app and back forces a repaint and "unsticks" it). Explicitly
        # flushing pending Qt events here gives it an immediate chance to repaint instead of
        # relying on that.
        slicer.app.processEvents()

    # Every node this module creates is looked up by name in the scene rather than cached on
    # the widget instance (self.xxx = ...). This makes the whole osteotomy/correction workflow
    # safe to use with Slicer's "Reload" (slicer.util.reloadScriptedModule) - a reload creates a
    # brand-new widget object with none of the old instance state, but the scene (and therefore
    # anything looked up by name) is untouched by a module reload.
    def _plane_node_name(self, side, bone):
        return f"BoneAngleMeter {side} {bone.lower()} proximal plane"

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
                pass

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
                pass
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

        node_name = f"BoneAngleMeter {side} {bone.lower()} hinge point"
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

        # DIRECT CORRECTION MODE: independent of landmark measurements.  This is also a
        # diagnostic path for validating plane geometry.  The signed frontal correction is
        # a rotation about local Y of the MASTER plane; torsion is a rotation about the true
        # anatomical shaft line (see above), through the point on it closest to the MASTER cut
        # centre.  When DISTAL is master, the inverse relative transform is required to
        # construct the proximal counterpart.
        if hasattr(self, "correction_mode") and self.correction_mode.currentText == "Direct correction":
            sign = -1.0 if master_is_distal else 1.0
            vv_angle = sign * float(self.direct_frontal_value.value)
            torsion_for_second = sign * float(self.direct_torsion_value.value)
            R_v = osteotomy_logic.matrix_to_numpy(
                osteotomy_logic.rotation_matrix_about_line(master_origin, master_y, vv_angle))[:3, :3]
            x1 = osteotomy_logic.normalize(R_v @ master_x)
            y1 = osteotomy_logic.normalize(R_v @ master_y)
            n1 = osteotomy_logic.normalize(R_v @ master_n)
            R_t = osteotomy_logic.matrix_to_numpy(
                osteotomy_logic.rotation_matrix_about_line(
                    shaft_point_at_cut, shaft_axis, torsion_for_second))[:3, :3]
            new_x = osteotomy_logic.normalize(R_t @ x1)
            new_y = osteotomy_logic.normalize(R_t @ y1)
            new_n = osteotomy_logic.normalize(R_t @ n1)
            other_origin, _, _, _ = self._get_plane_frame(other_node, side, bone)
            self._set_plane_frame(other_node, other_origin, new_x, new_y, new_n)
            self._update_ml_markers(other_node, side, bone)
            self._update_actual_plane_angle(master_n, new_n)
            return True, (f"{label} plane updated directly: frontal {vv_angle:+.1f} deg, "
                          f"torsion {torsion_for_second:+.1f} deg.")

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

        vv_delta = self.target_wedge_value.value - current_wedge

        # Frontal correction is deliberately purely mathematical:
        #
        #     delta = target angle - measured angle
        #
        # The measured label (varus/valgus), the fact that the current value is above or
        # below 90 degrees, and the selected CWO/OWO technique MUST NOT change this sign.
        # Left/right is already encoded in the anatomical frame through the directed
        # medial->lateral landmark axis.  The only sign reversal required here is the
        # kinematic one caused by constructing the opposite physical plane: if DISTAL is
        # the master, we derive PROXIMAL with the inverse relative rotation.
        vv_angle = -vv_delta if master_is_distal else vv_delta

        # STEP 1: frontal/angular correction.  Only the SECOND frame is rotated.
        R_v = osteotomy_logic.matrix_to_numpy(
            osteotomy_logic.rotation_matrix_about_line(master_origin, ap, vv_angle))[:3, :3]
        x1 = osteotomy_logic.normalize(R_v @ master_x)
        y1 = osteotomy_logic.normalize(R_v @ master_y)
        n1 = osteotomy_logic.normalize(R_v @ master_n)

        # STEP 2: torsion of the SECOND frame about the TRUE anatomical shaft line (see the
        # docstring above) - the same axis onPreviewCorrection's actual fragment transform
        # uses, so this plane's final orientation matches where the fragment transform will
        # really put the corresponding cut face. For a distal master the relative torsion
        # direction is reversed because we are constructing the proximal counterpart from the
        # distal reference.
        torsion_for_second = -torsion_delta if master_is_distal else torsion_delta
        R_t = osteotomy_logic.matrix_to_numpy(
            osteotomy_logic.rotation_matrix_about_line(
                shaft_point_at_cut, shaft_axis, torsion_for_second))[:3, :3]
        new_x = osteotomy_logic.normalize(R_t @ x1)
        new_y = osteotomy_logic.normalize(R_t @ y1)
        new_n = osteotomy_logic.normalize(R_t @ n1)

        # Keep the surgeon's SECOND-plane position.  Only its orientation is regenerated.
        other_origin, _, _, _ = self._get_plane_frame(other_node, side, bone)
        self._set_plane_frame(other_node, other_origin, new_x, new_y, new_n)
        self._update_ml_markers(other_node, side, bone)
        self._update_actual_plane_angle(master_n, new_n)

        return True, (
            f"{label} plane updated: target-current frontal rotation {vv_angle:+.1f} deg, then torsion "
            f"{torsion_for_second:+.1f} deg about the true anatomical shaft line.")

    def _distal_plane_node_name(self, side, bone):
        return f"BoneAngleMeter {side} {bone.lower()} distal plane"

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

    def _get_measurement_by_name(self, panel, name):
        for i in range(panel.measurement_stack.count):
            widget = panel.measurement_stack.widget(i)
            if widget.measurement.name == name:
                return widget.measurement
        return None

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

    def onSuggestRotation(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        panel = self.left_panel if side == "left" else self.right_panel
        measurement_name = self.ROTATION_MEASUREMENT_NAME[bone]
        measurement = self._get_measurement_by_name(panel, measurement_name)
        if measurement is None:
            self.correction_status.setText(f"Measurement '{measurement_name}' not found.")
            return
        ready, value, description = measurement()
        if not ready:
            self.correction_status.setText(
                f"Place the landmarks for '{measurement_name}' ({side}) first.")
            return
        target = self.target_rotation_value.value
        magnitude = abs(value - target)
        self.rotation_angle.setValue(magnitude)
        self.correction_status.setText(
            f"Suggestion from '{measurement_name}': {value:.1f}\N{DEGREE SIGN} ({description}), "
            f"target {target:.1f}\N{DEGREE SIGN} -> correction {magnitude:.1f}\N{DEGREE SIGN}. "
            "Check with the preview - adjust invert as needed.")

    def _preview_node_names(self, side, bone):
        return (f"BoneAngleMeter {side} {bone.lower()} proximal fragment",
                f"BoneAngleMeter {side} {bone.lower()} distal fragment (corrected)",
                f"BoneAngleMeter {side} {bone.lower()} correction transform",
                f"BoneAngleMeter {side} {bone.lower()} final corrected bone (combined)",
                f"BoneAngleMeter {side} {bone.lower()} wedge REMOVE")

    def onPreviewCorrection(self):
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
        if hinge_point is None:
            self.correction_status.setText(
                "The proximal and distal planes are (nearly) parallel - no wedge defined yet. "
                "Tilt the distal plane in the 3D view or slice views, or drag its border "
                "rotation handles.")
            return

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
        if not wedge_ready:
            self.correction_status.setText(
                f"Place the landmarks for '{wedge_measurement_name}' ({side}) first.")
            return
        wedge_magnitude = abs(wedge_value - self.target_wedge_value.value)

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
        shaft_point_at_cut = osteotomy_logic.closest_point_on_line(
            proximal_origin, anatomical_origin, shaft_axis_check)

        alignment_report = ""
        if is_cwo and self._cwo_alignment_is_congruent():
            fixed_origin, fixed_normal = ((proximal_origin, proximal_normal) if moving_is_distal
                                          else (distal_origin, distal_normal))
            desired_rotation, _desired_info = self._desired_correction_rotation(
                side, bone, moving_is_distal, fixed_origin, fixed_normal)
            matrix, alignment_report = self._contact_congruent_matrix(
                proximal_origin, proximal_normal, distal_origin, distal_normal, hinge_point,
                moving_is_distal, shaft_axis_check, anatomical_origin, rotation_deg,
                ap_axis, ml_axis, abs(wedge_delta),
                side=side, bone=bone, desired_rotation=desired_rotation)
        else:
            matrix = osteotomy_logic.anatomical_correction_matrix(
                shaft_point_at_cut, shaft_axis_check, rotation_deg,
                hinge_point, ap_axis, wedge_deg)
            if is_cwo:
                # DIRECTION from the planes (as intended): the fixed sign convention of
                # "target - measured" about the AP axis is mirrored depending on side and dataset
                # orientation and could OPEN the wedge instead of closing it.  Use whichever
                # sign brings the moving cut face closer to the fixed one.
                n_p = np.asarray(proximal_normal, dtype=float); n_p = n_p / np.linalg.norm(n_p)
                n_d = np.asarray(distal_normal, dtype=float); n_d = n_d / np.linalg.norm(n_d)
                moving_n, fixed_n = (n_d, n_p) if moving_is_distal else (n_p, n_d)

                def _face_residual(candidate):
                    rotated = osteotomy_logic.matrix_to_numpy(candidate)[:3, :3] @ moving_n
                    return float(np.degrees(np.arccos(min(1.0, abs(float(rotated @ fixed_n))))))

                flipped = osteotomy_logic.anatomical_correction_matrix(
                    shaft_point_at_cut, shaft_axis_check, rotation_deg,
                    hinge_point, ap_axis, -wedge_deg)
                if _face_residual(flipped) < _face_residual(matrix):
                    matrix = flipped
                    wedge_deg = -wedge_deg
                alignment_report = (f"Legacy alignment: angle left between the cut faces "
                                    f"{_face_residual(matrix):.1f}\N{DEGREE SIGN}.")

            # CONTACT CONSTRAINT (CWO): after the angular/torsional correction, the moving
            # fragment must still sit on the fixed osteotomy surface.  Rotations about anatomical
            # axes can otherwise leave the corrected fragment visibly displaced from the cut even
            # when the requested angles are correct.  Translate the transformed centre of the
            # MOVING cut onto the centre of the FIXED cut.  This does not alter varus/valgus or
            # torsion; it only removes the rigid-body offset created by the rotations.
            #
            # Proximal master -> distal fragment moves: distal cut -> proximal/master cut.
            # Distal master   -> proximal fragment moves: proximal cut -> distal/master cut.
            if is_cwo:
                H_contact = osteotomy_logic.matrix_to_numpy(matrix)
                if moving_is_distal:
                    moving_cut_origin = np.asarray(distal_origin, dtype=float)
                    fixed_cut_origin = np.asarray(proximal_origin, dtype=float)
                else:
                    moving_cut_origin = np.asarray(proximal_origin, dtype=float)
                    fixed_cut_origin = np.asarray(distal_origin, dtype=float)

                moving_cut_h = np.ones(4, dtype=float)
                moving_cut_h[:3] = moving_cut_origin
                transformed_cut_origin = (H_contact @ moving_cut_h)[:3]
                contact_translation = fixed_cut_origin - transformed_cut_origin

                T_contact = np.eye(4, dtype=float)
                T_contact[:3, 3] = contact_translation
                matrix = osteotomy_logic.numpy_to_vtk_matrix(T_contact @ H_contact)

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
            proximal_polydata, distal_polydata = osteotomy_logic.clip_fragments(
                bone_polydata, proximal_origin, proximal_normal, shaft_axis)
            technique_note = " (OWO - distal plane is a virtual target only, no bone removed)"

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

        # Optionally hide the original segmentation so the corrected preview is not obscured by
        # overlapping surfaces. Preserve its previous 3D visibility and restore it on Remove preview.
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is not None and segmentation_node.GetDisplayNode() is not None:
            display_node = segmentation_node.GetDisplayNode()
            if not hasattr(self, "_original_segmentation_visibility_3d"):
                try:
                    self._original_segmentation_visibility_3d = bool(display_node.GetVisibility3D())
                except Exception:
                    self._original_segmentation_visibility_3d = bool(display_node.GetVisibility())
            if self.hide_original_in_preview.isChecked():
                if hasattr(display_node, "SetVisibility3D"):
                    display_node.SetVisibility3D(False)
                else:
                    display_node.SetVisibility(False)

        self.correction_status.setText(
            f"Preview active ({self.osteotomy_technique.currentText}): wedge {wedge_description}, "
            f"rotation {self.rotation_angle.value:.1f}\N{DEGREE SIGN}. Yellow = proximal fragment, "
            "turquoise = distal fragment, red = wedge REMOVE, beige = combined final model - the one "
            "matching the reference-plane choice shows the corrected position. If the direction "
            "looks wrong, tilt the distal plane differently or change the reference plane.")
        if alignment_report:
            self.correction_status.setText(self.correction_status.text + "\n" + alignment_report)

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

    def onResetFragmentShift(self):
        old_x = self.fragment_shift_x.blockSignals(True)
        old_y = self.fragment_shift_y.blockSignals(True)
        self.fragment_shift_x.setValue(0.0)
        self.fragment_shift_y.setValue(0.0)
        self.fragment_shift_x.blockSignals(old_x)
        self.fragment_shift_y.blockSignals(old_y)
        self._on_fragment_shift_changed()

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
        return value if description == "Innenrotation" else -value

    def _predict_measurements(self, side, bone, H, plane_origin, plane_normal, moving_is_distal):
        '''Move the landmarks lying on the MOVING side of the given (fixed) cut plane by the 4x4
        matrix H, read the wedge and torsion measurements, and restore the landmarks exactly.
        Returns {"wedge": value or None, "torsion": signed value or None} or None.'''
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
        result = {"wedge": None, "torsion": None}
        try:
            for i, point in enumerate(originals):
                on_distal_side = float(np.dot(point - o, n)) * distal_sign > 0.0
                if on_distal_side == moving_is_distal:
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
        finally:
            for i, point in enumerate(originals):
                fiducials.SetNthControlPointPositionWorld(i, *point)
        return result

    def _format_prediction(self, bone, predicted):
        deg = "\N{DEGREE SIGN}"
        parts = []
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

        def as_h(R):
            H = np.eye(4, dtype=float)
            H[:3, :3] = R
            H[:3, 3] = pivot - R @ pivot
            return H

        R_frontal = np.eye(3)
        if frontal > 0.01:
            best = None
            for sign in (1.0, -1.0):
                candidate = rot3(ap, sign * frontal)
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
                candidate = rot3(shaft, sign * torsion) @ R_frontal
                predicted = self._predict_measurements(side, bone, as_h(candidate), plane_origin, plane_normal, moving_is_distal)
                if predicted is None or predicted["torsion"] is None:
                    continue
                error = abs(predicted["torsion"] - float(self.target_rotation_value.value))
                if best is None or error < best[0]:
                    best = (error, candidate)
            if best is not None:
                R = best[1]
        predicted = self._predict_measurements(side, bone, as_h(R), plane_origin, plane_normal, moving_is_distal)
        return R, {"frontal": frontal, "torsion": torsion, "predicted": predicted}

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
        for box in (self.fragment_shift_x, self.fragment_shift_y,
                    self.direct_frontal_value, self.direct_torsion_value):
            was_blocked = box.blockSignals(True)
            box.setValue(0.0)
            box.blockSignals(was_blocked)
        self.actual_plane_angle_label.setText("Master/SECOND angle: --")

        # 4) re-read the CURRENT landmarks: measured values + calculated corrections
        if self.correction_mode.currentIndex == 0:
            self._on_target_values_changed()
        else:
            self._on_direct_values_changed()

        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is not None and segmentation_node.GetDisplayNode() is not None:
            segmentation_node.GetDisplayNode().SetVisibility3D(True)

        message = (f"Planning reset ({side} {bone.lower()}): preview, gizmo and cutting planes "
                   f"removed; values re-read from the current landmarks - "
                   f"{self.current_values_label.text}")
        self.osteotomy_status.setText(message)
        self.correction_status.setText("No preview.")

    def _set_cutting_plane_handles(self, side, bone, interactive):
        for name in (self._plane_node_name(side, bone), self._distal_plane_node_name(side, bone)):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetHandlesInteractive(bool(interactive))

    def _cwo_alignment_is_congruent(self):
        selector = getattr(self, "cwo_alignment_selector", None)
        return selector is None or selector.currentIndex == 0

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
                       f"correction - re-derive the SECOND plane (or use the bisecting button).")
        if side is not None and bone is not None:
            predicted = self._predict_measurements(side, bone, H, fixed_o, fixed_n, moving_is_distal)
            if predicted is not None:
                report += "\nPredicted after correction: " + self._format_prediction(bone, predicted)
        return osteotomy_logic.numpy_to_vtk_matrix(H), report

    # ------------------------------------------------------------------
    # Live X/Y fine shift: parent transform + gizmo restricted to the master plane's X/Y axes
    # ------------------------------------------------------------------
    def _fine_shift_node_names(self, side, bone):
        return (f"BoneAngleMeter {side} {bone.lower()} fine shift (X/Y)",
                f"BoneAngleMeter {side} {bone.lower()} fine shift gizmo")

    def _remove_fine_shift_observers(self):
        for node, tag in getattr(self, "_fine_shift_observers", []):
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
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
            combined_node.SetAndObservePolyData(combined_polydata)
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

    def onClearPreview(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        removed_any = False
        self._remove_fine_shift_observers()
        self._fine_shift_state = None
        self._set_cutting_plane_handles(side, bone, True)
        for name in self._preview_node_names(side, bone) + self._fine_shift_node_names(side, bone):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
                removed_any = True
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is not None and segmentation_node.GetDisplayNode() is not None and hasattr(self, "_original_segmentation_visibility_3d"):
            display_node = segmentation_node.GetDisplayNode()
            if hasattr(display_node, "SetVisibility3D"):
                display_node.SetVisibility3D(self._original_segmentation_visibility_3d)
            else:
                display_node.SetVisibility(self._original_segmentation_visibility_3d)
            del self._original_segmentation_visibility_3d
        self.correction_status.setText("Preview removed." if removed_any else "No preview to remove.")

    def onSaveScreenshot(self):
        file_name = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(), "Save screenshot", "", "PNG (*.png)")
        if not file_name:
            return
        if not file_name.lower().endswith(".png"):
            file_name += ".png"
        try:
            # ScreenCapture is a standard Slicer core module - captureImageFromView renders the
            # actual OpenGL content of the given view, which a plain QWidget.grab() can't
            # reliably do for a VTK render view.
            import ScreenCapture
            capture_logic = ScreenCapture.ScreenCaptureLogic()
            view = slicer.app.layoutManager().threeDWidget(0).threeDView()
            capture_logic.captureImageFromView(view, file_name)
            self.correction_status.setText(f"Screenshot saved: {file_name}")
        except Exception as e:
            self.correction_status.setText(f"Screenshot failed: {e}")

    def cleanup(self):
        self._remove_fine_shift_observers()
        self._stop_lasso()
        self._stop_guide_edit()

    def enter(self):
        # Called by Slicer whenever the user switches to this module's tab. Also call the base
        # class's own enter(), if it defines one, in case Slicer's module-switching machinery
        # relies on it.
        if hasattr(ScriptedLoadableModuleWidget, "enter"):
            ScriptedLoadableModuleWidget.enter(self)
        self._current_panel().activate()
        # The embedded clean-up editor observes the views while this module is shown, so its
        # effects also work when picked directly from its own effect buttons.
        editor = getattr(self, "_cleanup_editor_widget", None)
        if editor is not None:
            editor.setupViewObservations()

    def exit(self):
        # Called by Slicer whenever the user switches away to another module's tab.
        if hasattr(ScriptedLoadableModuleWidget, "exit"):
            ScriptedLoadableModuleWidget.exit(self)
        self._stop_lasso()
        self._stop_guide_edit()
        editor = getattr(self, "_cleanup_editor_widget", None)
        if editor is not None:
            editor.setActiveEffect(None)
            editor.removeViewObservations()
        self._current_panel().deactivate()

    def _current_panel(self):
        return self.side_stack.currentWidget()

    def _on_side_changed(self, index):
        old_panel = self.right_panel if index == 1 else self.left_panel
        old_panel.deactivate()
        self.side_stack.setCurrentIndex(index)
        self._current_panel().activate()

    def _on_osteotomy_bone_changed(self, index):
        bone = self.osteotomy_bone.currentText
        # Reset the target fields to this bone's default whenever the bone selection changes -
        # a target left over from the other bone (e.g. 90° carried over from Femur to Tibia)
        # would silently produce a wrong correction magnitude.
        self.target_wedge_value.setValue(self.WEDGE_TARGET_DEFAULT[bone])
        self.target_rotation_value.setValue(self.ROTATION_TARGET_DEFAULT[bone])
        self._update_plane_handle_interactivity()

    def _on_master_plane_changed(self, index=None):
        # Selecting the master plane must never move either cutting plane.
        # The surgeon positions the master manually and explicitly presses the
        # create/recalculate button when the second plane should be derived.
        self._update_plane_handle_interactivity()

    def _on_correction_mode_changed(self, index=None):
        direct = hasattr(self, "correction_mode") and self.correction_mode.currentText == "Direct correction"
        # Keep the compact UI simple: direct inputs are shown only in Direct mode; target/measurement
        # controls remain visible for reference but are disabled so they cannot affect geometry.
        for w in (getattr(self, "direct_frontal_value", None), getattr(self, "direct_torsion_value", None)):
            if w is not None:
                w.setVisible(direct)
        for w in (getattr(self, "target_wedge_value", None), getattr(self, "target_rotation_value", None)):
            if w is not None:
                w.setEnabled(not direct)
        if hasattr(self, "actual_plane_angle_label"):
            self.actual_plane_angle_label.setText("Master/SECOND angle: --")

    def _on_direct_values_changed(self, value=None):
        if hasattr(self, "calculated_wedge") and self.correction_mode.currentText == "Direct correction":
            self.calculated_wedge.setValue(float(self.direct_frontal_value.value))
            self.rotation_angle.setValue(float(self.direct_torsion_value.value))
        # Deliberately do not move planes automatically; explicit Create/update SECOND is required.

    def _update_actual_plane_angle(self, master_normal, second_normal):
        if not hasattr(self, "actual_plane_angle_label"):
            return
        a = osteotomy_logic.normalize(np.asarray(master_normal, dtype=float))
        b = osteotomy_logic.normalize(np.asarray(second_normal, dtype=float))
        dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(dot)))
        self.actual_plane_angle_label.setText(f"Master/SECOND angle: {angle:.2f} °")

    def _on_target_values_changed(self, value=None):
        """Refresh measured values, target-derived corrections and the derived plane.

        The UI deliberately accepts postoperative TARGET values only.  Delta angles are
        calculated internally and displayed read-only, so the surgeon never has to subtract
        current from target values by hand.
        """
        if not hasattr(self, "rotation_angle"):
            return
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        panel = self.left_panel if side == "left" else self.right_panel

        current_parts = []
        wedge_measurement = self._get_measurement_by_name(panel, self.WEDGE_MEASUREMENT_NAME[bone])
        if wedge_measurement is not None:
            ready, current_wedge, wedge_description = wedge_measurement()
            if ready:
                wedge_delta = self.target_wedge_value.value - current_wedge
                self.calculated_wedge.setValue(wedge_delta)
                current_parts.append(f"{self.WEDGE_MEASUREMENT_NAME[bone]} {current_wedge:.1f}°")
            else:
                self.calculated_wedge.setValue(0.0)

        ok, delta, rotation_message = self._calculated_rotation_delta(side, bone)
        if ok:
            self.rotation_angle.setValue(delta)
            # rotation_message already contains the signed current value. Keep the compact
            # current-anatomy line readable by extracting the first clause.
            current_parts.append(rotation_message.split(", target")[0])
        else:
            self.rotation_angle.setValue(0.0)

        if hasattr(self, "current_values_label"):
            self.current_values_label.setText(
                " | ".join(current_parts) if current_parts else
                "Current anatomy: place the required landmarks")

        # Deliberately do NOT update cutting planes here. Target values only update
        # the displayed calculated corrections. Plane geometry changes exclusively
        # when the surgeon presses "Create/update SECOND plane from MASTER".

    def _calculated_rotation_delta(self, side, bone):
        """Return signed target-current torsion correction. Measurement axes define the angle;
        the physical correction is still executed about the anatomical shaft line.
        """
        panel = self.left_panel if side == "left" else self.right_panel
        name = self.ROTATION_MEASUREMENT_NAME[bone]
        measurement = self._get_measurement_by_name(panel, name)
        if measurement is None:
            return False, 0.0, f"Measurement '{name}' not found."
        ready, value, description = measurement()
        if not ready:
            return False, 0.0, f"Place the landmarks for '{name}' ({side}) first."
        # Convert the magnitude-only measurement into a signed anatomical value. Target values
        # are entered in the positive named direction (Antetorsion for femur; Innenrotation for
        # tibia). This keeps the UI clinical while preserving a deterministic transform sign.
        if bone == "Femur":
            current_signed = value if description == "Antetorsion" else -value
        else:
            current_signed = value if description == "Innenrotation" else -value
        target_signed = self.target_rotation_value.value
        return True, target_signed - current_signed, (
            f"{name}: current {current_signed:+.1f}°, target {target_signed:+.1f}°")

    def _update_plane_handle_interactivity(self, index=None):
        """Master plane: translation + rotation. Derived plane: translation only.

        This is the key safety rule of the new workflow. X/Y translation of the derived plane
        allows centering the cut in the bone; Z translation changes cut spacing/shortening. Its
        orientation cannot be changed manually, so target correction cannot be corrupted.
        """
        if not hasattr(self, "reference_plane_selector"):
            return
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        master_is_distal = self.reference_plane_selector.currentText.startswith("Distal")
        for node_name, is_master in (
                (self._plane_node_name(side, bone), not master_is_distal),
                (self._distal_plane_node_name(side, bone), master_is_distal)):
            plane_node = slicer.mrmlScene.GetFirstNodeByName(node_name)
            if plane_node is None:
                continue
            display_node = plane_node.GetDisplayNode()
            if display_node is None:
                continue
            if hasattr(display_node, "SetHandlesInteractive"):
                display_node.SetHandlesInteractive(True)
            if hasattr(display_node, "SetTranslationHandleVisibility"):
                display_node.SetTranslationHandleVisibility(True)
            if hasattr(display_node, "SetRotationHandleVisibility"):
                display_node.SetRotationHandleVisibility(bool(is_master))

    def _apply_opacity(self, segmentation_node, opacity=None):
        display_node = segmentation_node.GetDisplayNode()
        if display_node is None:
            return
        if opacity is None:
            opacity = self.opacity_slider.value / 100.0
        # SetOpacity3D is the segmentation-specific API for 3D surface opacity; fall back to the
        # generic display node opacity if it's ever unavailable.
        if hasattr(display_node, "SetOpacity3D"):
            display_node.SetOpacity3D(opacity)
        else:
            display_node.SetOpacity(opacity)

    def onOpacityChanged(self, value):
        opacity = value / 100.0
        # Apply to every segmentation currently in the scene, not just the one this module
        # creates, so segmentations built separately (e.g. in Segment Editor) are covered too.
        for segmentation_node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
            self._apply_opacity(segmentation_node, opacity)

    def onSnapThresholdChanged(self, value):
        landmark_logic.set_bone_threshold(value)

    # ------------------------------------------------------------------
    # Landmark lists: reuse + re-link
    # ------------------------------------------------------------------
    @staticmethod
    def _get_or_create_fiducial_node(name):
        nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode") if n.GetName() == name]
        if not nodes:
            return slicer.modules.markups.logic().AddNewFiducialNode(name)

        def live_points(node):
            return sum(1 for i in range(node.GetNumberOfControlPoints())
                       if not node.GetNthControlPointLabel(i).endswith("(deleted)"))

        keep = max(nodes, key=live_points)
        for node in nodes:
            if node is not keep:
                slicer.mrmlScene.RemoveNode(node)
        return keep.GetID()

    @staticmethod
    def _relink_existing_landmarks(panel):
        '''Re-link landmarks to points already in the panel's list, matched by label; the newest
        point wins. Visibility is NOT used: the module hides all points whenever it is left.
        Deleted landmarks are relabelled "<name> (deleted)" (see SimpleLandmark.delete), so they
        never match.'''
        node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id)
        if node is None or node.GetNumberOfControlPoints() == 0:
            return 0
        by_name = {landmark.name: landmark for landmark in panel.landmarks}
        linked = []
        for i in reversed(range(node.GetNumberOfControlPoints())):
            landmark = by_name.get(node.GetNthControlPointLabel(i))
            if landmark is None or landmark.placed:
                continue
            landmark._id = i
            landmark.placed = True
            node.SetNthControlPointLocked(i, True)
            radius_text = node.GetAttribute(f"BoneAngleMeter.radius.{landmark.name}")
            if radius_text and hasattr(landmark, "radius"):
                try:
                    landmark.radius = float(radius_text)
                except ValueError:
                    pass
            linked.append(landmark)
        # Radius spinboxes were built before re-linking - bring them in line.
        for index in range(panel.measurement_stack.count):
            measurement_widget = panel.measurement_stack.widget(index)
            stack = getattr(measurement_widget, "landmark_stack", None)
            if stack is None:
                continue
            for j in range(stack.count):
                landmark_widget = stack.widget(j)
                spinbox = getattr(landmark_widget, "radius_spinbox", None)
                if spinbox is not None and hasattr(landmark_widget.landmark, "radius"):
                    was_blocked = spinbox.blockSignals(True)
                    spinbox.setValue(landmark_widget.landmark.radius)
                    spinbox.blockSignals(was_blocked)
        for landmark in linked:
            for callback in landmark.change_callbacks:
                callback()
        return len(linked)

    # ------------------------------------------------------------------
    # Segmentation clean-up: lasso (Scissors, erase outside) + small islands
    # ------------------------------------------------------------------
    def _ensure_cleanup_editor(self):
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
            self.segmentation_status.setText("No segmentation - run 'Apply' first.")
            return None
        volume_node = landmark_logic.get_bone_volume()
        if volume_node is None:
            volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if len(volumes) != 1:
                self.segmentation_status.setText("Could not identify the CT volume - run 'Apply' first.")
                return None
            volume_node = volumes[0]
            landmark_logic.set_bone_volume(volume_node)
        editor = self._cleanup_editor_widget
        if editor is None:
            editor = slicer.qMRMLSegmentEditorWidget()
            editor.setMRMLScene(slicer.mrmlScene)
            editor.setMaximumNumberOfUndoStates(5)
            editor_node = slicer.mrmlScene.GetFirstNodeByName("BoneAngleMeter segment editor")
            if editor_node is None:
                editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "BoneAngleMeter segment editor")
            editor.setMRMLSegmentEditorNode(editor_node)
            self._cleanup_editor_widget = editor
        if editor.segmentationNode() != segmentation_node:
            editor.setSegmentationNode(segmentation_node)
        if hasattr(editor, "setSourceVolumeNode"):
            editor.setSourceVolumeNode(volume_node)
        else:
            editor.setMasterVolumeNode(volume_node)
        editor.setCurrentSegmentID(segmentation_node.GetSegmentation().GetNthSegmentID(0))
        return editor

    def onLassoToggled(self, checked):
        '''Own lasso on the bone segmentation (keep inside, full depth). The embedded Segment
        Editor's Scissors did not receive mouse events in this module, so the lasso is handled
        like the guide tools: own view observers, orange trace, one stroke per click.'''
        if not checked:
            self._stop_lasso()   # landmark placement stays paused until a landmark is clicked
            return
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
            self.segmentation_status.setText("No segmentation - run 'Apply' first.")
            was_blocked = self.lasso_button.blockSignals(True)
            self.lasso_button.setChecked(False)
            self.lasso_button.blockSignals(was_blocked)
            return
        self._stop_guide_edit()
        self._pause_landmark_placement()  # no landmark placement while drawing
        self._hide_preview_for_guide_edit()  # cut on the visible ORIGINAL bone
        self._lasso_hidden_preview = list(getattr(self, "_guide_hidden_preview", []))
        self._guide_hidden_preview = []
        self._guide_active_tool = "bone_lasso"
        self._install_guide_observers()
        self.segmentation_status.setText(
            "Lasso active: in the 3D view hold the left mouse button and draw around the region to "
            "KEEP (orange trace). On release everything outside is removed through the full depth "
            "and the lasso switches off - rotate, then click 'Lasso' again for another stroke.")

    def _pause_landmark_placement(self):
        '''Stop placing / dragging the current landmark (landmarks stay visible) and give the mouse
        back to the view. Placement resumes when a landmark is clicked in the list, or via
        Next / Previous.'''
        panel = self._current_panel()
        measurement_widget = panel.measurement_stack.currentWidget()
        if measurement_widget is not None and getattr(measurement_widget, "enabled", False):
            current = measurement_widget.landmark_stack.currentWidget()
            if current is not None:
                current.landmark.stop_interaction()
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
        interaction_node.SetPlaceModePersistence(0)

    def _stop_lasso(self):
        button = getattr(self, "lasso_button", None)
        was_active = button is not None and button.checked
        if getattr(self, "_guide_active_tool", None) == "bone_lasso":
            self._remove_guide_observers()
            self._guide_active_tool = None
        if button is not None and button.checked:
            was_blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(was_blocked)
        for node_id in getattr(self, "_lasso_hidden_preview", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(True)
        self._lasso_hidden_preview = []
        if was_active and getattr(self, "segmentation_status", None) is not None:
            self.segmentation_status.setText("Lasso off.")

    def onRemoveSmallIslands(self):
        editor = self._ensure_cleanup_editor()
        if editor is None:
            return
        self._stop_lasso()
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        self._bone_push_undo(segmentation_node)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            editor.setActiveEffectByName("Islands")
            effect = editor.activeEffect()
            effect.setParameter("Operation", "REMOVE_SMALL_ISLANDS")
            effect.setParameter("MinimumSize", str(int(self.island_min_size.value)))
            effect.self().onApply()
            editor.setActiveEffect(None)
        finally:
            qt.QApplication.restoreOverrideCursor()
        self.segmentation_status.setText(
            f"Islands smaller than {int(self.island_min_size.value)} voxels removed.")

    def onDeleteSegmentationEdits(self):
        '''Discard every clean-up edit: restore the segmentation as it was before the first
        lasso / islands step (since the last 'Apply').'''
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        baseline = getattr(self, "_bone_baseline", None)
        if segmentation_node is None or baseline is None:
            self.segmentation_status.setText("No clean-up edits to delete.")
            return
        self._stop_lasso()
        self._bone_set_labelmap(segmentation_node, baseline)
        self._bone_baseline = None
        self.segmentation_status.setText("Clean-up edits deleted - segmentation restored. Start again with the lasso.")

    # --- clean-up baseline + lasso cut for the bone segmentation --------------
    def _bone_labelmap_copy(self, segmentation_node):
        segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)
        image = slicer.vtkOrientedImageData()
        segmentation_node.GetBinaryLabelmapRepresentation(segment_id, image)
        copy = slicer.vtkOrientedImageData()
        copy.DeepCopy(image)
        return copy

    def _bone_set_labelmap(self, segmentation_node, image):
        segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)
        slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
            image, segmentation_node, segment_id, slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE)

    def _bone_push_undo(self, segmentation_node):
        '''Before the FIRST clean-up edit keep a copy of the segmentation ('Delete' restores it).'''
        if segmentation_node is None or getattr(self, "_bone_baseline", None) is not None:
            return
        self._bone_baseline = self._bone_labelmap_copy(segmentation_node)

    def _bone_lasso_apply(self, polygon):
        '''Keep only bone voxels whose projection lies inside the screen polygon (full depth).'''
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None:
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._bone_push_undo(segmentation_node)
            image = self._bone_labelmap_copy(segmentation_node)
            extent = image.GetExtent()
            dims = (extent[5] - extent[4] + 1, extent[3] - extent[2] + 1, extent[1] - extent[0] + 1)
            voxels = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims)
            kk, jj, ii = np.nonzero(voxels)
            if len(kk) == 0:
                return
            matrix = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(matrix)
            M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
            renderer = self._guide_renderer()
            render_window = self._guide_view.renderWindow()
            width, height = render_window.GetSize()
            camera = renderer.GetActiveCamera()
            pm = camera.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)
            P = np.array([[pm.GetElement(r, c) for c in range(4)] for r in range(4)])
            A = (P @ M).astype(np.float32)          # voxel index -> clip coordinates
            ijk = np.stack([ii + extent[0], jj + extent[2], kk + extent[4]], axis=1).astype(np.float32)
            clip = ijk @ A[:, :3].T + A[:, 3]
            xs = (clip[:, 0] / clip[:, 3] + 1.0) * 0.5 * width
            ys = (clip[:, 1] / clip[:, 3] + 1.0) * 0.5 * height
            poly = np.asarray(polygon, dtype=np.float32)
            keep = np.zeros(len(xs), dtype=bool)
            box = (xs >= poly[:, 0].min()) & (xs <= poly[:, 0].max()) & (ys >= poly[:, 1].min()) & (ys <= poly[:, 1].max())
            candidates = np.nonzero(box)[0]
            if len(candidates):
                keep[candidates] = guide_logic.points_in_polygon(np.c_[xs[candidates], ys[candidates]], poly)
            removed = int((~keep).sum())
            voxels[kk[~keep], jj[~keep], ii[~keep]] = 0
            image.GetPointData().GetScalars().Modified()
            image.Modified()
            self._bone_set_labelmap(segmentation_node, image)
            self.segmentation_status.setText(
                f"Lasso: {removed} voxels outside removed, {int(keep.sum())} kept. 'Delete' restores "
                f"the segmentation from before the clean-up.")
        finally:
            qt.QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    # Guide planning - step 1: footprint on the bone SURFACE (mesh), editing, extrusion
    # ------------------------------------------------------------------
    GUIDE_MESH_NAME = "BoneAngleMeter guide bone surface"
    GUIDE_FOOTPRINT_NAME = "BoneAngleMeter guide footprint"
    GUIDE_BODY_NAME = "BoneAngleMeter guide body"
    GUIDE_SEGMENTATION_NAME = "BoneAngleMeter guide"   # earlier voxel-based version (hidden if present)

    def _guide_direction_frame(self, side, bone):
        '''(d, shaft, across, error): d = unit vector to the chosen guide side, shaft = proximal->
        distal, across = d x shaft. Cranial from the limb's handedness: with shaft proximal->distal
        and ml medial->lateral, cranial = shaft x ml for a LEFT and ml x shaft for a RIGHT limb.'''
        shaft, ml, _, error = self._get_anatomical_frame(side, bone)
        if error:
            return None, None, None, error
        shaft = guide_logic.unit(shaft)
        ml = np.asarray(ml, dtype=float)
        ml = guide_logic.unit(ml - float(np.dot(ml, shaft)) * shaft)
        cranial = guide_logic.unit(np.cross(shaft, ml) if side == "left" else np.cross(ml, shaft))
        d = {"Cranial": cranial, "Caudal": -cranial, "Lateral": ml, "Medial": -ml}[self.guide_side_selector.currentText]
        return d, shaft, guide_logic.unit(np.cross(d, shaft)), None

    def _guide_mesh_node(self):
        return slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_MESH_NAME)

    @staticmethod
    def _guide_outer(mesh):
        '''Outer-surface mask of the guide side (all True for meshes made before it existed).'''
        if mesh.GetPointData().GetArray("Outer") is None:
            return np.ones(mesh.GetNumberOfPoints(), dtype=bool)
        return guide_logic.get_selection(mesh, name="Outer")

    def onSuggestGuideFootprint(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        self._stop_guide_edit()
        plane_nodes = [slicer.mrmlScene.GetFirstNodeByName(name) for name in
                       (self._plane_node_name(side, bone), self._distal_plane_node_name(side, bone))]
        plane_nodes = [node for node in plane_nodes if node is not None]
        if not plane_nodes:
            self.guide_status.setText("Insert the cutting plane(s) in Osteotomy Planning first.")
            return
        d, shaft, across, error = self._guide_direction_frame(side, bone)
        if error:
            self.guide_status.setText(error)
            return
        bone_polydata = self._get_bone_polydata()
        if bone_polydata is None or bone_polydata.GetNumberOfPoints() == 0:
            self.guide_status.setText("No bone surface - run 'Apply' in 3D Segmentation first.")
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            origins = [np.asarray(self._get_plane_origin_normal(node)[0], dtype=float) for node in plane_nodes]
            centre = np.mean(origins, axis=0)
            t_values = [float(np.dot(o - centre, shaft)) for o in origins]
            margin = float(self.guide_footprint_margin.value)
            half_width = 0.5 * float(self.guide_footprint_width.value)
            t_min, t_max = min(t_values) - margin, max(t_values) + margin
            mesh = guide_logic.extract_roi_mesh(bone_polydata, centre, shaft, across, d,
                                                (t_min - 15.0, t_max + 15.0), half_width + 20.0, 45.0,
                                                closest_point=centre)
            if mesh.GetNumberOfPoints() == 0:
                self.guide_status.setText("No bone surface found around the osteotomy.")
                return
            # Re-centre across the bone (cross-section centroid at the level of the cuts); the
            # diaphyseal shaft axis can lie outside a curved metaphysis.
            P, _ = guide_logic.points_and_normals(mesh)
            rel = P - centre
            slab = np.abs(rel @ shaft - 0.5 * (t_min + t_max)) < 3.0
            if np.any(slab):
                shift = P[slab].mean(axis=0) - centre
                centre = centre + shift - float(np.dot(shift, shaft)) * shaft
            selected = guide_logic.orient_and_suggest(mesh, centre, shaft, across, d, t_min, t_max, half_width)
            outer = guide_logic.outer_mask(mesh, centre, shaft, across, d)
            selected = selected & outer
            guide_logic.set_selection(mesh, selected)
            guide_logic.set_selection(mesh, outer, name="Outer")

            mesh_node = self._guide_mesh_node()
            if mesh_node is None:
                mesh_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_MESH_NAME)
                mesh_node.CreateDefaultDisplayNodes()
            mesh_node.SetAndObservePolyData(mesh)
            mesh_node.GetDisplayNode().SetVisibility(False)   # computation only; you see the bone
            self._guide_undo_stack = []
            self._set_guide_body(None)   # an old body does not match the new footprint
            old = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_SEGMENTATION_NAME)
            if old is not None and old.GetDisplayNode() is not None:
                old.GetDisplayNode().SetVisibility(False)
            area = self._update_guide_footprint_display()
            self.guide_status.setText(
                f"Footprint suggested on the {self.guide_side_selector.currentText.lower()} side of the "
                f"{side} {bone.lower()}: {int(selected.sum())} surface points, ~{area:.0f} mm\N{SUPERSCRIPT TWO}. "
                f"Refine with brush / lasso, then 'Extrude guide body'.")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def _update_guide_footprint_display(self):
        mesh_node = self._guide_mesh_node()
        if mesh_node is None:
            return 0.0
        mesh = mesh_node.GetPolyData()
        patch = guide_logic.patch_from_selection(mesh, guide_logic.get_selection(mesh) & self._guide_outer(mesh), lift=0.08)
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_FOOTPRINT_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_FOOTPRINT_NAME)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(1.0, 0.5, 0.05)
            node.GetDisplayNode().SetScalarVisibility(False)
        node.SetAndObservePolyData(patch)
        node.GetDisplayNode().SetVisibility(True)
        if patch.GetNumberOfCells() == 0:
            return 0.0
        mass = vtk.vtkMassProperties()
        mass.SetInputData(patch)
        mass.Update()
        return float(mass.GetSurfaceArea())

    # --- editing tools -------------------------------------------------
    def onGuideToolToggled(self, key, checked):
        if not checked:
            if getattr(self, "_guide_active_tool", None) == key:
                self._stop_guide_edit()   # placement stays paused until a landmark is clicked
            return
        if self._guide_mesh_node() is None:
            self.guide_status.setText("Suggest a footprint first.")
            button = self._guide_tool_buttons[key]
            was_blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(was_blocked)
            return
        for other, button in self._guide_tool_buttons.items():
            if other != key and button.checked:
                was_blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(was_blocked)
        if getattr(self, "_guide_active_tool", None) in (None, "bone_lasso"):
            self._stop_lasso()
            self._pause_landmark_placement()   # no landmark placement while editing
            self._hide_preview_for_guide_edit()
        self._guide_active_tool = key
        self._install_guide_observers()
        texts = {"brush_add": "Brush +: hold the left mouse button and move over the bone surface to colour it.",
                 "brush_remove": "Brush -: hold the left mouse button and move over the coloured area to remove it.",
                 "lasso_add": "Lasso +: draw around the area to colour (only the visible surface is affected).",
                 "lasso_remove": "Lasso -: draw around the area to uncolour (only the visible surface is affected)."}
        hint = (" Drag on the background to rotate." if key.startswith("brush")
                else " The lasso switches off after each stroke (click it again for the next one).")
        self.guide_status.setText(texts[key] + hint + " Shift+drag pans, right button / wheel zooms. "
                                  "Click the tool again to stop.")

    def _hide_preview_for_guide_edit(self):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        self._guide_hidden_preview = []
        for name in self._preview_node_names(side, bone):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.IsA("vtkMRMLModelNode") and node.GetDisplayNode() is not None \
                    and node.GetDisplayNode().GetVisibility():
                node.GetDisplayNode().SetVisibility(False)
                self._guide_hidden_preview.append(node.GetID())
        bone_segmentation = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if bone_segmentation is not None and bone_segmentation.GetDisplayNode() is not None:
            bone_segmentation.GetDisplayNode().SetVisibility3D(True)

    def _install_guide_observers(self):
        self._remove_guide_observers()
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        interactor = view.interactor()
        self._guide_view = view
        self._guide_tags = {}
        for event in ("LeftButtonPressEvent", "MouseMoveEvent", "LeftButtonReleaseEvent"):
            self._guide_tags[event] = interactor.AddObserver(event, self._on_guide_mouse, 1.0)
        self._guide_observers = [(interactor, tag) for tag in self._guide_tags.values()]
        if self._guide_active_tool == "bone_lasso":
            return   # the bone lasso needs no surface locators
        mesh = self._guide_mesh_node().GetPolyData()
        self._guide_cell_locator = vtk.vtkCellLocator()
        self._guide_cell_locator.SetDataSet(mesh)
        self._guide_cell_locator.BuildLocator()
        self._guide_point_locator = vtk.vtkPointLocator()
        self._guide_point_locator.SetDataSet(mesh)
        self._guide_point_locator.BuildLocator()

    def _remove_guide_observers(self):
        for interactor, tag in getattr(self, "_guide_observers", []):
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._guide_observers = []
        self._guide_drag = False
        self._clear_lasso_feedback()

    def _abort_guide_event(self, caller, event):
        tag = getattr(self, "_guide_tags", {}).get(event)
        if tag is not None:
            command = caller.GetCommand(tag)
            if command is not None:
                command.SetAbortFlag(1)

    def _on_guide_mouse(self, caller, event):
        tool = getattr(self, "_guide_active_tool", None)
        if tool is None:
            return
        x, y = caller.GetEventPosition()
        if event == "LeftButtonPressEvent":
            # With Shift / Ctrl / Alt the mouse keeps its normal function (Shift+drag = pan).
            if caller.GetShiftKey() or caller.GetControlKey() or caller.GetAltKey():
                self._guide_drag = False
                return
            if tool.startswith("brush"):
                hit = self._guide_pick(x, y)
                if hit is None:
                    self._guide_drag = False      # drag started on the background: rotate the view
                    return
                self._guide_push_undo()
                self._guide_drag = True
                self._guide_brush_at(x, y, tool == "brush_add", hit=hit)
            else:
                if tool != "bone_lasso":
                    self._guide_push_undo()
                self._guide_drag = True
                self._guide_lasso_points = [(x, y)]
                self._update_lasso_feedback()
            self._abort_guide_event(caller, event)
        elif event == "MouseMoveEvent":
            if not self._guide_drag:
                return
            if tool.startswith("brush"):
                self._guide_brush_at(x, y, tool == "brush_add")
            else:
                last = self._guide_lasso_points[-1]
                if abs(x - last[0]) + abs(y - last[1]) >= 3:
                    self._guide_lasso_points.append((x, y))
                    self._update_lasso_feedback()
            self._abort_guide_event(caller, event)
        elif event == "LeftButtonReleaseEvent":
            if not self._guide_drag:
                return
            self._guide_drag = False
            if tool == "bone_lasso":
                points = list(getattr(self, "_guide_lasso_points", []))
                self._clear_lasso_feedback()
                self._abort_guide_event(caller, event)
                if len(points) >= 3:
                    self._bone_lasso_apply(points)
                qt.QTimer.singleShot(0, lambda: self.lasso_button.setChecked(False))
                return
            if tool.startswith("lasso"):
                points = list(getattr(self, "_guide_lasso_points", []))
                self._clear_lasso_feedback()
                if len(points) >= 3:
                    self._guide_apply_lasso(points, tool == "lasso_add")
            self._refresh_guide_body_if_present()
            self._abort_guide_event(caller, event)
            if tool.startswith("lasso"):
                # One lasso per click: switch the tool off so the view can be rotated/panned again.
                button = self._guide_tool_buttons.get(tool)
                qt.QTimer.singleShot(0, lambda b=button: b.setChecked(False) if b is not None else None)

    def _guide_renderer(self):
        return self._guide_view.renderWindow().GetRenderers().GetFirstRenderer()

    def _guide_pick(self, x, y):
        '''First bone-surface point under the display position (x, y), or None.'''
        renderer = self._guide_renderer()
        ends = []
        for z in (0.0, 1.0):
            renderer.SetDisplayPoint(float(x), float(y), z)
            renderer.DisplayToWorld()
            w = renderer.GetWorldPoint()
            ends.append([w[0] / w[3], w[1] / w[3], w[2] / w[3]])
        t = vtk.reference(0.0)
        sub_id = vtk.reference(0)
        hit = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        if not self._guide_cell_locator.IntersectWithLine(ends[0], ends[1], 0.0, t, hit, pcoords, sub_id):
            return None
        return hit

    def _guide_brush_at(self, x, y, add, hit=None):
        renderer = self._guide_renderer()
        if hit is None:
            hit = self._guide_pick(x, y)
            if hit is None:
                return
        ids = vtk.vtkIdList()
        self._guide_point_locator.FindPointsWithinRadius(float(self.guide_brush_radius.value), hit, ids)
        if ids.GetNumberOfIds() == 0:
            return
        idx = np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=np.int64)
        mesh = self._guide_mesh_node().GetPolyData()
        P, N = guide_logic.points_and_normals(mesh)
        camera = np.asarray(renderer.GetActiveCamera().GetPosition(), dtype=float)
        facing = np.sum(N[idx] * (camera - P[idx]), axis=1) > 0.0
        outer = self._guide_outer(mesh)[idx]
        selection = guide_logic.get_selection(mesh)
        selection[idx[facing & outer]] = add
        guide_logic.set_selection(mesh, selection)
        self._update_guide_footprint_display()

    def _guide_apply_lasso(self, polygon, add):
        from vtk.util.numpy_support import vtk_to_numpy
        render_window = self._guide_view.renderWindow()
        renderer = self._guide_renderer()
        render_window.Render()
        width, height = render_window.GetSize()
        mesh = self._guide_mesh_node().GetPolyData()
        P, N = guide_logic.points_and_normals(mesh)
        camera = renderer.GetActiveCamera()
        m = camera.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)
        M = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])
        hom = np.c_[P, np.ones(len(P))] @ M.T
        ndc = hom[:, :3] / hom[:, 3:4]
        vx0, vy0, vx1, vy1 = renderer.GetViewport()
        xs = (ndc[:, 0] + 1.0) * 0.5 * (vx1 - vx0) * width + vx0 * width
        ys = (ndc[:, 1] + 1.0) * 0.5 * (vy1 - vy0) * height + vy0 * height
        depth = (ndc[:, 2] + 1.0) * 0.5
        zbuffer = vtk.vtkFloatArray()
        render_window.GetZbufferData(0, 0, width - 1, height - 1, zbuffer)
        Z = vtk_to_numpy(zbuffer).reshape(height, width)
        xi = np.clip(np.round(xs).astype(int), 0, width - 1)
        yi = np.clip(np.round(ys).astype(int), 0, height - 1)
        on_screen = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        visible = on_screen & (depth <= Z[yi, xi] + 2e-3)
        facing = np.sum(N * (np.asarray(camera.GetPosition(), dtype=float) - P), axis=1) > 0.0
        inside = guide_logic.points_in_polygon(np.c_[xs, ys], polygon)
        affected = inside & visible & facing & self._guide_outer(mesh)
        selection = guide_logic.get_selection(mesh)
        selection[affected] = add
        guide_logic.set_selection(mesh, selection)
        area = self._update_guide_footprint_display()
        self.guide_status.setText(f"{'Added' if add else 'Removed'} {int(affected.sum())} surface points; "
                                  f"footprint now ~{area:.0f} mm\N{SUPERSCRIPT TWO}.")

    def _update_lasso_feedback(self):
        renderer = self._guide_renderer()
        if getattr(self, "_guide_lasso_actor", None) is None:
            self._guide_lasso_polydata = vtk.vtkPolyData()
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(self._guide_lasso_polydata)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.55, 0.05)
            actor.GetProperty().SetLineWidth(2)
            renderer.AddActor2D(actor)
            self._guide_lasso_actor = actor
        points = vtk.vtkPoints()
        for x, y in self._guide_lasso_points:
            points.InsertNextPoint(float(x), float(y), 0.0)
        lines = vtk.vtkCellArray()
        n = len(self._guide_lasso_points)
        if n >= 2:
            lines.InsertNextCell(n + 1)
            for i in range(n):
                lines.InsertCellPoint(i)
            lines.InsertCellPoint(0)
        self._guide_lasso_polydata.SetPoints(points)
        self._guide_lasso_polydata.SetLines(lines)
        self._guide_lasso_polydata.Modified()
        self._guide_view.renderWindow().Render()

    def _clear_lasso_feedback(self):
        actor = getattr(self, "_guide_lasso_actor", None)
        if actor is not None:
            try:
                self._guide_renderer().RemoveActor2D(actor)
                self._guide_view.renderWindow().Render()
            except Exception:
                pass
        self._guide_lasso_actor = None

    def _guide_body_polydata(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if node is None or node.GetPolyData() is None or node.GetPolyData().GetNumberOfCells() == 0:
            return None
        copy = vtk.vtkPolyData()
        copy.DeepCopy(node.GetPolyData())
        return copy

    def _set_guide_body(self, polydata):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if polydata is None:
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
            return
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_BODY_NAME)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(0.35, 0.7, 0.95)
            node.GetDisplayNode().SetOpacity(0.85)
        node.SetAndObservePolyData(polydata)

    def _guide_push_undo(self):
        '''One undo step = footprint selection AND guide body (None if there was none).'''
        mesh_node = self._guide_mesh_node()
        if mesh_node is None:
            return
        self._guide_undo_stack.append((guide_logic.get_selection(mesh_node.GetPolyData()).copy(),
                                       self._guide_body_polydata()))
        del self._guide_undo_stack[:-20]

    def onGuideUndo(self):
        mesh_node = self._guide_mesh_node()
        if mesh_node is None or not getattr(self, "_guide_undo_stack", []):
            self.guide_status.setText("Nothing to undo.")
            return
        selection, body = self._guide_undo_stack.pop()
        guide_logic.set_selection(mesh_node.GetPolyData(), selection)
        area = self._update_guide_footprint_display()
        self._set_guide_body(body)
        self.guide_status.setText(f"Undone; footprint ~{area:.0f} mm\N{SUPERSCRIPT TWO}"
                                  f"{', guide body restored' if body is not None else ', no guide body'}.")

    def onGuideDelete(self):
        '''Clear the whole footprint and remove the guide body (the surface stays, so painting can
        start from scratch without a new suggestion).'''
        mesh_node = self._guide_mesh_node()
        if mesh_node is None:
            self.guide_status.setText("Nothing to delete.")
            return
        mesh = mesh_node.GetPolyData()
        guide_logic.set_selection(mesh, np.zeros(mesh.GetNumberOfPoints(), dtype=bool))
        self._update_guide_footprint_display()
        self._set_guide_body(None)
        self._guide_undo_stack = []
        self.guide_status.setText("Footprint and guide body deleted. Start again with 'Suggest "
                                  "footprint' or paint with Brush + / Lasso +.")

    def _refresh_guide_body_if_present(self):
        '''After a footprint edit: an existing body is re-extruded so it always matches.'''
        if slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME) is not None:
            self._extrude_guide_body(push_undo=False)

    def _stop_guide_edit(self):
        self._remove_guide_observers()
        for button in getattr(self, "_guide_tool_buttons", {}).values():
            if button.checked:
                was_blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(was_blocked)
        self._guide_active_tool = None
        for node_id in getattr(self, "_guide_hidden_preview", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(True)
        self._guide_hidden_preview = []

    # --- extrusion --------------------------------------------------------
    def onExtrudeGuideBody(self):
        self._stop_guide_edit()
        self._extrude_guide_body(push_undo=True)

    def _extrude_guide_body(self, push_undo=True):
        mesh_node = self._guide_mesh_node()
        if mesh_node is None:
            self.guide_status.setText("Suggest a footprint first.")
            return
        mesh = mesh_node.GetPolyData()
        selection = guide_logic.get_selection(mesh) & self._guide_outer(mesh)
        patch = guide_logic.clean_patch(guide_logic.patch_from_selection(mesh, selection, lift=0.0))
        if push_undo:
            self._guide_push_undo()
        if patch.GetNumberOfCells() == 0:
            self._set_guide_body(None)
            self.guide_status.setText("The footprint is empty - no guide body.")
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            gap = float(self.guide_gap.value)
            thickness = float(self.guide_thickness.value)
            body = guide_logic.extrude_patch(patch, gap, thickness)
            self._set_guide_body(body)
            open_edges = guide_logic.open_edge_count(body)
            mass = vtk.vtkMassProperties()
            mass.SetInputData(body)
            mass.Update()
            closed = "closed" if open_edges == 0 else f"NOT closed ({open_edges} open edges)"
            self.guide_status.setText(
                f"Guide body extruded: fit gap {gap:.2f} mm, wall {thickness:.1f} mm, volume "
                f"~{mass.GetVolume():.0f} mm\N{SUPERSCRIPT THREE}, mesh {closed}. It follows further "
                f"footprint edits; Undo also removes/restores it.")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onApplySegmentation(self):
        self.segmentation_status.setText("working...")

        # Get volume data
        volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if len(volume_nodes) == 0:
            self.segmentation_status.setText("No data found")
            return
        if len(volume_nodes) > 1:
            self.segmentation_status.setText("Multiple volumes found. Only one volume is supported currently.")
            return
        volume_node = volume_nodes[0]
        landmark_logic.set_bone_volume(volume_node)

        # Create segmentation
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is not None:
            slicer.mrmlScene.RemoveNode(segmentation_node)
        segmentation_node = slicer.mrmlScene.AddNewNodeByClassWithID("vtkMRMLSegmentationNode", "", self.SEGMENTATION_NODE_NAME)
        segmentation_node.CreateDefaultDisplayNodes() # only needed for display
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        segmentation_node.SetName(self.SEGMENTATION_NODE_NAME)
        segment_id = segmentation_node.GetSegmentation().AddEmptySegment("Bones")

        # Create segment editor to get access to effects
        segment_editor_widget = slicer.qMRMLSegmentEditorWidget()
        segment_editor_widget.setMRMLScene(slicer.mrmlScene)
        segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
        segment_editor_widget.setSegmentationNode(segmentation_node)
        segment_editor_widget.setCurrentSegmentID(segment_id)
        if hasattr(segment_editor_widget, "setSourceVolumeNode"):
            segment_editor_widget.setSourceVolumeNode(volume_node)
        else:
            segment_editor_widget.setMasterVolumeNode(volume_node)

        # Thresholding
        segment_editor_widget.setActiveEffectByName("Threshold")
        effect = segment_editor_widget.activeEffect()
        effect.setParameter("MinimumThreshold", f"{self.threshold_lower.value}")
        effect.setParameter("MaximumThreshold", f"{self.threshold_upper.value}")
        effect.self().onApply()

        # Clean up
        segment_editor_widget = None
        slicer.mrmlScene.RemoveNode(segment_editor_node)

        # Smoothing
        segmentation_node.GetSegmentation().SetConversionParameter("Smoothing factor",f"{self.smoothing.value}")

        # Make segmentation results visible in 3D
        segmentation_node.CreateClosedSurfaceRepresentation()

        # Center the 3d View on the scene
        layout_manager = slicer.app.layoutManager()
        three_d_Widget = layout_manager.threeDWidget(0)
        three_d_view = three_d_Widget.threeDView()
        three_d_view.resetFocalPoint()

        # Make segmentation invisible in sliced and set colors
        segmentation_node.GetDisplayNode().SetAllSegmentsVisibility2DOutline(False)
        segmentation_node.GetDisplayNode().SetAllSegmentsVisibility2DFill(False)
        segmentation_node.GetSegmentation().GetSegment(segment_id).SetColor(241/255, 241/255, 145/255)
        self.onOpacityChanged(self.opacity_slider.value)

        self._stop_lasso()
        self._ensure_cleanup_editor()
        self._bone_baseline = None   # a new segmentation starts a new clean-up
        self.segmentation_status.setText("ok")

class MeasurementsPanel(qt.QWidget):
    '''
    Panel containing one side's measurement UI (landmark stack of measurements). Embedded
    directly in the module's own sidebar via a side selector + stacked widget, so the whole
    module lives in a single page next to the 3D/slice views with no separate window at all.
    '''
    def __init__(self, side, markup_node_id, base_widget):
        super().__init__()

        self.markup_node_id = markup_node_id
        self.base_widget = base_widget

        if side.lower() == 'left':
            self.side = 'left'
        elif side.lower() == 'right':
            self.side = 'right'
        else:
            raise ValueError(f"side should be eigher 'left' or 'right'. Found '{side}'.")

        self.import_landmarks_button = qt.QPushButton("Import")
        self.import_landmarks_button.setToolTip(f"Import {self.side} landmarks from a CSV file.")
        self.import_landmarks_button.clicked.connect(self._import_landmarks)
        self.import_landmarks_button.setDefault(False)
        self.import_landmarks_button.setAutoDefault(False)
        self.export_landmarks_button = qt.QPushButton("Export")
        self.export_landmarks_button.setToolTip(f"Export {self.side} landmarks to a CSV file.")
        self.export_landmarks_button.clicked.connect(self._export_landmarks)
        self.export_landmarks_button.setDefault(False)
        self.export_landmarks_button.setAutoDefault(False)
        self.export_measurements_button = qt.QPushButton("Export Results")
        self.export_measurements_button.setToolTip(f"Export {self.side} measurement results to a CSV file.")
        self.export_measurements_button.clicked.connect(self._export_measurements)
        self.export_measurements_button.setDefault(False)
        self.export_measurements_button.setAutoDefault(False)

        self.measurement_list = qt.QListWidget(self)
        self.measurement_stack = qt.QStackedWidget(self)
        self.measurement_list.currentRowChanged.connect(self._change_row)

        # Model nodes for visualizing fitted spheres (femur head / condyle centers), keyed by
        # sphere label (e.g. "femur head"). Created lazily the first time they're needed.
        self.sphere_nodes = {}
        self.spheres_visible = True

        # Model nodes for visualizing the axis lines compared in each measurement, keyed by
        # axis label (e.g. "Femoral shaft axis"). Colors are assigned by position (1st axis /
        # 2nd axis) from AXIS_PALETTE, consistently across every measurement.
        self.axis_nodes = {}
        self.axes_visible = True
        self.AXIS_PALETTE = [(0.15, 0.85, 0.95), (0.95, 0.35, 0.85), (0.95, 0.65, 0.15), (0.55, 0.85, 0.25)]

        self.show_spheres_checkbox = qt.QCheckBox("Show fitted spheres")
        self.show_spheres_checkbox.setChecked(True)
        self.show_spheres_checkbox.setToolTip(
            "Show/hide the translucent spheres fitted to the femur head and condyles in the 3D view.")
        self.show_spheres_checkbox.toggled.connect(self._on_show_spheres_toggled)

        self.show_axes_checkbox = qt.QCheckBox("Show axis lines")
        self.show_axes_checkbox.setChecked(True)
        self.show_axes_checkbox.setToolTip(
            "Show/hide the colored lines for the two axes compared by the current measurement.")
        self.show_axes_checkbox.toggled.connect(self._on_show_axes_toggled)

        self.show_labels_checkbox = qt.QCheckBox("Show landmark labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.setToolTip("Show/hide the text labels next to each landmark point in the 3D view.")
        self.show_labels_checkbox.toggled.connect(self._on_show_labels_toggled)

        self.delete_all_button = qt.QPushButton("Delete All")
        self.delete_all_button.setToolTip(f"Removes every placed {self.side} landmark so they can be re-placed from scratch.")
        self.delete_all_button.clicked.connect(self._delete_all_landmarks)
        self.delete_all_button.setDefault(False)
        self.delete_all_button.setAutoDefault(False)

        # Create deep-copy of all landmarks for this side
        self.landmarks = deepcopy(LANDMARKS)
        for landmark in self.landmarks:
            landmark.set_markups_node_id(self.markup_node_id)

        # Create all measurements
        for measurement in MEASUREMENTS:
            m = measurement()
            m.set_side(self.side)
            m.register_landmarks(self.landmarks)
            self.measurement_stack.addWidget(MeasurementWidget(m, self))
            self.measurement_list.addItem(getattr(m, "display_name", m.name))
        
        self.measurement_list.setMaximumHeight(140)

        # 2x2 grid instead of a single row: a QHBoxLayout sums every button's minimum width
        # together, which was the main thing preventing the sidebar from being dragged narrower
        # than that sum. A grid only requires as much width as the single widest button.
        checkboxes_grid = qt.QGridLayout()
        checkboxes_grid.addWidget(self.show_spheres_checkbox, 0, 0)
        checkboxes_grid.addWidget(self.show_axes_checkbox, 0, 1)
        checkboxes_grid.addWidget(self.show_labels_checkbox, 1, 0)
        checkboxes_row_widget = qt.QWidget()
        checkboxes_row_widget.setLayout(checkboxes_grid)

        controls_layout = qt.QVBoxLayout()
        controls_layout.addWidget(self.measurement_list)
        controls_layout.addWidget(checkboxes_row_widget)
        controls_widget = qt.QWidget()
        controls_widget.setLayout(controls_layout)

        # Same reasoning as the checkboxes above: a 2x2 grid instead of one QHBoxLayout row, so
        # the sidebar's minimum width is set by the single widest button ("Export Results"), not
        # by the sum of all four.
        action_grid = qt.QGridLayout()
        buttons = [self.delete_all_button, self.import_landmarks_button,
                   self.export_landmarks_button, self.export_measurements_button]
        for index, button in enumerate(buttons):
            button.setMaximumHeight(24)
            action_grid.addWidget(button, index // 2, index % 2)
        action_row_widget = qt.QWidget()
        action_row_widget.setLayout(action_grid)

        layout = qt.QVBoxLayout()
        layout.addWidget(controls_widget)
        layout.addWidget(self.measurement_stack, 1)
        layout.addWidget(action_row_widget)

        self.setLayout(layout)

    def _export_landmarks(self):
        file_name = qt.QFileDialog.getSaveFileName(self, 'Export landmarks', '',"CSV File (*.csv)")
        if file_name == "": 
            return
        with open(file_name, 'w+', newline='') as csvfile:
            fieldnames = ['landmark name', 'x', 'y', 'z', 'radius']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';' 
                                    if locale.localeconv()['decimal_point'] == "," else ",", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for landmark in self.landmarks:
                if landmark.placed:
                    position = landmark.get_position()
                    radius = getattr(landmark, "radius", None)
                    writer.writerow({"landmark name": landmark.name, 
                                     "x": locale.str(position[0]), 
                                     "y": locale.str(position[1]), 
                                     "z": locale.str(position[2]),
                                     "radius": locale.str(radius) if radius is not None else ""})


    def _import_landmarks(self):
        file_name = qt.QFileDialog.getOpenFileName(self, 'Import landmarks', '',"CSV File (*.csv)")
        if file_name == "":
            return
        landmark_dict = {lm.name: lm for lm in self.landmarks}
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';'
                                    if locale.localeconv()['decimal_point'] == "," else ",", quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                try:
                    name = row['landmark name']
                    x = locale.atof(row['x'])
                    y = locale.atof(row['y'])
                    z = locale.atof(row['z'])
                except Exception as e:
                    errorDisplay("Invalid file. Must contain columns: 'landmark name', 'x', 'y', 'z'")
                    return
                if name not in landmark_dict.keys():
                    errorDisplay(f"Unknown landmark {name}")
                    return
                landmark = landmark_dict[name]
                landmark.define(x, y, z)
                # Older exports won't have a "radius" column at all, and it's only meaningful
                # for sphere landmarks - restore it only when both are present.
                radius_text = row.get('radius') if hasattr(row, 'get') else None
                if radius_text and hasattr(landmark, "set_radius"):
                    try:
                        landmark.set_radius(locale.atof(radius_text))
                    except ValueError:
                        pass
        # force update
        self.measurement_stack.currentWidget().disable() 
        self.measurement_stack.currentWidget().enable()

    def _export_measurements(self):
        file_name = qt.QFileDialog.getSaveFileName(self, 'Export measurements', '',"CSV File (*.csv)")
        if file_name == "":
            return

        with open(file_name, 'w+', newline='') as csvfile:
            fieldnames = ['measurement', 'value', 'description']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';' 
                                    if locale.localeconv()['decimal_point'] == "," else ",", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for i in range(self.measurement_stack.count):
                measurement = self.measurement_stack.widget(i).measurement
                result_ready, result_value, result_string = measurement()
                if result_ready:
                    writer.writerow({"measurement": measurement.name, 
                                     "value": locale.str(result_value), 
                                     "description": result_string})
        

    def _change_row(self, i):
        self.measurement_stack.currentWidget().disable()
        self.measurement_stack.setCurrentIndex(i)
        self.measurement_stack.currentWidget().enable()

    # Sphere visualization for fitted centers (femur head / condyles)
    SPHERE_COLORS = {
        "femur head": (0.9, 0.2, 0.2),
        "medial femur condyle": (0.2, 0.6, 0.9),
        "lateral femur condyle": (0.2, 0.8, 0.4),
    }

    def _sphere_node_name(self, label):
        return f"BoneAngleMeter {self.side} {label} sphere"

    def _on_show_spheres_toggled(self, checked):
        self.spheres_visible = bool(checked)
        if not self.spheres_visible:
            self.hide_all_spheres()
        else:
            # Re-show spheres for whichever measurement is currently active, if it's ready
            current_widget = self.measurement_stack.currentWidget()
            if current_widget is not None:
                current_widget.update_measurement()

    def show_sphere(self, label, center, radius):
        if not self.spheres_visible:
            self.hide_sphere(label)
            return

        node = self.sphere_nodes.get(label)
        if node is None:
            node = slicer.mrmlScene.GetFirstNodeByName(self._sphere_node_name(label))
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self._sphere_node_name(label))
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
            color = self.SPHERE_COLORS.get(label, (0.8, 0.8, 0.2))
            display_node.SetColor(*color)
            display_node.SetOpacity(0.35)
            display_node.SetVisibility2D(False)
        self.sphere_nodes[label] = node

        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(float(center[0]), float(center[1]), float(center[2]))
        sphere_source.SetRadius(max(float(radius), 0.01))
        sphere_source.SetThetaResolution(32)
        sphere_source.SetPhiResolution(32)
        sphere_source.Update()
        node.SetAndObservePolyData(sphere_source.GetOutputDataObject(0))
        node.GetDisplayNode().SetVisibility(True)

    def hide_sphere(self, label):
        node = self.sphere_nodes.get(label)
        if node is not None and node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(False)

    def hide_all_spheres(self):
        for label in list(self.sphere_nodes.keys()):
            self.hide_sphere(label)

    def _on_show_axes_toggled(self, checked):
        self.axes_visible = bool(checked)
        if not self.axes_visible:
            self.hide_all_axes()
        else:
            current_widget = self.measurement_stack.currentWidget()
            if current_widget is not None:
                current_widget.update_measurement()

    def _axis_node_name(self, label):
        return f"BoneAngleMeter {self.side} {label} axis"

    def show_axis(self, label, start, end, color):
        if not self.axes_visible:
            self.hide_axis(label)
            return

        node = self.axis_nodes.get(label)
        if node is None:
            node = slicer.mrmlScene.GetFirstNodeByName(self._axis_node_name(label))
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self._axis_node_name(label))
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
            display_node.SetVisibility2D(False)
        self.axis_nodes[label] = node

        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(float(start[0]), float(start[1]), float(start[2]))
        line_source.SetPoint2(float(end[0]), float(end[1]), float(end[2]))
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputConnection(line_source.GetOutputPort())
        tube_filter.SetRadius(1.0)
        tube_filter.SetNumberOfSides(12)
        tube_filter.Update()
        node.SetAndObservePolyData(tube_filter.GetOutputDataObject(0))
        node.GetDisplayNode().SetColor(*color)
        node.GetDisplayNode().SetVisibility(True)

    def hide_axis(self, label):
        node = self.axis_nodes.get(label)
        if node is not None and node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(False)

    def hide_all_axes(self):
        for label in list(self.axis_nodes.keys()):
            self.hide_axis(label)

    def _on_show_labels_toggled(self, checked):
        markups_node = slicer.mrmlScene.GetNodeByID(self.markup_node_id)
        if markups_node is None:
            return
        display_node = markups_node.GetDisplayNode()
        if display_node is None:
            return
        # Different Slicer versions expose this under different names; use whichever is available.
        if hasattr(display_node, "SetPointLabelsVisibility"):
            display_node.SetPointLabelsVisibility(bool(checked))
        elif hasattr(display_node, "SetTextVisibility"):
            display_node.SetTextVisibility(bool(checked))
        else:
            display_node.SetTextScale(1.0 if checked else 0.0)

    def _delete_all_landmarks(self):
        if qt.QMessageBox.question(self, "Delete all landmarks",
                                    f"Delete all placed {self.side} landmarks? This cannot be undone.",
                                    qt.QMessageBox.Yes | qt.QMessageBox.No) != qt.QMessageBox.Yes:
            return

        current_widget = self.measurement_stack.currentWidget()
        if current_widget is not None:
            current_widget.disable()

        # Belt-and-suspenders: hide every sphere/axis model node ever created for this side, not
        # just the ones belonging to the currently active measurement tab. Delete All should
        # leave nothing visible behind regardless of which tab happens to be open.
        self.hide_all_spheres()
        self.hide_all_axes()

        for landmark in self.landmarks:
            landmark._id = None
            landmark.placed = False

        markups_node = slicer.mrmlScene.GetNodeByID(self.markup_node_id)
        if markups_node is not None:
            markups_node.RemoveAllControlPoints()

        if current_widget is not None:
            current_widget.enable()

    def activate(self):
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeID(self.markup_node_id)

        # Auto-select first item
        if self.measurement_list.currentRow == -1 and self.measurement_list.count > 0:
            # setCurrentRow triggers currentRowChanged -> _change_row, which already calls
            # enable() on the newly-selected measurement - don't call it again below, or
            # start_interaction() runs twice without an intervening stop_interaction(),
            # registering duplicate observers.
            self.measurement_list.setCurrentRow(0)
        else:
            current_widget = self.measurement_stack.currentWidget()
            if current_widget is not None:
                current_widget.enable()

    def deactivate(self):
        self.measurement_stack.currentWidget().disable()
        self.hide_all_spheres()
        self.hide_all_axes()


class MeasurementWidget(qt.QWidget):
    '''
    Widget that displays the landmarks and results of a measurement
    '''
    def __init__(self, measurement, dialog):
        super().__init__()

        self.measurement = measurement
        self.dialog = dialog
        self.enabled = False

        self.landmark_stack = qt.QStackedWidget(self)
        self.landmark_list = qt.QListWidget()
        self.next_landmark_button = qt.QPushButton("Next")
        self.prev_landmark_button = qt.QPushButton("Previous")
        self.delete_landmark_button = qt.QPushButton("Delete")
        self.delete_landmark_button.setToolTip("Delete the currently selected landmark so it can be placed again.")
        self.next_landmark_button.setDefault(True)
        self.next_landmark_button.setAutoDefault(False)
        self.prev_landmark_button.setDefault(False)
        self.prev_landmark_button.setAutoDefault(False)
        self.delete_landmark_button.setDefault(False)
        self.delete_landmark_button.setAutoDefault(False)

        self.landmark_group_box = qt.QGroupBox("Landmarks")
        self.landmark_list.setMaximumHeight(110)

        nav_row = qt.QHBoxLayout()
        nav_row.addWidget(self.prev_landmark_button)
        nav_row.addWidget(self.next_landmark_button)
        nav_row_widget = qt.QWidget()
        nav_row_widget.setLayout(nav_row)

        landmark_layout = qt.QVBoxLayout()
        landmark_layout.addWidget(self.landmark_list)
        landmark_layout.addWidget(nav_row_widget)
        landmark_layout.addWidget(self.delete_landmark_button)
        landmark_layout.addWidget(self.landmark_stack)
        self.landmark_group_box.setLayout(landmark_layout)

        self.name_label = qt.QLabel(f"<big>{getattr(self.measurement, 'display_name', self.measurement.name)}</big>")
        self.name_label.setWordWrap(True)
        self.description_label = qt.QLabel(self.measurement.description)
        self.description_label.setWordWrap(True)
        self.measurement_label = qt.QLabel(f"Not available")
        self.measurement_label.setWordWrap(True)
        self.axes_label = qt.QLabel("")
        self.axes_label.setWordWrap(True)
        
        layout = qt.QFormLayout()
        layout.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
        layout.addRow(self.name_label)
        layout.addRow("<b>Description</b>", self.description_label)
        layout.addRow("<b>Result</b>", self.measurement_label)
        layout.addRow("<b>Axes</b>", self.axes_label)
        layout.addRow(self.landmark_group_box)

        self.setLayout(layout)

        self.next_shortcut = qt.QShortcut(qt.QKeySequence('Return'), self)
        self.next_shortcut.activated.connect(self._next_row)
        self.next_shortcut.setContext(qt.Qt.ApplicationShortcut)

        self.prev_shortcut = qt.QShortcut(qt.QKeySequence('Shift+Return'), self)
        self.prev_shortcut.activated.connect(self._prev_row)
        self.prev_shortcut.setContext(qt.Qt.ApplicationShortcut)

        for landmark in self.measurement.get_landmarks():
            widget_cls = SphereLandmarkWidget if hasattr(landmark, "radius") else LandmarkWidget
            self.landmark_stack.addWidget(widget_cls(landmark))
            self.landmark_list.addItem(landmark.name)
            landmark.add_change_callback(self.update_measurement)

        self.landmark_list.currentRowChanged.connect(self._change_row)
        # Clicking the (already) selected landmark resumes placing / editing it, e.g. after the
        # lasso or guide tools paused landmark placement.
        self.landmark_list.itemClicked.connect(lambda item: self._resume_current_landmark())
        self.next_landmark_button.clicked.connect(self._next_row)
        self.prev_landmark_button.clicked.connect(self._prev_row)
        self.delete_landmark_button.clicked.connect(self._delete_current_landmark)

        self.update_measurement()

    def enable(self):
        # Auto-select first item
        if self.landmark_list.currentRow == -1 and self.landmark_list.count > 0:
            # Emit signal currentRowChanged
            self.landmark_list.setCurrentRow(0)

        for landmark in self.measurement.landmarks:
            landmark.show()
        
        current_landmark_widget = self.landmark_stack.currentWidget()
        current_landmark_widget.landmark.start_interaction()
                   
        self.next_shortcut.setEnabled(True)
        self.prev_shortcut.setEnabled(True)
        self.enabled = True

        self.update_measurement()
        
    def _resume_current_landmark(self):
        if not self.enabled:
            return
        current = self.landmark_stack.currentWidget()
        if current is not None:
            current.landmark.start_interaction()

    def disable(self):
        for landmark in self.measurement.landmarks:
            landmark.stop_interaction()
            landmark.hide()
        self.next_shortcut.setEnabled(False)
        self.prev_shortcut.setEnabled(False)
        self.enabled = False
        for label in self.measurement.SPHERE_LABELS:
            self.dialog.hide_sphere(label)
        for label in self.measurement.AXIS_LABELS:
            self.dialog.hide_axis(label)

    def update_measurement(self):
        if self.enabled:
            try:
                result_ready, result_value, result_string = self.measurement()
                if result_ready:
                    lines = self.measurement.result_lines(result_value, result_string) \
                        if hasattr(self.measurement, "result_lines") else None
                    self.measurement_label.setText("<br>".join(lines) if lines else
                                                   f"{result_value:.2f}\N{DEGREE SIGN} {result_string}")
                    self.measurement_label.setStyleSheet("QLabel { background-color : green}")
                    for label, (center, radius) in self.measurement.get_spheres().items():
                        self.dialog.show_sphere(label, center, radius)
                    axis_items = list(self.measurement.get_axes().items())
                    legend_parts = []
                    for i, (label, (start, end)) in enumerate(axis_items):
                        color = self.dialog.AXIS_PALETTE[i % len(self.dialog.AXIS_PALETTE)]
                        self.dialog.show_axis(label, start, end, color)
                        r, g, b = [int(round(c * 255)) for c in color]
                        legend_parts.append(f'<span style="color:rgb({r},{g},{b})">&#9632;</span> {label}')
                    self.axes_label.setText("<br>".join(legend_parts))
                else:
                    self.measurement_label.setText(result_string)
                    self.measurement_label.setStyleSheet("QLabel { background-color : orange}")
                    self.axes_label.setText("")
                    for label in self.measurement.SPHERE_LABELS:
                        self.dialog.hide_sphere(label)
                    for label in self.measurement.AXIS_LABELS:
                        self.dialog.hide_axis(label)
            except Exception as e:
                self.measurement_label.setText(f"Error executing measurement: {e}")
                self.measurement_label.setStyleSheet("QLabel { background-color : red}")
                self.axes_label.setText("")
                for label in self.measurement.SPHERE_LABELS:
                    self.dialog.hide_sphere(label)
                for label in self.measurement.AXIS_LABELS:
                    self.dialog.hide_axis(label)

    # Internal callbacks
    def _next_row(self):
        if self.landmark_list.currentRow + 1 < self.landmark_list.count:
            # Emit signal currentRowChanged
            self.landmark_list.setCurrentRow(self.landmark_list.currentRow + 1)

    def _prev_row(self):
        if self.landmark_list.currentRow > 0:
            # Emit signal currentRowChanged
            self.landmark_list.setCurrentRow(self.landmark_list.currentRow - 1)

    def _change_row(self, i):
        old_landmark_widget = self.landmark_stack.currentWidget()
        old_landmark_widget.landmark.stop_interaction()
        self.landmark_stack.setCurrentIndex(i)
        new_landmark_widget = self.landmark_stack.currentWidget()
        new_landmark_widget.landmark.start_interaction()

    def _delete_current_landmark(self):
        current_landmark_widget = self.landmark_stack.currentWidget()
        if current_landmark_widget is None:
            return
        landmark = current_landmark_widget.landmark
        landmark.stop_interaction()
        landmark.delete()
        landmark.start_interaction()

class LandmarkWidget(qt.QWidget):
    '''
    Info-box for landmarks
    '''
    def __init__(self, landmark):
        super(LandmarkWidget, self).__init__()

        self.landmark = landmark

        self.landmark_label = qt.QLabel(f"<big>{self.landmark.name}</big>")
        self.landmark_label.setWordWrap(True)
        self.description_label = qt.QLabel(self.landmark.description)
        self.description_label.setWordWrap(True)
        self.position_label = qt.QLabel("Position: not set")
        if self.landmark.image_path != "":
            path = osp.join(MODULE_PATH, self.landmark.image_path)
            pixmap = qt.QPixmap(path)
            if pixmap.isNull():
                raise ValueError(f"Could not load image from file '{path}'")
            self.image = ResizableImage(pixmap)
        else:
            self.image = None

        layout = qt.QVBoxLayout()
        layout.addWidget(self.landmark_label)
        layout.addWidget(self.description_label)
        layout.addWidget(self.position_label)
        if self.image is not None:
            layout.addWidget(self.image)
            layout.setStretch(layout.count()-1, 1)
        self.setLayout(layout)

        self.landmark.add_change_callback(self._update_position_label)
        self._update_position_label()

    def _update_position_label(self):
        if self.landmark.placed:
            p = self.landmark.get_position()
            self.position_label.setText(f"Position: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f} mm")
        else:
            self.position_label.setText("Position: not set")


class SphereLandmarkWidget(LandmarkWidget):
    '''
    Info-box for a SphereLandmark: adds a radius control on top of the usual position/description
    panel, so the user can place the center point and then manually grow/shrink the sphere (drawn
    live in the 3D view) until it visually matches the bone surface.
    '''
    def __init__(self, landmark):
        super().__init__(landmark)

        self.radius_spinbox = qt.QDoubleSpinBox()
        self.radius_spinbox.setRange(1.0, 100.0)
        self.radius_spinbox.setSingleStep(0.5)
        self.radius_spinbox.setSuffix(" mm")
        self.radius_spinbox.setValue(landmark.radius)
        self.radius_spinbox.setToolTip(
            "Adjust the sphere's radius in the 3D view until it best matches the condyle surface.")
        self.radius_spinbox.valueChanged.connect(self._on_radius_changed)

        # Put the radius spinbox on the same row as the position readout, instead of its own
        # separate row, for a more compact layout.
        insert_index = self.layout().indexOf(self.position_label)
        self.layout().removeWidget(self.position_label)

        position_radius_row = qt.QHBoxLayout()
        position_radius_row.addWidget(self.position_label, 1)
        position_radius_row.addWidget(qt.QLabel("Radius"))
        position_radius_row.addWidget(self.radius_spinbox)
        position_radius_row_widget = qt.QWidget()
        position_radius_row_widget.setLayout(position_radius_row)

        self.layout().insertWidget(insert_index, position_radius_row_widget)

    def _on_radius_changed(self, value):
        self.landmark.set_radius(value)


class ResizableImage(qt.QLabel):
    """
    Widget that displays an image scaled once to a fixed target width, keeping aspect ratio.
    Deliberately not dynamically responsive to container resizing - trying to negotiate width via
    size policies and height-for-width proved unreliable across Slicer's Qt/stylesheet
    environment (repeatedly caused the sidebar to be forced wider than the screen). A fixed
    target width that comfortably fits Slicer's typical sidebar is far more robust. Click the
    image to view it larger in a popup.
    """
    TARGET_WIDTH = 260  # px - adjust here if it's still too wide/narrow for your sidebar
    POPUP_MAX_WIDTH = 900
    POPUP_MAX_HEIGHT = 700

    def __init__(self, pixmap) -> None:
        super().__init__()
        self.raw_pixmap = pixmap
        self.setPixmap(self.raw_pixmap.scaled(
            qt.QSize(self.TARGET_WIDTH, 100000), qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation))
        self.setAlignment(qt.Qt.AlignCenter)
        self.setCursor(qt.Qt.PointingHandCursor)
        self.setToolTip("Click to view full size")
        self._popup = None

    def mousePressEvent(self, event):
        self._show_popup()
        event.accept()

    def _show_popup(self):
        # Reuse the existing popup (just bring it to front) instead of stacking up duplicates
        # if the image is clicked more than once.
        if self._popup is not None:
            self._popup.raise_()
            self._popup.activateWindow()
            return

        popup = qt.QDialog(self)
        popup.setWindowTitle("Landmark reference image")

        pixmap = self.raw_pixmap
        if pixmap.width() > self.POPUP_MAX_WIDTH or pixmap.height() > self.POPUP_MAX_HEIGHT:
            pixmap = pixmap.scaled(qt.QSize(self.POPUP_MAX_WIDTH, self.POPUP_MAX_HEIGHT),
                                    qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)

        image_label = qt.QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(qt.Qt.AlignCenter)

        layout = qt.QVBoxLayout()
        layout.addWidget(image_label)
        popup.setLayout(layout)

        # Non-modal: lets the user keep it open while placing landmarks in the 3D view.
        popup.finished.connect(self._on_popup_closed)
        self._popup = popup
        popup.show()

    def _on_popup_closed(self):
        self._popup = None