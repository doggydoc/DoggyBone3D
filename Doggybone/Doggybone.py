# Doggybone 3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

import json
import logging
import re
import os
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
from DoggyboneLib import measurement_logic
importlib.reload(measurement_logic)
from DoggyboneLib import landmark_logic
importlib.reload(landmark_logic)
from DoggyboneLib import osteotomy_logic
from DoggyboneLib import guide_logic
importlib.reload(osteotomy_logic)
importlib.reload(guide_logic)
from DoggyboneLib import measurements
importlib.reload(measurements)
from DoggyboneLib import landmarks
importlib.reload(landmarks)

from DoggyboneLib.measurements import MEASUREMENTS
from DoggyboneLib.landmarks import LANDMARKS

MODULE_PATH = osp.dirname(__file__)
#
# Doggybone
#

# Status colours used everywhere in the module (light pastel, dark text - readable in light and dark themes)
STATUS_OK_STYLE = "QLabel { background-color: #C6EFCE; color: #1B3D1F; padding: 2px 4px; }"
# measured values: a light tint of their section's header colour
MEASUREMENT_VALUE_STYLE = "QLabel { background-color: #E6DEF8; color: #322659; padding: 2px 4px; }"   # Measurements (#6b46c1)
OSTEOTOMY_VALUE_STYLE = "QLabel { background-color: #F8DDCC; color: #5A2A0C; padding: 2px 4px; }"     # Osteotomy (#c05621)
STATUS_WARN_STYLE = "QLabel { background-color: #FFE5B4; color: #4A3000; padding: 2px 4px; }"
STATUS_ERROR_STYLE = "QLabel { background-color: #F8CBCB; color: #5A1010; padding: 2px 4px; font-weight: bold; }"


class Doggybone(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Doggybone 3D"
    self.parent.categories = ["Orthopedics"]
    self.parent.dependencies = []
    self.parent.contributors = ["Matthias Frank (Kleintierklinik Frank & Partner, Freiburg)",
                                "Juliette Burg Personnaz, Andreas Voigt (CTK, LMU Munich; BoneAngleMeter)"]
    self.parent.helpText = """
Doggybone 3D: landmark-based angle measurement of the canine hind limb, corrective osteotomy planning
(closing / opening wedge, bisecting principle, contact-congruent alignment, torsion) and
patient-specific cutting-guide design.
<p><b>For research use only - not a certified medical device.</b> All planning results must be
checked by the surgeon.</p>
"""
    self.parent.acknowledgementText = """
Doggybone 3D is developed by Matthias Frank, Kleintierklinik Frank &amp; Partner, Freiburg, Germany.
It is based on BoneAngleMeter by Juliette Burg Personnaz and Andreas Voigt, developed during
Juliette Burg Personnaz' doctorate at CTK, LMU Munich
(<a href="https://github.com/jburgp/BoneAngleMeter_public">github.com/jburgp/BoneAngleMeter_public</a>).
MIT License.
"""
    iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "Doggybone.png")
    if os.path.exists(iconPath):
      self.parent.icon = qt.QIcon(iconPath)


class DoggyboneWidget(ScriptedLoadableModuleWidget):
    """
    Base widget that manages the measurement groups
    """

    SEGMENTATION_NODE_NAME = "Doggybone Bone Segmentation"

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
    WEDGE_TARGET_DEFAULT = {"Femur": 94.0, "Tibia": 0.0}   # aLDFA target (canine reference ~94°)
    ROTATION_TARGET_DEFAULT = {"Femur": 25.0, "Tibia": 0.0}   # femoral antetorsion target
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
        # patient (kept in the scene); the number is printed on the guides and used in file names
        self.patient_name = qt.QLineEdit()
        self.patient_name.setPlaceholderText("name")
        self.patient_number = qt.QLineEdit()
        self.patient_number.setPlaceholderText("no.")
        self.patient_number.setToolTip("Patient number - printed on the guides together with bone, side "
                                       "and an arrow towards proximal, and used in the STL file names.")
        patient_row = qt.QHBoxLayout()
        patient_row.setContentsMargins(0, 0, 0, 0)
        patient_row.addWidget(self.patient_name, 3)
        patient_row.addWidget(qt.QLabel("No."))
        patient_row.addWidget(self.patient_number, 2)
        patient_widget = qt.QWidget()
        patient_widget.setLayout(patient_row)
        segmentation_form_layout.addRow("Patient", patient_widget)
        self._load_patient()
        self.patient_name.textChanged.connect(lambda _t: self._store_patient())
        self.patient_number.textChanged.connect(lambda _t: self._store_patient())
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
        self.fill_cavities_button = qt.QPushButton("Fill cavities")
        self.fill_cavities_button.setToolTip(
            "Make the bone SOLID: medullary canal, pores and cancellous gaps are filled (slice by slice "
            "across the bone and in 3D, small cortical gaps closed) - only the outer bone surface remains, "
            "so painting the guide footprint only reaches the outside. Neighbouring bones (patella, "
            "fabellae) stay separate. 'Delete' undoes it.")
        self.fill_cavities_button.clicked.connect(self.onFillCavities)
        islands_row.addWidget(self.fill_cavities_button, 1)
        self.remove_bone_button = qt.QPushButton("Remove bone (click)")
        self.remove_bone_button.setCheckable(True)
        self.remove_bone_button.setToolTip(
            "Click a bone in the 3D view (patella, fabella, tibia ...) to delete it from the segmentation - "
            "the whole separate piece; thin partial-volume bridges to the femur are broken. Click the button "
            "again to stop. 'Delete' undoes all clean-up edits.")
        self.remove_bone_button.toggled.connect(self.onRemoveBoneToggled)
        segmentation_form_layout.addRow("Separate bones", self.remove_bone_button)
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
                logging.exception("Doggybone: non-critical step failed (continuing)")
        editor.setEffectNameOrder(["Scissors", "Islands"])
        editor_node = slicer.mrmlScene.GetFirstNodeByName("Doggybone segment editor")
        if editor_node is None:
            editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "Doggybone segment editor")
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
            self._get_or_create_fiducial_node("DoggyboneFiducialsLeft"), self)
        self.right_panel = MeasurementsPanel('right',
            self._get_or_create_fiducial_node("DoggyboneFiducialsRight"), self)
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

        self.reference_plane_selector = qt.QComboBox()
        self.reference_plane_selector.addItems(["Proximal", "Distal"])
        self.reference_plane_selector.currentIndexChanged.connect(self._on_master_plane_changed)
        osteotomy_form_layout.addRow("Master cutting plane", self.reference_plane_selector)
        self.active_plane_selector = self.reference_plane_selector

        # (kept for the status text it receives, not shown in the panel)
        self.actual_plane_angle_label = qt.QLabel("Master/SECOND angle: --")

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
        self._on_osteotomy_bone_changed(self.osteotomy_bone.currentIndex)


        # CWO is always contact-congruent (hinge closure + torsion about the contact normal); the
        # former measurement-based alternative was removed.

        self.osteotomy_insert_button = qt.QPushButton("Insert/update MASTER plane")
        self.osteotomy_insert_button.clicked.connect(self.onInsertMasterPlane)
        osteotomy_form_layout.addRow(self.osteotomy_insert_button)

        # The MASTER plane is aligned by the bisecting principle automatically when it is created
        # (onInsertMasterPlane); afterwards it can be moved/rotated by hand.

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

        # Clinical fine adjustment AFTER the calculated angular correction.  These are
        # translations in the LOCAL X/Y axes of the MASTER cutting plane only.  They do not
        # change varus/valgus, torsion, or the plane-normal (contact) position.
        self.fragment_shift_x = qt.QDoubleSpinBox()
        self.fragment_shift_x.setRange(-20.0, 20.0)
        self.fragment_shift_x.setDecimals(1)
        self.fragment_shift_x.setSingleStep(0.5)
        self.fragment_shift_x.setSuffix(" mm")
        self.fragment_shift_x.setToolTip("Fine translation of the corrected fragment along local X of the MASTER cutting plane.")
        # (not shown in the panel - holds the X/Y gizmo's shift)

        self.fragment_shift_y = qt.QDoubleSpinBox()
        self.fragment_shift_y.setRange(-20.0, 20.0)
        self.fragment_shift_y.setDecimals(1)
        self.fragment_shift_y.setSingleStep(0.5)
        self.fragment_shift_y.setSuffix(" mm")
        self.fragment_shift_y.setToolTip("Fine translation of the corrected fragment along local Y of the MASTER cutting plane.")
        # (not shown in the panel - holds the X/Y gizmo's shift)

        self.reset_fragment_shift_button = qt.QPushButton("Reset X/Y shift")
        self.reset_fragment_shift_button.clicked.connect(self.onResetFragmentShift)
        # (reset button not shown in the panel)
        self.fragment_shift_x.valueChanged.connect(self._on_fragment_shift_changed)
        self.fragment_shift_y.valueChanged.connect(self._on_fragment_shift_changed)

        self.clear_preview_button = qt.QPushButton("Remove preview")
        self.clear_preview_button.clicked.connect(self.onClearPreview)

        self.reset_planning_button = qt.QPushButton("Reset planning (re-read landmarks)")
        self.reset_planning_button.setToolTip(
            "Back to zero for the selected side and bone: removes the preview, the X/Y gizmo and "
            "both cutting planes, sets fragment shift and direct correction values to 0, and "
            "re-reads the measured values and calculated corrections from the CURRENT landmarks "
            "(so corrected landmarks are taken over). Target values are kept.")
        self.reset_planning_button.clicked.connect(self.onResetOsteotomyPlanning)

        self.save_screenshot_button = qt.QPushButton("Save screenshot of 3D view...")
        self.save_screenshot_button.setToolTip(
            "Saves the current 3D view as a PNG image - e.g. to document the preview of the "
            "final result. Zoom/rotate the 3D view as desired first.")
        self.save_screenshot_button.clicked.connect(self.onSaveScreenshot)
        # the four preview buttons in two rows (saves space)
        self.preview_button.setText("Show/update preview")
        self.reset_planning_button.setText("Reset planning")
        self.save_screenshot_button.setText("Save 3D screenshot...")
        preview_grid = qt.QGridLayout()
        preview_grid.setContentsMargins(0, 0, 0, 0)
        preview_grid.setHorizontalSpacing(4)
        preview_grid.setVerticalSpacing(2)
        preview_grid.addWidget(self.preview_button, 0, 0)
        preview_grid.addWidget(self.clear_preview_button, 0, 1)
        preview_grid.addWidget(self.reset_planning_button, 1, 0)
        preview_grid.addWidget(self.save_screenshot_button, 1, 1)
        preview_widget = qt.QWidget()
        preview_widget.setLayout(preview_grid)
        osteotomy_form_layout.addRow(preview_widget)

        self.correction_status = qt.QLabel()
        self.correction_status.setVisible(False)

        self.hide_original_in_preview = qt.QCheckBox("Hide original segmentation in preview")
        self.hide_original_in_preview.setChecked(True)
        osteotomy_form_layout.addRow(self.hide_original_in_preview)
        self.show_corrected_landmarks = qt.QCheckBox("Show landmarks, spheres and axes in the corrected position")
        self.show_corrected_landmarks.setChecked(True)
        self.show_corrected_landmarks.setToolTip(
            "With an active preview: the landmarks of the moved fragment are carried along with the "
            "correction (incl. fine shift); spheres and axes are drawn from them and the angles are "
            "measured on the corrected bone.")
        self.show_corrected_landmarks.toggled.connect(lambda _c: self._update_corrected_landmarks())
        osteotomy_form_layout.addRow(self.show_corrected_landmarks)

        self.segment_opacity_slider = qt.QSlider(qt.Qt.Horizontal)
        self.segment_opacity_slider.setMinimum(0)
        self.segment_opacity_slider.setMaximum(100)
        self.segment_opacity_slider.setValue(90)
        self.segment_opacity_slider.setToolTip(
            "3D opacity of the (reduced) bone segments of the preview. Lower it to see the "
            "corrected landmarks, spheres and axes inside the bone.")
        self.segment_opacity_slider.valueChanged.connect(lambda _v: self._apply_segment_opacity())
        osteotomy_form_layout.addRow("Segment opacity", self.segment_opacity_slider)

        # Guide planning (stage 2, upcoming): placeholder section so the entry point is already
        # visible once the guide geometry (cutting slot + drill sleeves, STL export) is built on
        # top of the plane(s)/correction above.
        guide_collapsible_button = ctk.ctkCollapsibleButton()
        guide_collapsible_button.text = "Guide Planning"
        guide_collapsible_button.setStyleSheet(
            "ctkCollapsibleButton { background-color: #718096; color: white; font-weight: bold; }")
        guide_collapsible_button.collapsed = True
        guide_collapsible_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        self._guide_collapsible_button = guide_collapsible_button
        self.layout.addWidget(guide_collapsible_button)
        guide_layout = qt.QVBoxLayout(guide_collapsible_button)
        guide_info_label = qt.QLabel(
            "Footprint = coloured area on the bone surface where the guide sits. Paint it by hand "
            "with Brush + / Lasso + anywhere on the outer bone surface around the osteotomy (rotate "
            "the view as needed), then extrude it into the guide body.")
        guide_info_label.setWordWrap(True)
        guide_layout.addWidget(guide_info_label)

        guide_form = qt.QFormLayout()
        guide_layout.addLayout(guide_form)
        self.guide_mode_saw = qt.QRadioButton("2. Saw guide")
        self.guide_mode_reduction = qt.QRadioButton("1. Reduction guide")
        self.guide_mode_saw.setToolTip("Cutting guide on the bone as it is (saw slots, pin sleeves).")
        self.guide_mode_reduction.setToolTip(
            "Guide painted on the fragments in the CORRECTED position; sleeves over the pins of the saw "
            "guide hold the reduction. Keep it small to leave room for the plate.")
        # workflow order: pins after the reduction -> reduction guide first, then the saw guide
        self.guide_mode_reduction.setChecked(True)
        self._guide_mode = "reduction"
        self._guide_mode_group = qt.QButtonGroup()
        self._guide_mode_group.addButton(self.guide_mode_saw)
        self._guide_mode_group.addButton(self.guide_mode_reduction)
        self.guide_mode_reduction.toggled.connect(lambda _c: self.onGuideModeChanged())
        mode_row = qt.QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(self.guide_mode_reduction)
        mode_row.addWidget(self.guide_mode_saw)
        mode_widget = qt.QWidget()
        mode_widget.setLayout(mode_row)
        guide_form.addRow("Guide", mode_widget)
        self.reduction_type = qt.QComboBox()
        self.reduction_type.addItems(["Pin bar (seated on the bone)", "Bone-contact footprint"])
        self.reduction_type.setToolTip(
            "Bar: a rod with a boss around every pin holds the pins in the corrected position - no "
            "footprint, the bone stays free for the plate. Footprint: painted guide on the corrected bone.")
        self.reduction_standoff = qt.QDoubleSpinBox()
        self.reduction_standoff.setRange(3.0, 60.0)
        self.reduction_standoff.setValue(15.0)
        self.reduction_standoff.setSuffix(" mm")
        self.reduction_standoff.setToolTip("Height of the bar's underside region above the highest pin entry.")
        self.reduction_bar_diameter = qt.QDoubleSpinBox()
        self.reduction_bar_diameter.setRange(3.0, 15.0)
        self.reduction_bar_diameter.setValue(6.0)
        self.reduction_bar_diameter.setSuffix(" mm")
        self.reduction_bar_diameter.setToolTip("Diameter of the connecting rod.")
        self.reduction_boss_height = qt.QDoubleSpinBox()
        self.reduction_boss_height.setRange(4.0, 25.0)
        self.reduction_boss_height.setValue(10.0)
        self.reduction_boss_height.setSuffix(" mm")
        self.reduction_boss_height.setToolTip("Length of the guiding boss around each pin (longer = less play).")
        reduction_grid = qt.QGridLayout()
        reduction_grid.setContentsMargins(0, 0, 0, 0)
        reduction_grid.setHorizontalSpacing(6)
        # the reduction guide is always the pin bar (no painted bone-contact footprint): Type not shown
        self.reduction_type.setCurrentIndex(0)
        self.reduction_type.setVisible(False)
        self.reduction_standoff.setToolTip("Top of the guide above the highest pin entry.")
        self.reduction_bar_diameter.setToolTip("Width of the webs joining the pin bosses.")
        for col, (label, field) in enumerate((("Height", self.reduction_standoff),
                                              ("Width", self.reduction_bar_diameter))):
            reduction_grid.addWidget(qt.QLabel(label), 1, 2 * col)
            field.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            field.setMinimumWidth(55)
            reduction_grid.addWidget(field, 1, 2 * col + 1)
        def row_spin(lo, hi, step, suffix, tip, decimals=1, special=None):
            box = qt.QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.setSuffix(suffix)
            box.setToolTip(tip)
            if special:
                box.setSpecialValueText(special)
            box.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            box.setMinimumWidth(55)
            return box
        self.pin_row_count = qt.QSpinBox()
        self.pin_row_count.setRange(1, 4)
        self.pin_row_count.setValue(2)
        self.pin_row_count.setToolTip("Number of pins per fragment.")
        self.pin_row_spacing = row_spin(0.0, 30.0, 0.5, " mm", "Distance between the pins of a row (auto: from "
                                        "the pin diameter - sleeve + 2 mm, at least 3 x pin).", special="auto")
        self.pin_row_distance_proximal = row_spin(0.0, 60.0, 0.5, " mm", "Proximal fragment: distance of the "
                                                  "first pin from its cut (auto: slot + wall + pin + 1 mm).", special="auto")
        self.pin_row_distance_distal = row_spin(0.0, 60.0, 0.5, " mm", "Distal fragment: distance of the first "
                                                "pin from its cut (auto: slot + wall + pin + 1 mm).", special="auto")
        self.place_pin_rows_button = qt.QPushButton("Place pins")
        self.place_pin_rows_button.setToolTip(
            "One row of centred, parallel pins per fragment (parallel to the cut plane(s)). Drag the "
            "row handles along the bone to adjust. With 'Pins': click the bone to add a pin, click a pin "
            "to select it and move it with the small gizmo, click the selected pin again to remove it. "
            "Added or moved pins are kept when the rows are recalculated.")
        self.place_pin_rows_button.clicked.connect(self.onPlacePinRows)
        reduction_grid.addWidget(qt.QLabel("Pins"), 2, 0)
        reduction_grid.addWidget(self.pin_row_count, 2, 1)
        reduction_grid.addWidget(qt.QLabel("Spacing"), 2, 2)
        reduction_grid.addWidget(self.pin_row_spacing, 2, 3)
        reduction_grid.addWidget(qt.QLabel("Prox."), 3, 0)
        reduction_grid.addWidget(self.pin_row_distance_proximal, 3, 1)
        reduction_grid.addWidget(qt.QLabel("Dist."), 3, 2)
        reduction_grid.addWidget(self.pin_row_distance_distal, 3, 3)
        self._reduction_diam_label = qt.QLabel("Pin size")        # pin diameter shown here in reduction mode
        reduction_grid.addWidget(self._reduction_diam_label, 4, 0)
        self.pin_centre_checkbox = qt.QCheckBox("Centre in bone")
        self.pin_centre_checkbox.setChecked(False)
        self.pin_centre_checkbox.setToolTip(
            "Off (default): a pin set by hand or moved with the gizmo goes exactly where you put it. "
            "On: it is snapped to the centre of the bone cross-section (the gizmo then only chooses its "
            "level along the bone). The automatic pin rows are always centred. All pins always stay "
            "parallel to each other and to the cut plane(s).")
        self.pin_centre_checkbox.setVisible(False)   # not offered: manual pins go exactly where they are put
        reduction_grid.addWidget(self.place_pin_rows_button, 5, 0, 1, 3)
        self.reduction_delete_pins_button = qt.QPushButton("Delete pins")
        self.reduction_delete_pins_button.setToolTip("Delete all pins (and the reduction guide built on them).")
        self.reduction_delete_pins_button.clicked.connect(self.onDeletePins)
        reduction_grid.addWidget(self.reduction_delete_pins_button, 5, 3, 1, 3)
        self.reduction_ring_checkbox = qt.QCheckBox("Ring (closed frame around all pins)")
        self.reduction_ring_checkbox.setChecked(True)
        self.reduction_ring_checkbox.setToolTip(
            "The webs run as a closed ring along the outline of all pins (plus links to inner pins) instead "
            "of a single chain - two bridges across the osteotomy, much stiffer against rotation and "
            "bending. Needs pins that do not all lie in one line.")
        reduction_grid.addWidget(self.reduction_ring_checkbox, 6, 0, 1, 6)
        self.pin_row_pattern = qt.QComboBox()
        self.pin_row_pattern.addItems(["Row", "Staggered", "Manual"])
        self.pin_row_pattern.setCurrentIndex(1)
        self.pin_row_pattern.setToolTip(
            "Pin distribution: 'Row' - automatic, all pins on the bone centre line; 'Staggered' - automatic, "
            "pins alternately offset to both sides of the centre (as far as the bone allows, max. 4 mm) - more "
            "stable and gives a wider ring; 'Manual' - click every pin on the corrected bone yourself.")
        self.pin_row_pattern.currentIndexChanged.connect(lambda _i: self._pin_distribution_changed())
        self.reduction_filled_checkbox = qt.QCheckBox("Filled frame")
        self.reduction_filled_checkbox.setChecked(False)
        self.reduction_filled_checkbox.setToolTip(
            "The whole area inside the ring becomes solid - stiffer, but it covers more bone (less room for "
            "the plate).")
        reduction_grid.addWidget(qt.QLabel("Pin distrib."), 4, 2)
        reduction_grid.addWidget(self.pin_row_pattern, 4, 3, 1, 3)
        self.reduction_filled_checkbox.setVisible(False)    # not offered (open ring only)
        self.reduction_contact_only_checkbox = qt.QCheckBox("Bone contact only at the pins")
        self.reduction_contact_only_checkbox.setChecked(True)
        self.reduction_contact_only_checkbox.setToolTip(
            "The guide touches the bone only with the small cylinders around the pins; the webs / ring / "
            "frame follow the bone surface 'Clearance' mm above it - stiff and low.")
        self.reduction_clearance = qt.QDoubleSpinBox()
        self.reduction_clearance.setRange(1.0, 30.0)
        self.reduction_clearance.setSingleStep(0.5)
        self.reduction_clearance.setValue(3.0)
        self.reduction_clearance.setSuffix(" mm")
        self.reduction_clearance.setToolTip("Free space between the bone and the underside of the webs / frame.")
        self.reduction_clearance.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        self.reduction_clearance.setMinimumWidth(55)
        self.reduction_contact_only_checkbox.toggled.connect(self.reduction_clearance.setEnabled)
        reduction_grid.addWidget(self.reduction_contact_only_checkbox, 8, 0, 1, 4)
        reduction_grid.addWidget(self.reduction_clearance, 8, 4, 1, 2)
        for box in (self.pin_row_count, self.pin_row_spacing, self.pin_row_distance_proximal,
                    self.pin_row_distance_distal):
            box.valueChanged.connect(lambda _v: self._pin_rows_changed())
        self.reduction_type.currentIndexChanged.connect(lambda _i: self._update_guide_tool_visibility())
        self.reduction_options_widget = qt.QWidget()
        self.reduction_options_widget.setLayout(reduction_grid)
        self.reduction_options_widget.setVisible(True)           # shown in reduction mode only (default)
        guide_form.addRow(self.reduction_options_widget)
        self.guide_side_selector = qt.QComboBox()
        self.guide_side_selector.addItems(["Cranial", "Caudal", "Medial", "Lateral"])
        self.guide_side_selector.setToolTip(
            "Side of the bone the guide sits on - the same choice for every bone. Directions come "
            "from the anatomical frame (shaft axis, medial/lateral landmarks, limb side).")
        # Internal only: orients the prepared surface; the footprint is painted all around by hand.
        self.guide_side_selector.setVisible(False)

        def spin(minimum, maximum, value, step, suffix, tip, decimals=1):
            box = qt.QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setValue(value)
            box.setSuffix(suffix)
            box.setToolTip(tip)
            return box

        # Internal only: size of the prepared bone-surface region around the cuts (no suggestion).
        self.guide_footprint_width = spin(4.0, 200.0, 80.0, 1.0, " mm", "")
        self.guide_footprint_margin = spin(2.0, 60.0, 12.0, 1.0, " mm",
                                           "How far the suggested area reaches beyond the cuts (room for the pins).")
        self.guide_footprint_margin.setValue(25.0)

        # Surface editing tools (3D view): brush / lasso, add / remove, undo.
        self._guide_tool_buttons = {}
        tools_row = qt.QGridLayout()
        tools_row.setContentsMargins(0, 0, 0, 0)
        tools_row.setHorizontalSpacing(4)
        tools_row.setVerticalSpacing(2)
        tool_positions = {"brush_add": (0, 0), "brush_remove": (0, 1), "lasso_add": (1, 0),
                          "lasso_remove": (1, 1), "brush_smooth": (2, 0), "pins": (3, 0)}
        for key, label, tip in (("brush_add", "Brush +", "Colour the surface under the mouse (drag)."),
                                ("brush_remove", "Brush -", "Uncolour the surface under the mouse (drag) - removes footprint and Smooth marks."),
                                ("brush_smooth", "Smooth", "Retouch the contact surface of the finished guide: only the "
                                 "guide is shown - paint on its inner side. A stroke starting on yellow erases."),
                                ("pins", "Edit pins", "Pins on the corrected bone: click on the bone to add a pin (parallel "
                                 "to the others), click a pin to delete it, Cmd/Ctrl + click a pin to move it with the "
                                 "gizmo. The saw guide takes the pins over, pre-angled."),
                                ("lasso_add", "Lasso +", "Draw around the area to colour (visible surface only)."),
                                ("lasso_remove", "Lasso -", "Draw around the area to uncolour (visible surface only).")):
            button = qt.QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.toggled.connect(lambda checked, k=key: self.onGuideToolToggled(k, checked))
            button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            tools_row.addWidget(button, *tool_positions[key])
            self._guide_tool_buttons[key] = button
        self.guide_delete_button = qt.QPushButton("Delete")
        self.guide_delete_button.setToolTip(
            "Delete the whole footprint and the guide body - then paint again "
            "or by painting with Brush + / Lasso +.")
        self.guide_delete_button.clicked.connect(self.onGuideDelete)
        self.guide_delete_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        tools_row.addWidget(self.guide_delete_button, 2, 1)
        self.guide_pins_delete_button = qt.QPushButton("Pins delete")
        self.guide_pins_delete_button.setToolTip("Remove all pin marks, sleeves and bores (the guide is rebuilt without pins).")
        self.guide_pins_delete_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        self.guide_pins_delete_button.clicked.connect(self.onDeletePins)
        tools_row.addWidget(self.guide_pins_delete_button, 3, 1)
        tools_widget = qt.QWidget()
        tools_widget.setLayout(tools_row)
        guide_form.addRow(tools_widget)
        self._update_guide_tool_visibility()
        guide_fields = []
        self.guide_brush_radius = spin(0.5, 15.0, 3.0, 0.5, " mm", "Brush radius.")
        guide_fields.append(("Brush radius", self.guide_brush_radius))
        self.guide_side_only_checkbox = qt.QCheckBox("Guide side only")
        self.guide_side_only_checkbox.checked = False   # always free manual painting
        self.guide_side_only_checkbox.setToolTip(
            "On: brush / lasso only reach the outer bone surface as seen from the guide side. "
            "Off (retouch): they reach any visible bone surface facing you - rotate the view onto "
            "a hollow or a flank and paint it in.")
        self.guide_side_only_checkbox.setVisible(False)

        self.guide_gap = spin(0.0, 2.0, 0.2, 0.05, " mm",
                              "Fit gap: the guide's inner surface is lifted this far off the bone.", decimals=2)
        guide_fields.append(("Fit gap", self.guide_gap))
        self.guide_thickness = spin(1.0, 10.0, 3.0, 0.5, " mm", "Wall thickness of the guide body.")
        guide_fields.append(("Wall thickness", self.guide_thickness))
        self.guide_rim_smoothing = spin(0.0, 3.0, 1.0, 0.2, " mm",
                                        "Smooths the outline of the footprint (fringes, notches) before extrusion.")
        guide_fields.append(("Rim smoothing", self.guide_rim_smoothing))
        self.guide_top_smoothing = qt.QDoubleSpinBox()
        self.guide_top_smoothing.setRange(0.0, 15.0)
        self.guide_top_smoothing.setSingleStep(0.5)
        self.guide_top_smoothing.setDecimals(1)
        self.guide_top_smoothing.setSuffix(" mm")
        self.guide_top_smoothing.setValue(3.0)
        self.guide_top_smoothing.setToolTip(
            "Smoothing radius of the OUTER guide surface in mm (0 = follows the bone). The bone-contact "
            "side stays exact; bone bumps are filled in; the wall never gets thinner than 80 % of the "
            "wall thickness.")
        guide_fields.append(("Top smoothing", self.guide_top_smoothing))
        for control in (self.guide_rim_smoothing, self.guide_top_smoothing, self.guide_gap, self.guide_thickness):
            control.valueChanged.connect(lambda _v: self._refresh_guide_body_if_present())

        def combo(items, default, tip):
            box = qt.QComboBox()
            for item in items:
                box.addItem(item)
            box.setCurrentIndex(items.index(default))
            box.setToolTip(tip)
            box.currentIndexChanged.connect(lambda _i: self._refresh_guide_body_if_present())
            return box
        self.guide_blade = combo(["0.4 mm", "0.6 mm", "0.8 mm", "1.0 mm", "1.27 mm"], "1.0 mm",
                                 "Saw blade thickness. Slot = blade + 0.2 mm clearance; the kerf lies in the wedge.")
        guide_fields.append(("Saw blade", self.guide_blade))
        self.guide_block_height = combo(["6 mm", "8 mm", "10 mm", "12 mm", "15 mm"], "10 mm",
                                        "Height of the saw blocks above the highest bone point (blade guidance).")
        guide_fields.append(("Slot height", self.guide_block_height))
        self.guide_pin = combo(["Auto (bone size)", "1.2 mm", "1.6 mm", "2.0 mm"], "Auto (bone size)",
                               "Fixation pin diameter. Auto: bone up to 15 mm -> 1.2 mm, larger -> 1.6 mm.")
        guide_fields.insert(0, ("Pin size", self.guide_pin))        # pin diameter first (top left)
        # parameters two per row (label, field, label, field) - keeps the panel narrow
        fields_grid = qt.QGridLayout()
        fields_grid.setContentsMargins(0, 0, 0, 0)
        fields_grid.setHorizontalSpacing(6)
        fields_grid.setVerticalSpacing(2)
        self._guide_field_labels = {}
        for index, (label, field) in enumerate(guide_fields):
            row, col = divmod(index, 2)
            text = qt.QLabel(label)
            self._guide_field_labels[label] = (text, field)
            fields_grid.addWidget(text, row, 2 * col)
            field.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            field.setMinimumWidth(60)
            if isinstance(field, qt.QComboBox):
                field.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
                field.setMinimumContentsLength(4)
            fields_grid.addWidget(field, row, 2 * col + 1)
        fields_grid.setColumnStretch(1, 1)
        fields_grid.setColumnStretch(3, 1)
        fields_widget = qt.QWidget()
        fields_widget.setLayout(fields_grid)
        guide_form.addRow(fields_widget)
        self.guide_extrude_button = qt.QPushButton("Extrude guide body")
        self.guide_extrude_button.clicked.connect(self.onExtrudeGuideBody)
        self.guide_export_button = qt.QPushButton("Export STL...")
        self.guide_export_button.setToolTip("Save the guide (body + saw blocks) as a binary STL in mm.")
        self.guide_export_button.clicked.connect(self.onExportGuideStl)
        extrude_row = qt.QHBoxLayout()
        extrude_row.setContentsMargins(0, 0, 0, 0)
        extrude_row.addWidget(self.guide_extrude_button)
        extrude_row.addWidget(self.guide_export_button)
        self.guide_clean_button = qt.QPushButton("Clean up guide")
        self.guide_clean_button.setToolTip(
            "Rebuild the guide as ONE clean, closed solid: all parts united, loose triangles and islands "
            "removed, broken edges closed, and the whole contact surface re-made as an even offset of the "
            "bone (fit gap). Run it after 'Extrude guide body', before exporting.")
        self.guide_clean_button.clicked.connect(self.onCleanGuide)
        extrude_row.addWidget(self.guide_clean_button)
        extrude_widget = qt.QWidget()
        extrude_widget.setLayout(extrude_row)
        guide_form.addRow(extrude_widget)
        self.guide_label_mode = qt.QComboBox()
        self.guide_label_mode.addItems(["Break-off tag", "On the guide", "No label"])
        self.guide_label_mode.setToolTip(
            "Patient no., bone, side and an arrow to proximal. 'Break-off tag': a small plate at the proximal "
            "end of the guide, joined by a thin neck - break it off with pliers when it is no longer needed. "
            "'On the guide': raised text on a free flat area of the guide. Built with 'Extrude guide body'.")
        guide_form.addRow("Label", self.guide_label_mode)


        self.guide_status = qt.QLabel("No footprint yet.")
        self.guide_status.setWordWrap(True)
        self.guide_status.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
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
        return (f"Doggybone {side} {bone.lower()} proximal fragment",
                f"Doggybone {side} {bone.lower()} distal fragment (corrected)",
                f"Doggybone {side} {bone.lower()} correction transform",
                f"Doggybone {side} {bone.lower()} final corrected bone (combined)",
                f"Doggybone {side} {bone.lower()} wedge REMOVE")

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
                hinge_point, ap_axis, wedge_deg, frontal_first=moving_is_distal)
            if True:   # OWO (CWO always takes the contact-congruent branch above)
                # OWO: the SAME verified correction as the SECOND-plane derivation
                # (_desired_correction_rotation: signs checked on the landmarks, angles solved,
                # derotation about the moving fragment's own axis) - so the moved cut face and the
                # virtual SECOND plane always coincide. Wedge about the hinge, torsion about the
                # true shaft line.
                fixed_o, fixed_n = ((proximal_origin, proximal_normal) if moving_is_distal
                                    else (distal_origin, distal_normal))
                _R, info = self._desired_correction_rotation(side, bone, moving_is_distal, fixed_o, fixed_n)
                if isinstance(info, dict):
                    wedge_deg = info["frontal_signed"]
                    rotation_deg = info["torsion_signed"]
                    matrix = osteotomy_logic.anatomical_correction_matrix(
                        shaft_point_at_cut, shaft_axis_check, rotation_deg, hinge_point, ap_axis, wedge_deg,
                        frontal_first=moving_is_distal)
                    alignment_report = (f"Correction verified on the landmarks: wedge {abs(wedge_deg):.1f}\N{DEGREE SIGN}, "
                                        f"torsion {abs(rotation_deg):.1f}\N{DEGREE SIGN}"
                                        + (": predicted " + self._format_prediction(bone, info["predicted"])
                                           if info.get("predicted") else "") + "."
                                        + (("\n" + info["warning"]) if info.get("warning") else ""))

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
        self._update_corrected_landmarks()

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

        def compose(R_torsion, R_frontal_part):
            # derotation about the MOVING fragment's own long axis (see anatomical_correction_matrix)
            return R_torsion @ R_frontal_part if moving_is_distal else R_frontal_part @ R_torsion

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
            frontal_signed = frontal if np.allclose(R_frontal, rot3(ap, frontal)) else -frontal
        torsion_signed = torsion
        if torsion > 0.01:
            torsion_signed = torsion if np.allclose(R, compose(rot3(shaft, torsion), R_frontal)) else -torsion
        frontal_signed, torsion_signed, _ = self._solve_correction_angles(
            lambda f, t: as_h(compose(rot3(shaft, t), rot3(ap, f))), frontal_signed, torsion_signed,
            side, bone, plane_origin, plane_normal, moving_is_distal)
        R = compose(rot3(shaft, torsion_signed), rot3(ap, frontal_signed))
        frontal, torsion = abs(frontal_signed), abs(torsion_signed)
        predicted = self._predict_measurements(side, bone, as_h(R), plane_origin, plane_normal, moving_is_distal)
        return R, {"frontal": frontal, "torsion": torsion, "predicted": predicted,
                   "frontal_signed": float(frontal_signed), "torsion_signed": float(torsion_signed),
                   "warning": getattr(self, "_correction_warning", "")}

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

    def _original_overlay_nodes(self, side):
        panel = self.left_panel if side == "left" else self.right_panel
        if panel is None:
            return []
        nodes = [slicer.mrmlScene.GetNodeByID(panel.markup_node_id)]
        nodes += list(getattr(panel, "sphere_nodes", {}).values())
        nodes += list(getattr(panel, "axis_nodes", {}).values())
        return [n for n in nodes if n is not None and n.GetScene() is not None and n.GetDisplayNode() is not None]

    def _hide_original_overlay(self, side):
        """While the corrected landmarks are shown, hide the ORIGINAL (pre-correction) landmarks,
        spheres and axes of that side; their visibility is remembered and restored later."""
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
        if not saved:
            return
        for node_id, visible in saved.items():
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(visible)

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
        if not state or not self.show_corrected_landmarks.isChecked():
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

        points, radii, moved_names = {}, {}, []
        for landmark in getattr(panel, "landmarks", []):
            if not getattr(landmark, "placed", False):
                continue
            p = np.asarray(landmark.get_position(), dtype=float)
            on_distal_side = float(np.dot(p - o, n)) * distal_sign > 0.0
            if on_distal_side == state["moving_is_distal"]:
                p = (H @ np.append(p, 1.0))[:3]
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
        display.SetGlyphScale(2.0)

        results, spheres, axes = [], {}, {}
        for bone_name in (self.WEDGE_MEASUREMENT_NAME.get(bone), self.ROTATION_MEASUREMENT_NAME.get(bone)):
            measurement = self._get_measurement_by_name(panel, bone_name) if bone_name else None
            if measurement is None:
                continue
            needed = [l.name for l in measurement.get_landmarks()]
            if any(name not in points for name in needed):
                continue
            corrected = copy.copy(measurement)          # own spheres/axes - the original stays as is
            try:
                value, description = corrected._measure(points, radii)
            except Exception as exc:                     # a measurement must never break the preview
                results.append(f"{measurement.name}: not measurable ({exc})")
                continue
            if bone_name == self.ROTATION_MEASUREMENT_NAME.get(bone):
                value = self._signed_rotation_value(bone, value, description)
                target = float(self.target_rotation_value.value)
                results.append(f"{measurement.name} {value:+.1f}\N{DEGREE SIGN} (target {target:+.1f}\N{DEGREE SIGN})")
            else:
                target = float(self.target_wedge_value.value)
                results.append(f"{description} {value:.1f}\N{DEGREE SIGN} (target {target:.1f}\N{DEGREE SIGN})")
            spheres.update(corrected.get_spheres())
            for label, ends in corrected.get_axes().items():
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
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        # without a preview the original bone must always be visible again - whatever hid it
        # (preview option, guide tools, a module reload that lost the remembered state)
        if segmentation_node is not None and segmentation_node.GetDisplayNode() is not None:
            display_node = segmentation_node.GetDisplayNode()
            display_node.SetVisibility(True)
            if hasattr(display_node, "SetVisibility3D"):
                display_node.SetVisibility3D(True)
        if hasattr(self, "_original_segmentation_visibility_3d"):
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
            # same colour code as the measurement tool: green = measured, orange = landmarks missing
            self.current_values_label.setStyleSheet(
                OSTEOTOMY_VALUE_STYLE if current_parts else STATUS_WARN_STYLE)

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
            radius_text = node.GetAttribute(f"Doggybone.radius.{landmark.name}")
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
            editor_node = slicer.mrmlScene.GetFirstNodeByName("Doggybone segment editor")
            if editor_node is None:
                editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "Doggybone segment editor")
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
        self._hide_preview_for_guide_edit(bone_lasso=True)  # cut on the visible ORIGINAL bone (any guide mode)
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

    # --- remove a whole bone (patella, fabella ...) by clicking it in 3D ------------------------------
    def onRemoveBoneToggled(self, checked):
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        interactor = view.interactor()
        tag = getattr(self, "_remove_bone_tag", None)
        if tag is not None:
            interactor.RemoveObserver(tag)
            self._remove_bone_tag = None
        if not checked:
            self.segmentation_status.setText("Bone removal off.")
            return
        self._stop_lasso()
        self._remove_bone_tag = interactor.AddObserver("LeftButtonPressEvent", self._on_remove_bone_click, 2.0)
        self.segmentation_status.setText(
            "Click the bone to remove (patella, fabella, tibia ...) in the 3D view - one bone per click, "
            "then the tool switches off. Drag on the background to rotate.")

    def _on_remove_bone_click(self, caller, event):
        if caller.GetShiftKey() or caller.GetControlKey() or caller.GetAltKey():
            return
        x, y = caller.GetEventPosition()
        bone = self._get_bone_polydata()
        if bone is None:
            return
        renderer = slicer.app.layoutManager().threeDWidget(0).threeDView().renderWindow().GetRenderers().GetFirstRenderer()
        ends = []
        for z in (0.0, 1.0):
            renderer.SetDisplayPoint(float(x), float(y), z)
            renderer.DisplayToWorld()
            wp = renderer.GetWorldPoint()
            ends.append(np.array([wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]]))
        key = (id(bone), bone.GetMTime())
        if getattr(self, "_remove_bone_locator_key", None) != key:
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(bone)
            locator.BuildLocator()
            self._remove_bone_locator, self._remove_bone_locator_key = locator, key
        t, sub_id = vtk.reference(0.0), vtk.reference(0)
        hit, pcoords = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        if not self._remove_bone_locator.IntersectWithLine(ends[0].tolist(), ends[1].tolist(), 0.0, t, hit, pcoords, sub_id):
            return                                    # background: rotate as usual
        command = caller.GetCommand(self._remove_bone_tag)
        if command is not None:
            command.SetAbortFlag(1)
        ray = ends[1] - ends[0]
        ray = ray / np.linalg.norm(ray)
        qt.QTimer.singleShot(0, lambda h=np.asarray(hit, float), r=ray: self._remove_bone_once(h, r))

    def _remove_bone_once(self, hit, ray):
        # one bone per click: the tool switches itself off again
        self.remove_bone_button.setChecked(False)
        self._remove_bone_at(hit, ray)

    def _remove_bone_at(self, hit, ray):
        """Remove the clicked bone (patella, fabella, tibia ...) from the bone segmentation.
        Works locally around the click: the bone is eroded step by step until the clicked piece comes
        loose from its neighbour (the piece no longer reaches the edge of the work box), then every bone
        voxel is given back to the piece or its neighbour by a watershed on the distance map - the cut
        runs through the narrowest point of the contact (joint space / partial-volume bridge)."""
        from vtk.util.numpy_support import vtk_to_numpy
        import scipy.ndimage as ndi
        self._remove_bone_last_click = (np.asarray(hit, float).tolist(), np.asarray(ray, float).tolist())
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None:
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            image = self._bone_labelmap_copy(segmentation_node)
            ext = image.GetExtent()
            dims = (ext[5] - ext[4] + 1, ext[3] - ext[2] + 1, ext[1] - ext[0] + 1)
            arr = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims)
            m = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(m)
            M = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])
            Minv = np.linalg.inv(M)
            spacing = np.array([np.linalg.norm(M[:3, c]) for c in range(3)])      # i, j, k
            mask = arr > 0
            # the voxel under the click: a little way INTO the bone along the view ray
            seed = None
            for depth in np.arange(0.0, 4.01, 0.25):
                p = (Minv @ np.append(hit + depth * ray, 1.0))[:3]
                i, j, k = (int(round(v)) for v in p)
                kk, jj, ii = k - ext[4], j - ext[2], i - ext[0]
                if 0 <= kk < dims[0] and 0 <= jj < dims[1] and 0 <= ii < dims[2] and mask[kk, jj, ii]:
                    seed = (kk, jj, ii)
                    break
            if seed is None:
                self.segmentation_status.setText("No bone voxel under the click - try again on the bone surface.")
                return
            # 1) the whole connected bone the click belongs to - already separate? then remove it as a whole
            labels_all, _ = ndi.label(mask)
            whole = labels_all == labels_all[seed]
            sizes_all = np.bincount(labels_all.ravel())[1:]
            if whole.sum() < 0.5 * float(sizes_all.max()):
                remove = whole
                lo = (0, 0, 0)
                note = "separate piece"
            else:
                # 2) attached to the big bone: work in a box around the click
                radius_mm = 40.0
                rad = [int(np.ceil(radius_mm / spacing[2 - d])) for d in range(3)]      # k, j, i
                lo = [max(0, seed[d] - rad[d]) for d in range(3)]
                hi = [min(dims[d], seed[d] + rad[d] + 1) for d in range(3)]
                sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] & whole[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                local_seed = tuple(seed[d] - lo[d] for d in range(3))
                edt = ndi.distance_transform_edt(sub, sampling=tuple(spacing[::-1]))
                border = np.zeros(sub.shape, dtype=bool)
                for d in range(3):
                    sl = [slice(None)] * 3
                    if lo[d] > 0:
                        sl[d] = 0
                        border[tuple(sl)] = True
                    if hi[d] < dims[d]:
                        sl[d] = -1
                        border[tuple(sl)] = True
                piece_core = None
                used_depth = 0.0
                for depth in np.arange(0.4, 1.61, 0.2):            # erosion depth in mm (deeper would cut off the femoral head)
                    core = edt > depth
                    labels, count = ndi.label(core)
                    if count == 0:
                        break
                    seed_label = int(labels[local_seed])
                    if seed_label == 0:            # the click voxel eroded away: nearest core voxel
                        window = labels[tuple(slice(max(0, c - 8), c + 9) for c in local_seed)]
                        found = window[window > 0]
                        if len(found) == 0:
                            break
                        seed_label = int(np.bincount(found).argmax())
                    candidate = labels == seed_label
                    if not (candidate & border).any():
                        piece_core, used_depth = candidate, float(depth)
                        other_core = core & ~candidate
                        break
                if piece_core is None:
                    self.segmentation_status.setText(
                        "That bone could not be separated from its neighbour here (it is the femur, or the contact "
                        "is too broad). Click the middle of the patella / fabella, or cut the bridge with the lasso.")
                    return
                # 3) watershed: every bone voxel goes to the nearer core, cut along the narrowest contact
                markers = np.zeros(sub.shape, dtype=np.int16)
                markers[other_core] = 2
                markers[piece_core] = 1
                markers[~sub] = 3                  # background is its own basin
                cost = (np.clip(edt.max() - edt, 0, None) / max(edt.max(), 1e-6) * 65000).astype(np.uint16)
                basins = ndi.watershed_ift(cost, markers)
                remove_local = (basins == 1) & sub
                remove = np.zeros(mask.shape, dtype=bool)
                remove[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = remove_local
                note = f"cut free at {used_depth:.1f} mm"
            removed = int(remove.sum())
            if removed == 0:
                self.segmentation_status.setText("Nothing to remove there.")
                return
            self._bone_push_undo(segmentation_node)
            arr[remove] = 0
            image.GetPointData().GetScalars().Modified()
            image.Modified()
            self._bone_set_labelmap(segmentation_node, image)
        finally:
            qt.QApplication.restoreOverrideCursor()
        for name in ("Doggybone guide bone surface", "Doggybone guide bone surface (reduction)"):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
        self.segmentation_status.setText(
            f"Bone removed ({removed * float(np.prod(spacing)):.0f} mm\N{SUPERSCRIPT THREE}, {note}). "
            "For the next bone press 'Remove bone (click)' again. 'Delete' undoes all clean-up edits.")

    def onFillCavities(self):
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        import scipy.ndimage as ndi
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
            self.segmentation_status.setText("No segmentation - run 'Apply' first.")
            return
        self._stop_lasso()
        self._bone_push_undo(segmentation_node)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            image = self._bone_labelmap_copy(segmentation_node)
            ext = image.GetExtent()
            dims = (ext[5] - ext[4] + 1, ext[3] - ext[2] + 1, ext[1] - ext[0] + 1)
            scalars = vtk_to_numpy(image.GetPointData().GetScalars())
            arr = scalars.reshape(dims)
            mask = arr > 0
            label = int(arr[mask].max()) if mask.any() else 1
            before = int(mask.sum())
            mask = ndi.binary_closing(mask, iterations=1) | mask          # seal hairline cortical gaps
            # slices across the bone: along the image axis in which the bone is longest
            coords = np.nonzero(mask)
            extents = [int(c.max() - c.min()) if len(c) else 0 for c in coords]
            m = vtk.vtkMatrix4x4()
            image.GetImageToWorldMatrix(m)
            spacing = [np.linalg.norm([m.GetElement(r, c) for r in range(3)]) for c in (2, 1, 0)]   # (k, j, i)
            axis = int(np.argmax([e * s for e, s in zip(extents, spacing)]))
            for index in range(mask.shape[axis]):
                sl = [slice(None)] * 3
                sl[axis] = index
                face = mask[tuple(sl)]
                if face.any():
                    mask[tuple(sl)] = ndi.binary_fill_holes(face)
            mask = ndi.binary_fill_holes(mask)
            arr[...] = np.where(mask, label, 0).astype(arr.dtype)
            image.GetPointData().GetScalars().Modified()
            image.Modified()
            self._bone_set_labelmap(segmentation_node, image)
        finally:
            qt.QApplication.restoreOverrideCursor()
        added = int(mask.sum()) - before
        # the prepared guide surfaces were made from the old bone - rebuild them on the next paint
        for name in ("Doggybone guide bone surface", "Doggybone guide bone surface (reduction)"):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
        self.segmentation_status.setText(
            f"Cavities filled: {added} voxels added - the bone is solid, only its outer surface remains. "
            "Paint the guide footprint again ('Delete' undoes the fill).")

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
    # Two guides share the whole painting/extrusion workflow: the SAW guide (on the bone as it is)
    # and the REDUCTION guide (on the fragments in the corrected position). Every guide node name
    # below depends on the active mode, so the tools work on the selected guide's own nodes.
    def _mode_name(self, base):
        return base if getattr(self, "_guide_mode", "saw") == "saw" else base + " (reduction)"

    def _guide_mode_is_reduction(self):
        return getattr(self, "_guide_mode", "saw") == "reduction"

    GUIDE_MESH_NAME = property(lambda self: self._mode_name("Doggybone guide bone surface"))
    GUIDE_FOOTPRINT_NAME = property(lambda self: self._mode_name("Doggybone guide footprint"))
    GUIDE_BODY_NAME = property(lambda self: self._mode_name("Doggybone guide body"))
    GUIDE_SEGMENTATION_NAME = "Doggybone guide"   # earlier voxel-based version (hidden if present)

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

    def _guide_external(self, mesh, recompute=False):
        '''Per-vertex mask of surface facing the OUTSIDE of the bone (no pores, cancellous bone
        or medullary canal). Computed from the bone segmentation once per prepared surface and
        stored on the mesh; computed on demand for surfaces prepared before it existed.'''
        if not recompute and mesh.GetPointData().GetArray("External") is not None:
            return guide_logic.get_selection(mesh, name="External")
        if self._guide_mode_is_reduction():
            # moved fragment no longer matches the labelmap; its ROI is the outer surface already
            external = np.ones(mesh.GetNumberOfPoints(), dtype=bool)
            guide_logic.set_selection(mesh, external, name="External")
            return external
        segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if segmentation_node is None or segmentation_node.GetSegmentation().GetNumberOfSegments() == 0:
            return np.ones(mesh.GetNumberOfPoints(), dtype=bool)
        segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)
        segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
        labelmap = segment.GetRepresentation(
            slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
        if labelmap is None:
            return np.ones(mesh.GetNumberOfPoints(), dtype=bool)
        external = guide_logic.external_mask(mesh, labelmap, segment.GetLabelValue())
        guide_logic.set_selection(mesh, external, name="External")
        return external

    @staticmethod
    def _guide_outer(mesh):
        '''Outer-surface mask of the guide side (all True for meshes made before it existed).'''
        if mesh.GetPointData().GetArray("Outer") is None:
            return np.ones(mesh.GetNumberOfPoints(), dtype=bool)
        return guide_logic.get_selection(mesh, name="Outer")

    def onSuggestGuideFootprint(self, empty=False):
        '''Prepare the bone surface around the osteotomy and suggest a footprint on the chosen
        side. empty=True prepares the surface only (nothing coloured) - used when a brush / lasso
        tool is switched on before any footprint exists.'''
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
        bone_polydata = self._guide_source_polydata()
        if bone_polydata is None or bone_polydata.GetNumberOfPoints() == 0:
            self.guide_status.setText(
                "No corrected fragments - run the correction preview first." if self._guide_mode_is_reduction()
                else "No bone surface - run 'Apply' in 3D Segmentation first.")
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            origins = [np.asarray(self._get_plane_origin_normal(node)[0], dtype=float) for node in plane_nodes]
            origins = self._guide_mode_plane_origins(plane_nodes, origins)
            centre = np.mean(origins, axis=0)
            t_values = [float(np.dot(o - centre, shaft)) for o in origins]
            margin = float(self.guide_footprint_margin.value)
            half_width = 0.5 * float(self.guide_footprint_width.value)
            t_min, t_max = min(t_values) - margin, max(t_values) + margin
            mesh = self._extract_guide_roi(bone_polydata, centre, shaft, across, d,
                                           (t_min - 15.0, t_max + 15.0), half_width + 20.0, 45.0)
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
            external = self._guide_external(mesh, recompute=True)
            outer = outer & external
            selected = selected & outer
            if empty:
                selected = np.zeros_like(selected)
            guide_logic.set_selection(mesh, selected)
            guide_logic.set_selection(mesh, outer, name="Outer")

            mesh_node = self._guide_mesh_node()
            if mesh_node is None:
                mesh_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_MESH_NAME)
                mesh_node.CreateDefaultDisplayNodes()
            mesh_node.SetAndObservePolyData(mesh)
            mesh_node.GetDisplayNode().SetVisibility(False)   # computation only; you see the bone
            mesh_node.SetAttribute("Doggybone.CorrectionSignature", self._correction_signature())
            self._guide_undo_stack = []
            self._set_guide_body(None)   # an old body does not match the new footprint
            old = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_SEGMENTATION_NAME)
            if old is not None and old.GetDisplayNode() is not None:
                old.GetDisplayNode().SetVisibility(False)
            area = self._update_guide_footprint_display()
            if empty:
                self.guide_status.setText(
                    "Bone surface around the osteotomy prepared - paint the footprint with Brush + / "
                    "Lasso + (rotate the view to reach every side).")
                return
            self.guide_status.setText(
                f"Footprint suggested on the {self.guide_side_selector.currentText.lower()} side of the "
                f"{side} {bone.lower()}: {int(selected.sum())} surface points, ~{area:.0f} mm\N{SUPERSCRIPT TWO}. "
                f"Refine with brush / lasso, then 'Extrude guide body'.")
        finally:
            qt.QApplication.restoreOverrideCursor()

    GUIDE_RELIEF_NAME = property(lambda self: self._mode_name("Doggybone guide smooth marks"))

    @staticmethod
    def _guide_relief(mesh):
        if mesh.GetPointData().GetArray("Relief") is None:
            return np.zeros(mesh.GetNumberOfPoints(), dtype=bool)
        return guide_logic.get_selection(mesh, name="Relief").copy()

    def _update_guide_relief_display(self, mesh):
        marks = self._guide_relief(mesh) & guide_logic.get_selection(mesh)
        patch = guide_logic.patch_from_selection(mesh, marks, lift=0.14)
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_RELIEF_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_RELIEF_NAME)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(1.0, 0.9, 0.1)
            node.GetDisplayNode().SetScalarVisibility(False)
        node.SetAndObservePolyData(patch)
        node.GetDisplayNode().SetVisibility(True)

    def _update_guide_footprint_display(self):
        mesh_node = self._guide_mesh_node()
        if mesh_node is None:
            return 0.0
        mesh = mesh_node.GetPolyData()
        self._update_guide_relief_display(mesh)
        patch = guide_logic.patch_from_selection(mesh, guide_logic.get_selection(mesh) & self._guide_external(mesh), lift=0.08)
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
    def _correction_signature(self):
        """Fingerprint of the current correction (incl. fine shift) - the reduction surface must be
        rebuilt when it changes, since the moving fragment is somewhere else then."""
        if not self._guide_mode_is_reduction():
            return ""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        transform = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[2])
        if transform is None:
            return "none"
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)
        return ";".join(f"{matrix.GetElement(r, c):.3f}" for r in range(3) for c in range(4))

    def _ensure_current_reduction_surface(self):
        """Reduction guide: rebuild the prepared surface if the correction changed since it was made.
        Returns True if it was rebuilt (the old footprint no longer fits and is discarded)."""
        if not self._guide_mode_is_reduction():
            return False
        node = self._guide_mesh_node()
        if node is None or node.GetAttribute("Doggybone.CorrectionSignature") == self._correction_signature():
            return False
        self.onSuggestGuideFootprint(empty=True)
        self.guide_status.setText("The correction changed since the reduction surface was prepared - it was "
                                  "rebuilt on the fragments in the CURRENT corrected position (old footprint "
                                  "discarded). Paint again with Brush + / Lasso +.")
        return True

    def onGuideToolToggled(self, key, checked):
        if checked and self._ensure_current_reduction_surface():
            button = self._guide_tool_buttons[key]
            was_blocked = button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(was_blocked)
        if not checked:
            if getattr(self, "_guide_active_tool", None) == key:
                self._stop_guide_edit()   # placement stays paused until a landmark is clicked
                if key == "pins" and getattr(self, "_pins_dirty", False):
                    self._pins_dirty = False
                    self.onExtrudeGuideBody()          # marked pins -> sleeves and bores
            return
        if self._guide_mesh_node() is None:
            # No surface yet (e.g. new scene): prepare it now instead of refusing the tool.
            self.onSuggestGuideFootprint(empty=True)
        if self._guide_mesh_node() is None:
            message = self.guide_status.text   # why the surface could not be prepared
            button = self._guide_tool_buttons[key]
            was_blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(was_blocked)
            slicer.util.warningDisplay(f"Brush / lasso cannot start: {message}", windowTitle="Doggybone 3D")
            return
        if key == "pins" and not self._guide_mode_is_reduction():
            button = self._guide_tool_buttons[key]
            was_blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(was_blocked)
            self.guide_status.setText("Pins are set AFTER the reduction: switch to 'Reduction guide' and click "
                                      "on the corrected bone - the saw guide gets its (pre-angled) sleeves from them.")
            return
        if key == "brush_smooth" and slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME) is None:
            button = self._guide_tool_buttons[key]
            was_blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(was_blocked)
            self.guide_status.setText("This tool works on the finished guide - Extrude guide body first.")
            return
        for other, button in self._guide_tool_buttons.items():
            if other != key and button.checked:
                was_blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(was_blocked)
        previous = getattr(self, "_guide_active_tool", None)
        if previous in (None, "bone_lasso"):
            self._stop_lasso()
            self._pause_landmark_placement()   # no landmark placement while editing
            self._show_view_for_tool(key)
        elif (previous == "brush_smooth") != (key == "brush_smooth"):
            self._restore_guide_hidden()
            self._show_view_for_tool(key)
        self._guide_active_tool = key
        self._install_guide_observers()
        texts = {"brush_add": "Brush +: hold the left mouse button and move over the bone surface to colour it.",
                 "brush_remove": "Brush -: hold the left mouse button and move over the coloured area to remove it.",
                 "pins": "Pins: click on the guide to mark where a pin should go (click a mark again to remove "
                        "it). Not in the wedge and not in the saw blocks. Switch Pins off (or Extrude) to build them.",
                 "brush_smooth": "Smooth: only the guide is shown. Turn it to see its inner (contact) side and "
                                 "paint over irregular spots (yellow) - they are bridged, never moved into the bone. "
                                 "A stroke that starts on yellow erases.",
                 "lasso_add": "Lasso +: draw around the area to colour (only the visible surface is affected).",
                 "lasso_remove": "Lasso -: draw around the area to uncolour (only the visible surface is affected)."}
        hint = (" Drag on the background to rotate." if key.startswith("brush") or key == "pins"
                else " The lasso switches off after each stroke (click it again for the next one).")
        self.guide_status.setText(texts[key] + hint + " Shift+drag pans, right button / wheel zooms. "
                                  "Click the tool again to stop.")

    def _hide_preview_for_guide_edit(self, bone_lasso=False):
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        self._guide_hidden_preview = []
        self._guide_locked_planes = []
        # painting: the bone surface fully OPAQUE (only the outer surface visible, no spongiosa /
        # medullary canal shining through); the user's opacity comes back when the tool is switched off
        already = {entry[0] for entry in getattr(self, "_guide_saved_opacity", [])}
        self._guide_saved_opacity = list(getattr(self, "_guide_saved_opacity", []))
        for name in (self.SEGMENTATION_NODE_NAME, self._preview_node_names(side, bone)[3]):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            display = node.GetDisplayNode() if node is not None else None
            if display is None:
                continue
            if node.GetID() in already:                 # saved before (tool switched): keep the user's value
                display.SetOpacity3D(1.0) if hasattr(display, "GetOpacity3D") else display.SetOpacity(1.0)
                continue
            if hasattr(display, "GetOpacity3D"):
                self._guide_saved_opacity.append((node.GetID(), "3d", float(display.GetOpacity3D())))
                display.SetOpacity3D(1.0)
            else:
                self._guide_saved_opacity.append((node.GetID(), "model", float(display.GetOpacity())))
                display.SetOpacity(1.0)
        # Painting the guide footprint: the cutting planes STAY visible for orientation (the lasso's
        # visibility test renders the bone surface alone, so they do not hide anything) - only locked,
        # so a stroke never grabs a plane handle. The bone lasso (segmentation) and the X/Y gizmo: hidden.
        plane_names = (self._plane_node_name(side, bone), self._distal_plane_node_name(side, bone))
        for name in plane_names + (self._fine_shift_node_names(side, bone)[1],):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None or node.GetDisplayNode() is None or not node.GetDisplayNode().GetVisibility():
                continue
            if name in plane_names and not bone_lasso:
                display = node.GetDisplayNode()
                self._guide_locked_planes.append((node.GetID(), bool(node.GetLocked()),
                                                  bool(display.GetHandlesInteractive())))
                node.SetLocked(True)
                display.SetHandlesInteractive(False)
                continue
            node.GetDisplayNode().SetVisibility(False)
            self._guide_hidden_preview.append(node.GetID())
        for name in self._preview_node_names(side, bone):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.IsA("vtkMRMLModelNode") and node.GetDisplayNode() is not None \
                    and node.GetDisplayNode().GetVisibility():
                node.GetDisplayNode().SetVisibility(False)
                self._guide_hidden_preview.append(node.GetID())
        bone_segmentation = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
        if self._guide_mode_is_reduction() and not bone_lasso:
            # paint on what the reduction guide sits on: the fragments in the corrected position
            if bone_segmentation is not None and bone_segmentation.GetDisplayNode() is not None \
                    and bone_segmentation.GetDisplayNode().GetVisibility():
                bone_segmentation.GetDisplayNode().SetVisibility(False)
                self._guide_hidden_preview.append(bone_segmentation.GetID())
            combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
            if combined is not None and combined.GetDisplayNode() is not None:
                combined.GetDisplayNode().SetVisibility(True)
                if combined.GetID() in self._guide_hidden_preview:
                    self._guide_hidden_preview.remove(combined.GetID())
                self._show_only_guide_surface(combined)
            return
        if bone_segmentation is not None and bone_segmentation.GetDisplayNode() is not None:
            bone_segmentation.GetDisplayNode().SetVisibility(True)      # may be off after a guide-mode switch
            bone_segmentation.GetDisplayNode().SetVisibility3D(True)
        if not bone_lasso:
            self._show_only_guide_surface(bone_segmentation)

    def _show_only_guide_surface(self, bone_node):
        """While painting: the femur surface around the osteotomy alone - patella, fabellae, tibia are hidden,
        so the footprint can be painted where they lie close to it (the tools only ever paint this surface)."""
        mesh_node = self._guide_mesh_node()
        if mesh_node is None or mesh_node.GetDisplayNode() is None or bone_node is None \
                or bone_node.GetDisplayNode() is None:
            return
        if bone_node.GetDisplayNode().GetVisibility():
            bone_node.GetDisplayNode().SetVisibility(False)
            self._guide_hidden_preview.append(bone_node.GetID())
        display = mesh_node.GetDisplayNode()
        if not display.GetVisibility():
            display.SetColor(0.93, 0.87, 0.6)
            display.SetOpacity(1.0)
            display.SetVisibility(True)
            self._guide_shown_surface = mesh_node.GetID()

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
        self._guide_bone_locator = self._guide_cell_locator
        if self._guide_active_tool == "brush_smooth" or (
                self._guide_active_tool == "pins" and not self._guide_mode_is_reduction()):
            self._build_guide_body_locator()          # reduction pins: picked on the corrected bone
        self._guide_point_locator = vtk.vtkPointLocator()
        self._guide_point_locator.SetDataSet(mesh)
        self._guide_point_locator.BuildLocator()

    def _build_guide_body_locator(self):
        """Smooth tool: picking happens on the finished guide, not on the bone."""
        body = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if body is None or body.GetPolyData() is None:
            return
        self._guide_body_polydata_for_pick = body.GetPolyData()
        self._guide_cell_locator = vtk.vtkCellLocator()
        self._guide_cell_locator.SetDataSet(self._guide_body_polydata_for_pick)
        self._guide_cell_locator.BuildLocator()

    def _smooth_contact_hit(self, x, y):
        """Point on the guide's CONTACT side under the mouse (within 1.5 mm of the bone) or None."""
        hit = self._guide_pick(x, y)
        if hit is None:
            return None
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        self._guide_bone_locator.FindClosestPoint(hit, closest, cell_id, sub_id, dist2)
        return hit if float(dist2) ** 0.5 <= 1.5 else None

    def _remove_guide_observers(self):
        for interactor, tag in getattr(self, "_guide_observers", []):
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                logging.exception("Doggybone: non-critical step failed (continuing)")
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
            # Cmd/Ctrl + click on a pin (reduction): select it for the gizmo
            if tool == "pins" and caller.GetControlKey() and not caller.GetShiftKey() \
                    and self._guide_mode_is_reduction():
                mark = self._pin_under_mouse(x, y)
                if mark is not None:
                    self._guide_drag = False
                    self._abort_guide_event(caller, event)
                    qt.QTimer.singleShot(0, lambda i=mark: self._select_reduction_pin(i, announce=True))
                    return
            # With Shift / Ctrl / Alt the mouse keeps its normal function (Shift+drag = pan).
            if caller.GetShiftKey() or caller.GetControlKey() or caller.GetAltKey():
                self._guide_drag = False
                return
            if tool == "pins":
                self._guide_drag = False
                if self._pin_gizmo_hovered():
                    return                         # the gizmo gets the drag, no new pin
                mark = self._pin_under_mouse(x, y)
                hit = None if mark is not None else self._guide_pick(x, y)
                if mark is None and hit is None:
                    return                         # background: rotate the view
                self._abort_guide_event(caller, event)
                if mark is not None:
                    qt.QTimer.singleShot(0, lambda i=mark: self._remove_pin(i))
                else:
                    qt.QTimer.singleShot(0, lambda h=list(hit): self._toggle_pin(h))
                return
            if tool.startswith("brush"):
                hit = self._smooth_contact_hit(x, y) if tool == "brush_smooth" else self._guide_pick(x, y)
                if hit is None:
                    self._guide_drag = False      # drag started on the background: rotate the view
                    return
                if tool == "brush_smooth":
                    mesh = self._guide_mesh_node().GetPolyData()
                    nearest = self._guide_point_locator.FindClosestPoint(hit)
                    self._smooth_erase = bool(self._guide_relief(mesh)[nearest])
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
            if tool == "brush_smooth":
                hit = self._smooth_contact_hit(x, y)
                if hit is not None:
                    self._guide_brush_at(x, y, False, hit=hit)
            elif tool.startswith("brush"):
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
            if tool == "brush_smooth":
                self._build_guide_body_locator()
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
        radius = float(self.guide_brush_radius.value)
        if getattr(self, "_guide_active_tool", None) == "brush_smooth":
            radius += 1.0      # the hit lies on the guide, up to ~1 mm off the bone
        self._guide_point_locator.FindPointsWithinRadius(radius, hit, ids)
        if ids.GetNumberOfIds() == 0:
            return
        idx = np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=np.int64)
        mesh = self._guide_mesh_node().GetPolyData()
        if getattr(self, "_guide_active_tool", None) == "brush_smooth":
            idx = idx[guide_logic.get_selection(mesh)[idx]]          # only inside the footprint
            relief = self._guide_relief(mesh)
            relief[idx] = not getattr(self, "_smooth_erase", False)
            guide_logic.set_selection(mesh, relief, name="Relief")
            self._update_guide_relief_display(mesh)
            return
        P, N = guide_logic.points_and_normals(mesh)
        camera = np.asarray(renderer.GetActiveCamera().GetPosition(), dtype=float)
        facing = np.sum(N[idx] * (camera - P[idx]), axis=1) > 0.0
        keep = facing & self._guide_external(mesh)[idx] & (
            self._guide_outer(mesh)[idx] if self.guide_side_only_checkbox.checked
            else np.ones(len(idx), dtype=bool))
        # Line of sight: a point counts only if nothing of the bone lies between it and the camera
        # (no painting of cancellous / inner surfaces behind the visible cortex).
        t = vtk.reference(0.0); sub_id = vtk.reference(0); hit_point = [0.0, 0.0, 0.0]; pc = [0.0, 0.0, 0.0]
        for n, i in enumerate(idx):
            if not keep[n]:
                continue
            direction = P[i] - camera
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue
            end = P[i] - direction / length * 0.3
            if self._guide_cell_locator.IntersectWithLine(camera.tolist(), end.tolist(), 0.0, t, hit_point, pc, sub_id):
                keep[n] = False
        tool = getattr(self, "_guide_active_tool", None)
        if tool == "brush_smooth":
            relief = self._guide_relief(mesh)
            relief[idx[keep]] = True
            guide_logic.set_selection(mesh, relief, name="Relief")
        else:
            selection = guide_logic.get_selection(mesh)
            selection[idx[keep]] = add
            guide_logic.set_selection(mesh, selection)
            if not add:
                relief = self._guide_relief(mesh)
                relief[idx[keep]] = False
                guide_logic.set_selection(mesh, relief, name="Relief")
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
        # Depth from the bone surface ONLY: slice planes shown in 3D, cutting planes, gizmos and
        # other models must not hide the bone for the visibility test.
        Z = self._guide_surface_depth(render_window, renderer, mesh, width, height)
        xi = np.clip(np.round(xs).astype(int), 0, width - 1)
        yi = np.clip(np.round(ys).astype(int), 0, height - 1)
        on_screen = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        visible = on_screen & (depth <= Z[yi, xi] + 2e-3)
        facing = np.sum(N * (np.asarray(camera.GetPosition(), dtype=float) - P), axis=1) > 0.0
        inside = guide_logic.points_in_polygon(np.c_[xs, ys], polygon)
        affected = inside & visible & facing & self._guide_external(mesh)
        if self.guide_side_only_checkbox.checked:
            affected &= self._guide_outer(mesh)
        selection = guide_logic.get_selection(mesh)
        selection[affected] = add
        guide_logic.set_selection(mesh, selection)
        area = self._update_guide_footprint_display()
        self.guide_status.setText(f"{'Added' if add else 'Removed'} {int(affected.sum())} surface points; "
                                  f"footprint now ~{area:.0f} mm\N{SUPERSCRIPT TWO}.")

    def _guide_surface_depth(self, render_window, renderer, mesh, width, height):
        '''Z-buffer of the 3D view with only the prepared bone surface rendered (all other props
        hidden for one frame, then restored).'''
        from vtk.util.numpy_support import vtk_to_numpy
        props = renderer.GetViewProps()
        props.InitTraversal()
        hidden = []
        for _ in range(props.GetNumberOfItems()):
            prop = props.GetNextProp()
            if prop is not None and prop.GetVisibility():
                prop.VisibilityOff()
                hidden.append(prop)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        try:
            render_window.Render()
            zbuffer = vtk.vtkFloatArray()
            render_window.GetZbufferData(0, 0, width - 1, height - 1, zbuffer)
            Z = vtk_to_numpy(zbuffer).reshape(height, width).copy()
        finally:
            renderer.RemoveActor(actor)
            for prop in hidden:
                prop.VisibilityOn()
            render_window.Render()
        return Z

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
                logging.exception("Doggybone: non-critical step failed (continuing)")
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
        if getattr(self, "_guide_active_tool", None) == "brush_smooth" or (
                getattr(self, "_guide_active_tool", None) == "pins" and not self._guide_mode_is_reduction()):
            self._build_guide_body_locator()      # always pick on the current guide

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

    def onCleanGuide(self):
        body_node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if body_node is None or body_node.GetPolyData() is None or body_node.GetPolyData().GetNumberOfPoints() == 0:
            self.guide_status.setText("No guide body - press 'Extrude guide body' first.")
            return
        if self._guide_mode_is_reduction():
            side = self.osteotomy_side.currentText.lower()
            bone = self.osteotomy_bone.currentText
            combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
            bone_polydata = combined.GetPolyData() if combined is not None else None
        else:
            bone_polydata = self._get_bone_polydata()
        labelmap, label_value = None, 1
        if not self._guide_mode_is_reduction():
            segmentation_node = slicer.mrmlScene.GetFirstNodeByName(self.SEGMENTATION_NODE_NAME)
            if segmentation_node is not None and segmentation_node.GetSegmentation().GetNumberOfSegments():
                segment = segmentation_node.GetSegmentation().GetSegment(segmentation_node.GetSegmentation().GetNthSegmentID(0))
                labelmap = segment.GetRepresentation(
                    slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
                label_value = segment.GetLabelValue()
        _d, shaft, _a, frame_error = self._guide_direction_frame(self.osteotomy_side.currentText.lower(),
                                                                 self.osteotomy_bone.currentText)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            cleaned, info = guide_logic.clean_guide(body_node.GetPolyData(), bone_polydata, float(self.guide_gap.value),
                                                    bone_labelmap=labelmap, label_value=label_value,
                                                    fill_axis=None if frame_error else shaft)
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._guide_body_before_clean = vtk.vtkPolyData()
        self._guide_body_before_clean.DeepCopy(body_node.GetPolyData())
        self._set_guide_body(cleaned)
        self.guide_status.setStyleSheet(STATUS_OK_STYLE if info["open_edges"] == 0 and info["bodies_after"] == 1
                                        else STATUS_WARN_STYLE)
        self.guide_status.setText(
            f"Guide cleaned: {info['parts_before']} parts -> {info['bodies_after']} solid body, "
            f"{info['open_edges']} open edges, {info['islands_dropped']} loose island(s) removed, contact surface "
            f"re-made {self.guide_gap.value:.2f} mm off the bone, volume {info['volume_mm3']:.0f} mm\N{SUPERSCRIPT THREE}. "
            "Check it, then 'Export STL...'.")

    def onGuideDelete(self):
        if not self._guide_mode_is_reduction():
            for label in ("proximal", "distal"):
                node = slicer.mrmlScene.GetFirstNodeByName(self._saw_box_gizmo_name(label))
                if node is not None:
                    slicer.mrmlScene.RemoveNode(node)
        '''Delete this guide: guide body, this mode's pins (+ axes, row handles) and - if there is one -
        the painted footprint (the surface stays, so painting can start from scratch).'''
        self._stop_guide_edit()
        mesh_node = self._guide_mesh_node()
        if mesh_node is not None and mesh_node.GetPolyData() is not None:
            mesh = mesh_node.GetPolyData()
            guide_logic.set_selection(mesh, np.zeros(mesh.GetNumberOfPoints(), dtype=bool))
            guide_logic.set_selection(mesh, np.zeros(mesh.GetNumberOfPoints(), dtype=bool), name="Relief")
            self._update_guide_footprint_display()
        pins_node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        if pins_node is not None:
            pins_node.RemoveAllControlPoints()          # this mode's own pin marks
        if self._guide_mode_is_reduction():
            handles = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
            if handles is not None:
                handles.RemoveAllControlPoints()
            self._pin_rows_active = False
            self._select_reduction_pin(None)
        self._pins_dirty = False
        self._set_pin_axes(None)
        self._set_guide_body(None)
        self._guide_undo_stack = []
        if self._guide_mode_is_reduction():
            self.guide_status.setText("Reduction guide and its pins deleted. Place the pin rows again.")
        else:
            self.guide_status.setText("Footprint and guide body deleted. Paint again with Brush + / Lasso +.")
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
        self._restore_guide_hidden()

    def _restore_guide_hidden(self):
        shown = getattr(self, "_guide_shown_surface", None)
        if shown is not None:
            node = slicer.mrmlScene.GetNodeByID(shown)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(False)
            self._guide_shown_surface = None
        for node_id, kind, value in getattr(self, "_guide_saved_opacity", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            display = node.GetDisplayNode() if node is not None else None
            if display is None:
                continue
            if kind == "3d":
                display.SetOpacity3D(value)
            else:
                display.SetOpacity(value)
        self._guide_saved_opacity = []
        for node_id, locked, handles in getattr(self, "_guide_locked_planes", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                node.SetLocked(locked)
                if node.GetDisplayNode() is not None:
                    node.GetDisplayNode().SetHandlesInteractive(handles)
        self._guide_locked_planes = []
        for node_id in getattr(self, "_guide_hidden_preview", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(True)
        self._guide_hidden_preview = []
        opacity = getattr(self, "_guide_body_opacity", None)
        body = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if opacity is not None and body is not None and body.GetDisplayNode() is not None:
            body.GetDisplayNode().SetOpacity(opacity)
        self._guide_body_opacity = None

    def _show_view_for_tool(self, key):
        if key == "brush_smooth":
            self._hide_all_but_guide()
        else:
            self._hide_preview_for_guide_edit()

    def _hide_all_but_guide(self):
        """Contact retouch: only the guide (and the yellow marks) stay visible in 3D."""
        keep = {self.GUIDE_BODY_NAME, self.GUIDE_RELIEF_NAME}
        self._guide_hidden_preview = []
        for class_name in ("vtkMRMLModelNode", "vtkMRMLSegmentationNode", "vtkMRMLMarkupsNode"):
            for node in slicer.util.getNodesByClass(class_name):
                if node.GetName() in keep or node.GetHideFromEditors():
                    continue
                display = node.GetDisplayNode()
                if display is not None and display.GetVisibility():
                    display.SetVisibility(False)
                    self._guide_hidden_preview.append(node.GetID())
        body = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if body is not None and body.GetDisplayNode() is not None:
            self._guide_body_opacity = body.GetDisplayNode().GetOpacity()
            body.GetDisplayNode().SetOpacity(1.0)
            body.GetDisplayNode().SetVisibility(True)
        marks = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_RELIEF_NAME)
        if marks is not None and marks.GetDisplayNode() is not None:
            marks.GetDisplayNode().SetVisibility(True)

    # --- extrusion --------------------------------------------------------
    def onExtrudeGuideBody(self):
        self._stop_guide_edit()
        self._extrude_guide_body(push_undo=True)

    def _extrude_guide_body(self, push_undo=True):
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
        selection = guide_logic.smooth_selection_rim(mesh, selection, float(self.guide_rim_smoothing.value)) & external
        patch, used = guide_logic.patch_from_selection(mesh, selection, lift=0.0, return_ids=True)
        relief = self._guide_relief(mesh)
        if len(used) and relief[used].any():
            patch = guide_logic.apply_contact_relief(patch, relief[used], mesh)
        patch = guide_logic.make_patch_manifold(patch)
        # A guide is ONE piece: keep only the largest connected part (drops detached fragments).
        patch = guide_logic.clean_patch(patch, keep_fraction=0.999)
        outline_points = None
        if patch.GetNumberOfCells() > 0:
            bone_locator = vtk.vtkCellLocator()
            bone_locator.SetDataSet(mesh)
            bone_locator.BuildLocator()
            patch = guide_logic.smooth_patch_outline(patch, bone_locator, float(self.guide_rim_smoothing.value))
            from vtk.util.numpy_support import vtk_to_numpy
            outline_points = vtk_to_numpy(patch.GetPoints().GetData()).astype(float)[
                guide_logic.boundary_point_ids(patch)]
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
            guide_logic.configure_slots(
                blade_thickness=float(self.guide_blade.currentText.split()[0]),
                block_height=float(self.guide_block_height.currentText.split()[0]))
            if self._guide_mode_is_reduction():
                slots, slot_notes, pin = [], [], ""          # no saw slots on the reduction guide
            else:
                slots, slot_notes, pin = self._plan_saw_slots(patch, gap)
                self._place_saw_box_gizmos(slots)
            if slots:
                patch = guide_logic.cut_slot_holes(patch, slots, extra=1.0)
            if self._guide_mode_is_reduction():
                pins, pin_note = self._plan_reduction_pins(mesh, gap)
            else:
                pins, pin_note = self._plan_pins(slots, mesh, gap)
                self._store_planned_pins(pins)
            if pins:
                patch = guide_logic.cut_pin_holes(patch, pins)
            if slots or pins:
                patch = guide_logic.make_patch_manifold(patch)
            body = guide_logic.extrude_patch(patch, gap, thickness,
                                             top_smoothing=float(self.guide_top_smoothing.value),
                                             ribs=slots or None, outline_points=outline_points,
                                             pins=pins or None)
            body, islands = guide_logic.drop_small_regions(body, min_volume=15.0, min_fraction=0.0)
            if slots:
                # the body is extruded along the bone normals and leans into the kerfs - push it back
                body, _pushed = guide_logic.push_out_of_kerfs(body, slots)
            parts = [body]
            if len(slots) == 2:
                # ONE block for both cuts: outer faces extended until they close, space between filled,
                # both kerfs cut through - no crossing plates, no fragments
                parts.append(guide_logic.saw_block_pair(slots[0], slots[1], self._pin_obb_tree(mesh), gap))
            else:
                for slot in slots:
                    parts += guide_logic.saw_block_shells(slot, None)
            n_block_end = len(parts)
            # (no tip bridge: both kerfs stay open from the bone to the top along the whole cut; the end
            #  caps at the slot ends and the body keep the guide in one piece)
            for planned_pin in pins:
                parts.append(guide_logic.sleeve_shell(self._pin_obb_tree(mesh), planned_pin, gap))
            self._set_pin_axes(guide_logic.pin_axis_polydata(pins) if pins else None)
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
            closed = "all parts closed" if open_edges == 0 else f"NOT closed ({open_edges} open edges)"
            if islands:
                closed += f", {islands} small free island(s) removed"
            if self._guide_mode_is_reduction():
                self.guide_status.setText(
                    f"Reduction guide extruded on the corrected fragments: fit gap {gap:.2f} mm, "
                    f"wall {thickness:.1f} mm, {closed}." + (f"\n{pin_note}" if pin_note else ""))
            else:
                self.guide_status.setText(
                    f"Guide extruded: fit gap {gap:.2f} mm, wall {thickness:.1f} mm, {closed}.\n"
                    f"Saw slots {guide_logic.SLOT_WIDTH:.1f} mm (blade {guide_logic.BLADE_THICKNESS:.1f} mm), "
                    f"kerf inside the wedge: " + ("; ".join(slot_notes) if slot_notes else "none") +
                    f".\nPins: {pin}" + (f"\n{pin_note}" if pin_note else "") + (f"\n{label_note}" if label_note else ""))
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onExportGuideStl(self):
        """Binary STL of the guide in millimetres, centred at 0,0,0 (for printing only the shape matters). Body and saw blocks are closed shells in ONE file - print preparation
        software (PreForm, Chitubox, PrusaSlicer, ...) unites overlapping shells."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME)
        if node is None or node.GetPolyData() is None or node.GetPolyData().GetNumberOfPoints() == 0:
            self.guide_status.setText("No guide body yet - Extrude guide body first.")
            return
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText.lower()
        kind = "reduction_guide" if self._guide_mode_is_reduction() else "guide"
        number = self.patient_number.text.strip() if getattr(self, "patient_number", None) else ""
        default_name = (f"{number}_" if number else "") + f"Doggybone_{kind}_{side}_{bone}.stl"
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

    def _guide_nodes_of_mode(self, mode):
        saved = getattr(self, "_guide_mode", "saw")
        self._guide_mode = mode
        try:
            return [self.GUIDE_FOOTPRINT_NAME, self.GUIDE_BODY_NAME, self.GUIDE_RELIEF_NAME,
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
                    display.SetVisibility(not corrected)
        self._set_sphere_edit_visibility(not corrected)

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

    def _label_mode(self):
        box = getattr(self, "guide_label_mode", None)
        return box.currentText if box is not None else "Break-off tag"

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

    PAINT_TOOLS = ("brush_add", "brush_remove", "lasso_add", "lasso_remove", "brush_smooth")

    def _place_diameter_field(self):
        """The one pin-diameter field: under 'Prox. / Dist.' in reduction mode, top left of the guide
        parameters in saw mode."""
        pair = getattr(self, "_guide_field_labels", {}).get("Pin size")
        grid = self.reduction_options_widget.layout() if getattr(self, "reduction_options_widget", None) else None
        if pair is None or grid is None or getattr(self, "_reduction_diam_label", None) is None:
            return
        param_label, field = pair
        if self._guide_mode_is_reduction():
            grid.addWidget(field, 4, 1)
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
        if getattr(self, "guide_delete_button", None) is not None:
            # pin bar: 'Delete pins' next to 'Place pins' is the one delete button
            self.guide_delete_button.setVisible(full)
        reduction = self._guide_mode_is_reduction()
        if "pins" in buttons:
            buttons["pins"].setVisible(reduction)             # pins are set after the reduction only
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

    def onGuideModeChanged(self):
        mode = "reduction" if self.guide_mode_reduction.checked else "saw"
        if mode == getattr(self, "_guide_mode", "saw"):
            return
        self._stop_guide_edit()
        self._restore_guide_hidden()
        stacks = getattr(self, "_guide_undo_stacks", {})
        stacks[getattr(self, "_guide_mode", "saw")] = getattr(self, "_guide_undo_stack", [])
        self._guide_undo_stacks = stacks
        self._guide_mode = mode
        self._guide_undo_stack = stacks.get(mode, [])
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
        if mode == "saw":
            self.guide_status.setText("Saw guide: paint / edit the cutting guide on the bone as it is.")
            return
        if self._ensure_current_reduction_surface():
            return
        problems = []
        if len({p["segment"] for p in self._load_planned_pins()}) < 2:
            problems.append("no pins in both fragments yet (set Pins on the saw guide and Extrude it)")
        if combined is None:
            problems.append("no corrected bone (run the correction preview)")
        self.guide_status.setText(
            "Reduction guide: paint a SMALL footprint on the corrected bone around the pins (Brush + / "
            "Lasso +), then 'Extrude guide body' - sleeves come at the saw guide's pins in the corrected "
            "position." + (("\nMissing: " + "; ".join(problems)) if problems else ""))

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

    def _plan_reduction_pins(self, mesh, gap):
        """Sleeves of the reduction guide: the saw guide's pins in the corrected position."""
        reduced, moved, error = self._reduction_guide_pins()
        if error:
            return [], error
        if not reduced:
            return [], "No pins yet - switch Pins on and click on the corrected bone."
        obb = self._pin_obb_tree(mesh)
        planned, rejected = [], []
        for pin in reduced:
            q = guide_logic.plan_pin(obb, pin["entry"] + 2.0 * pin["axis"], pin["axis"], pin["diameter"],
                                     gap, guide_logic.BLOCK_HEIGHT)
            if "error" in q:
                rejected.append(f"{pin['label']}: {q['error']} (outside the prepared surface)")
                continue
            q["label"], q["segment"] = pin["label"], pin["segment"]
            planned.append(q)
        axes = [p["axis"] for p in planned]
        spread = max((float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))) for a in axes for b in axes),
                     default=0.0)
        per_segment = {}
        for p in planned:
            per_segment[p["segment"]] = per_segment.get(p["segment"], 0) + 1
        note = (f"Sleeves at the saw-guide pins (corrected position): "
                + ", ".join(f"{v} {k}" for k, v in per_segment.items())
                + f", \u00d8 {planned[0]['diameter']:.1f} mm, axes parallel within {spread:.1f}\u00b0" if planned
                else "No sleeves.")
        if any(v < 2 for v in per_segment.values()) or len(per_segment) < 2:
            note += "\nWARNING: each fragment needs 2 pins, otherwise it can rotate about its pin."
        if spread > 1.0:
            note += "\nWARNING: pins not parallel after reduction - re-extrude the saw guide."
        if rejected:
            note += "\nNot built: " + "; ".join(rejected)
        return planned, note

    def _reduction_is_bar(self):
        box = getattr(self, "reduction_type", None)
        return box is not None and box.currentIndex == 0

    def _build_reduction_bar(self):
        """Reduction guide as a bar over the saw guide's pins in the corrected position."""
        from vtk.util.numpy_support import vtk_to_numpy
        reduced, moved, error = self._reduction_guide_pins()
        if error:
            self.guide_status.setText(error)
            return
        if len({p["segment"] for p in reduced}) < 2:
            self.guide_status.setText("The reduction guide needs pins in BOTH fragments - switch Pins on and "
                                      "click on the corrected bone.")
            return
        reference = guide_logic.unit(reduced[0]["axis"])
        axis = guide_logic.unit(np.sum([guide_logic.unit(p["axis"]) * (1.0 if float(np.dot(p["axis"], reference)) >= 0
                                                                           else -1.0) for p in reduced], axis=0))
        wall = float(getattr(guide_logic, "SLEEVE_WALL", 2.0))
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            if self._guide_mesh_node() is not None:
                self._guide_push_undo()
            side = self.osteotomy_side.currentText.lower()
            bone = self.osteotomy_bone.currentText
            combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
            bone_obb = None
            if combined is not None and combined.GetPolyData() is not None and combined.GetPolyData().GetNumberOfPoints():
                bone_obb = vtk.vtkOBBTree()
                bone_obb.SetDataSet(combined.GetPolyData())
                bone_obb.BuildLocator()
            if bone_obb is not None:
                # bosses AND connecting webs over the full height, seated on the corrected bone
                body, info = guide_logic.reduction_web(
                    reduced, axis, bone_obb, height=float(self.reduction_standoff.value),
                    web_width=float(self.reduction_bar_diameter.value), wall=wall,
                    gap=float(self.guide_gap.value),
                    ring=bool(getattr(self, "reduction_ring_checkbox", None) and self.reduction_ring_checkbox.isChecked()),
                    filled=bool(getattr(self, "reduction_filled_checkbox", None) and self.reduction_filled_checkbox.isChecked()),
                    contact_only=bool(getattr(self, "reduction_contact_only_checkbox", None)
                                      and self.reduction_contact_only_checkbox.isChecked()),
                    clearance=float(self.reduction_clearance.value) if getattr(self, "reduction_clearance", None) else 6.0)
            else:
                body, info = guide_logic.reduction_bar(
                    reduced, axis, standoff=float(self.reduction_standoff.value),
                    bar_diameter=float(self.reduction_bar_diameter.value),
                    boss_height=float(self.reduction_boss_height.value), wall=wall)
            label_note = ""
            if self._label_wanted():
                if self._label_mode() == "Break-off tag":
                    label, label_note = self._label_tag(body, axis, bone_obb)
                else:
                    label, label_note = self._reduction_guide_label(body, axis)
                    if label is None:          # no flat spot on the guide: the break-off tag instead
                        label, tag_note = self._label_tag(body, axis, bone_obb)
                        label_note = ("No flat area for the label on the guide - break-off tag used instead. "
                                      + tag_note) if label is not None else label_note
                if label is not None:
                    append = vtk.vtkAppendPolyData()
                    append.AddInputData(body)
                    append.AddInputData(label)
                    append.Update()
                    body = vtk.vtkPolyData()
                    body.DeepCopy(append.GetOutput())
            self._set_guide_body(body)
            shown = []
            for pin in reduced:
                q = dict(pin)
                h = float(np.dot(pin["entry"], axis))
                q["top"] = pin["entry"] + (info["top"] - h + 3.0) / max(float(np.dot(pin["axis"], axis)), 1e-6) * pin["axis"]
                shown.append(q)
            self._set_pin_axes(guide_logic.pin_axis_polydata(shown))
            footprint = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_FOOTPRINT_NAME)
            if footprint is not None and footprint.GetDisplayNode() is not None:
                footprint.GetDisplayNode().SetVisibility(False)    # the bar does not use the footprint
            clearance = None
            if "guide_height_range" in info:
                combined = None                         # seated web: nothing hovers over the bone
            side = self.osteotomy_side.currentText.lower()
            bone = self.osteotomy_bone.currentText
            if "guide_height_range" not in info:
                combined = slicer.mrmlScene.GetFirstNodeByName(self._preview_node_names(side, bone)[3])
            if combined is not None and combined.GetPolyData() is not None and combined.GetPolyData().GetNumberOfPoints():
                local = vtk_to_numpy(combined.GetPolyData().GetPoints().GetData()).astype(float) @ info["frame"].T
                near = np.zeros(len(local), dtype=bool)
                nodes = info["nodes"]                 # rods only - the bosses sit on the bone on purpose
                for i, j in info["edges"]:
                    A, B = nodes[i], nodes[j]
                    AB = B - A
                    t = np.clip(((local[:, 0] - A[0]) * AB[0] + (local[:, 1] - A[1]) * AB[1]) / max(float(AB @ AB), 1e-12), 0, 1)
                    near |= np.hypot(local[:, 0] - (A[0] + t * AB[0]), local[:, 1] - (A[1] + t * AB[1])) <= info["r_bar"]
                for n in nodes:                        # not under a boss
                    near &= np.hypot(local[:, 0] - n[0], local[:, 1] - n[1]) > info["r_boss"]
                if near.any():
                    clearance = float(info["zc"] - info["r_bar"] - local[near, 2].max())
        finally:
            qt.QApplication.restoreOverrideCursor()
        per_segment = {}
        for p in reduced:
            per_segment[p["segment"]] = per_segment.get(p["segment"], 0) + 1
        lines = [f"Reduction bar over {len(reduced)} pins (" + ", ".join(f"{v} {k}" for k, v in per_segment.items())
                 + f"), bores \u00d8 {2 * info['r_in']:.2f} mm, pin axes parallel within "
                 f"{info['max_axis_deviation_deg']:.1f}\N{DEGREE SIGN}; "
                 + (f"bosses and webs connected over the full height, seated on the corrected bone (fit gap "
                    f"{self.guide_gap.value:.2f} mm), {info['guide_height_range'][0]:.0f}-"
                    f"{info['guide_height_range'][1]:.0f} mm high." if "guide_height_range" in info
                    else "no bone contact."),
                 f"Pins must stand out at least {max(info['protrusion']):.0f} mm above the bone."]
        if clearance is not None:
            lines.append(("WARNING: the bar reaches into the corrected bone - increase Height. " if clearance < 1.0
                          else "") + f"Clearance bar - bone {clearance:.1f} mm.")
        if any(v < 2 for v in per_segment.values()):
            lines.append("WARNING: each fragment needs 2 pins, otherwise it can rotate about its pin.")
        if info["max_axis_deviation_deg"] > 1.0:
            lines.append("WARNING: pins not parallel after reduction - re-extrude the saw guide first.")
        if getattr(self, "reduction_ring_checkbox", None) is not None and self.reduction_ring_checkbox.isChecked() \
                and "guide_height_range" in info:
            if info.get("contact_only"):
                lines.append(f"Bone contact only through the cylinders around the pins; webs / frame "
                             f"{info['web_clearance']:.1f} mm above the bone (room for the plate).")
            lines.append(("Filled frame over all pins." if info.get("filled") else
                          "Ring: closed frame around all pins (two bridges across the osteotomy).") if info.get("ring")
                         else "Ring not possible - the pins lie (nearly) in one line; built as a single chain. "
                              "Offset some pins sideways for a ring.")
        if label_note:
            lines.append(label_note)
        self.guide_status.setText("\n".join(lines))

    PLANNED_PINS_ATTRIBUTE = "Doggybone.PlannedPins"

    def _store_planned_pins(self, pins):
        """Keep the planned pins in the scene (pins node attribute) - survives a module Reload."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.SAW_PINS_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.SAW_PINS_NAME)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
        keys = ("entry", "far", "top", "axis", "point")
        data = [dict({k: [float(x) for x in pin[k]] for k in keys},
                     r_in=float(pin["r_in"]), r_out=float(pin["r_out"]), diameter=float(pin["diameter"]),
                     label=str(pin.get("label", "")), segment=str(pin.get("segment", "")))
                for pin in pins]
        node.SetAttribute(self.PLANNED_PINS_ATTRIBUTE, json.dumps(data))

    def _load_planned_pins(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.SAW_PINS_NAME)
        text = node.GetAttribute(self.PLANNED_PINS_ATTRIBUTE) if node is not None else None
        if not text:
            return []
        pins = json.loads(text)
        for pin in pins:
            for k in ("entry", "far", "top", "axis", "point"):
                pin[k] = np.asarray(pin[k], dtype=float)
        return pins

    def _reduced_pins(self, pins):
        """Pins in the CORRECTED position: pins of the fragment that carries the correction
        transform are moved with it. Returns (pins, moved_segment, error)."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        transform = slicer.mrmlScene.GetFirstNodeByName(names[2])
        if transform is None:
            return [], None, "No correction transform - run the correction preview first."
        moved = None
        for segment, name in (("proximal", names[0]), ("distal", names[1])):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetParentTransformNode() is not None \
                    and node.GetParentTransformNode().GetID() == transform.GetID():
                moved = segment
        if moved is None:
            return [], None, "Cannot tell which fragment carries the correction transform."
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)
        M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        R, t = M[:3, :3], M[:3, 3]
        out = []
        for pin in pins:
            q = dict(pin)
            if pin["segment"] == moved:
                for k in ("entry", "far", "top", "point"):
                    q[k] = R @ pin[k] + t
                q["axis"] = guide_logic.unit(R @ pin["axis"])
            out.append(q)
        return out, moved, None

    SAW_PINS_NAME = "Doggybone guide pins"
    # reduction mode: pins set on the corrected bone ("... (reduction)"), saw mode: legacy marks
    GUIDE_PINS_NAME = property(lambda self: self._mode_name("Doggybone guide pins"))
    GUIDE_PIN_AXES_NAME = property(lambda self: self._mode_name("Doggybone guide pin axes"))

    def _pin_obb_tree(self, mesh):
        key = (id(mesh), mesh.GetMTime())
        if getattr(self, "_pin_obb_key", None) != key:
            obb = vtk.vtkOBBTree()
            obb.SetDataSet(mesh)
            obb.BuildLocator()
            self._pin_obb, self._pin_obb_key = obb, key
        return self._pin_obb

    def _pin_diameter(self):
        chosen = self.guide_pin.currentText
        if chosen.startswith("Auto"):
            return guide_logic.auto_pin_diameter(getattr(self, "_last_bone_diameter", None))
        return float(chosen.split()[0])

    def _cut_planes(self):
        """[(label, origin, unit normal)] of the existing cutting planes."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        planes = []
        for label, name in (("proximal", self._plane_node_name(side, bone)),
                            ("distal", self._distal_plane_node_name(side, bone))):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                origin, normal = self._get_plane_origin_normal(node)
                planes.append((label, np.asarray(origin, float), guide_logic.unit(np.asarray(normal, float))))
        return planes

    def _pin_segment(self, point):
        """('proximal' | 'distal', None) or (None, reason) for a pin through point."""
        planes = self._cut_planes()
        if not planes:
            return None, "no cutting plane"
        p = np.asarray(point, float)
        # the BORE must stay outside the saw block (sleeve may merge into the block)
        margin = guide_logic.SLOT_WIDTH + guide_logic.SLOT_WALL + guide_logic.sleeve_radii(self._pin_diameter())[0] + 0.3
        if len(planes) == 2:
            for index, (label, o, n) in enumerate(planes):
                o2 = planes[1 - index][1]
                towards_wedge = np.sign(float((o2 - o) @ n)) or 1.0
                s_val = float((p - o) @ n) * towards_wedge
                if s_val < 0:                                    # kept side of this cut
                    if -s_val < margin:
                        return None, f"too close to the {label} cut ({-s_val:.1f} mm, needs {margin:.1f} mm)"
                    return label, None
            return None, "inside the wedge"
        label, o, n = planes[0]
        shaft, _, _, error = self._get_anatomical_frame(self.osteotomy_side.currentText.lower(),
                                                        self.osteotomy_bone.currentText)
        s_val = float((p - o) @ n)
        if abs(s_val) < margin:
            return None, f"too close to the cut ({abs(s_val):.1f} mm, needs {margin:.1f} mm)"
        proximal = float((p - o) @ np.asarray(shaft, float)) < 0 if not error else s_val < 0
        return ("proximal" if proximal else "distal"), None

    def _pin_feedback(self, message, point, ok):
        """Rejected click: red status text and a short red cross at the spot."""
        self.guide_status.setStyleSheet("" if ok else STATUS_ERROR_STYLE)
        if ok:
            return
        self.guide_status.setText(message)
        slicer.util.showStatusMessage(message, 4000)
        name = "Doggybone pin rejected"
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
            node.GetDisplayNode().SetSelectedColor(1.0, 0.1, 0.1)
            node.GetDisplayNode().SetColor(1.0, 0.1, 0.1)
            node.GetDisplayNode().SetGlyphTypeFromString("Cross2D")
            node.GetDisplayNode().SetGlyphScale(3.0)
            node.GetDisplayNode().SetPointLabelsVisibility(False)
        node.RemoveAllControlPoints()
        node.AddControlPointWorld(vtk.vtkVector3d(*[float(x) for x in point]))

        def clear(node_id=node.GetID()):
            existing = slicer.mrmlScene.GetNodeByID(node_id)
            if existing is not None:
                existing.RemoveAllControlPoints()
        qt.QTimer.singleShot(1500, clear)

    def onDeletePins(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        count = node.GetNumberOfControlPoints() if node is not None else 0
        if node is not None:
            node.RemoveAllControlPoints()
        self._pins_dirty = False
        self._set_pin_axes(None)
        if self._guide_mode_is_reduction():
            # the reduction guide is built on these pins - without pins there is no guide
            self._set_guide_body(None)
            handles = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
            if handles is not None:
                handles.RemoveAllControlPoints()
            self._pin_rows_active = False
            self._select_reduction_pin(None)
        elif count and slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_BODY_NAME) is not None:
            self._extrude_guide_body(push_undo=False)
        self.guide_status.setText(f"{count} pin(s) deleted." if count else "No pins to delete.")

    def _pin_under_mouse(self, x, y, tolerance=2.5):
        """Index of the pin mark closest to the view ray through (x, y), if within tolerance mm.
        Works although the mark sits inside the (hollow) sleeve bore after extrusion."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        if node is None or node.GetNumberOfControlPoints() == 0:
            return None
        renderer = self._guide_renderer()
        ends = []
        for z in (0.0, 1.0):
            renderer.SetDisplayPoint(float(x), float(y), z)
            renderer.DisplayToWorld()
            wp = renderer.GetWorldPoint()
            ends.append(np.array([wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]]))
        d = guide_logic.unit(ends[1] - ends[0])
        best, best_dist = None, tolerance
        for i in range(node.GetNumberOfControlPoints()):
            q = np.array(node.GetNthControlPointPositionWorld(i)) - ends[0]
            dist = float(np.linalg.norm(q - (q @ d) * d))
            if dist < best_dist:
                best, best_dist = i, dist
        return best

    def _remove_pin(self, index):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        if node is None or index >= node.GetNumberOfControlPoints():
            return
        label = node.GetNthControlPointLabel(index)
        node.RemoveNthControlPoint(index)
        self._relabel_pins(node)
        self._pins_dirty = True
        self.guide_status.setStyleSheet("")
        self.guide_status.setText(f"Pin {label} removed ({node.GetNumberOfControlPoints()} marked). "
                                  "Switch Pins off to update the sleeves.")
        if self._guide_mode_is_reduction():
            self._show_reduction_pin_axes()
            self._place_pin_gizmo()

    # --- automatic pin rows (reduction mode) -------------------------------------------------
    PIN_ROW_HANDLES = "Doggybone pin row handles"

    def _pin_row_defaults(self):
        """(first-pin distance from the cut, spacing) from the pin diameter."""
        diameter = self._pin_diameter()
        r_in, r_out = guide_logic.sleeve_radii(diameter)
        margin = guide_logic.SLOT_WIDTH + guide_logic.SLOT_WALL + r_in + 0.3      # as _pin_segment checks
        first = np.ceil((margin + 1.0) * 2.0) / 2.0
        spacing = np.ceil(max(2.0 * r_out + 2.0, 3.0 * diameter) * 2.0) / 2.0
        return float(first), float(spacing)

    def _fragment_cut_frames(self, M, moved, fragments):
        """{segment: (cut origin, unit direction along the shaft INTO the fragment)} in the corrected pose.
        CWO: each fragment's own cut plane; OWO: the single (master) cut."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        planes = {"proximal": slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone)),
                  "distal": slicer.mrmlScene.GetFirstNodeByName(self._distal_plane_node_name(side, bone))}
        master = self.reference_plane_selector.currentText.lower()
        is_cwo = self.osteotomy_technique.currentText.startswith("CWO")
        _direction, shafts, _error = self._reduction_pin_frame(M, moved)
        frames = {}
        for segment, (polydata, _obb, _loc) in fragments.items():
            node = planes[segment] if is_cwo else planes.get(master)
            if node is None or shafts is None:
                continue
            origin, _normal = self._get_plane_origin_normal(node)
            origin = np.asarray(origin, float)
            if segment == moved:
                origin = (M @ np.append(origin, 1.0))[:3]
            from vtk.util.numpy_support import vtk_to_numpy
            centroid = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float).mean(axis=0)
            s_dir = shafts[segment]
            if float(np.dot(centroid - origin, s_dir)) < 0.0:
                s_dir = -s_dir
            frames[segment] = (origin, s_dir)
        return frames

    @staticmethod
    def _pin_section_normal(shaft, direction):
        """Normal of the cross-section used to centre a pin: the shaft direction with its component
        along the pin removed - the section then contains the whole pin line."""
        shaft = np.asarray(shaft, float)
        direction = guide_logic.unit(np.asarray(direction, float))
        normal = shaft - float(np.dot(shaft, direction)) * direction
        return guide_logic.unit(normal) if np.linalg.norm(normal) > 1e-6 else guide_logic.unit(shaft)

    def _row_pin(self, segment, level_point, s_dir, fragments, direction):
        polydata, obb, _loc = fragments[segment]
        centre = self._section_centroid(polydata, level_point, self._pin_section_normal(s_dir, direction),
                                        largest=True)
        if centre is None:
            return {"error": "no bone at that level"}
        pin = guide_logic.plan_pin(obb, centre, direction, self._pin_diameter(), float(self.guide_gap.value),
                                   guide_logic.BLOCK_HEIGHT)
        if "error" not in pin:
            pin["segment"], pin["centre"] = segment, centre
        return pin

    def _pin_distribution_manual(self):
        box = getattr(self, "pin_row_pattern", None)
        return box is not None and box.currentText == "Manual"

    def _pin_distribution_changed(self):
        if self._pin_distribution_manual():
            self._pin_rows_active = False          # nothing is generated automatically any more
            handles = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
            if handles is not None:                # 'P row' / 'D row' only move automatic rows
                handles.RemoveAllControlPoints()
            return
        self._pin_rows_changed()

    def onPlacePinRows(self):
        if not self._guide_mode_is_reduction():
            self.guide_status.setText("Pin rows are placed in the 'Reduction guide' mode.")
            return
        if self._pin_distribution_manual():
            button = getattr(self, "_guide_tool_buttons", {}).get("pins")
            if button is not None and not button.checked:
                button.setChecked(True)
            self.guide_status.setStyleSheet("")
            self.guide_status.setText("Manual pins: click on the corrected bone to set a pin (it gets the gizmo), "
                                      "click a pin to delete it, Cmd/Ctrl + click a pin to move it.")
            return
        self._stop_guide_edit()
        self._pin_rows_active = True
        self._place_pin_rows()

    def _pin_rows_changed(self):
        if self._pin_distribution_manual():
            return
        if getattr(self, "_pin_rows_active", False) and self._guide_mode_is_reduction():
            self._place_pin_rows()

    def _place_pin_rows(self):
        M, moved = self._correction_state()
        if M is None:
            self.guide_status.setText("No correction preview - show the correction preview first.")
            return
        direction, _shafts, error = self._reduction_pin_frame(M, moved)
        if error:
            self.guide_status.setText(error)
            return
        fragments = self._corrected_fragments()
        frames = self._fragment_cut_frames(M, moved, fragments)
        auto_first, auto_spacing = self._pin_row_defaults()
        spacing = float(self.pin_row_spacing.value) or auto_spacing
        count = int(self.pin_row_count.value)
        distances = {"proximal": float(self.pin_row_distance_proximal.value) or auto_first,
                     "distal": float(self.pin_row_distance_distal.value) or auto_first}
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.REDUCTION_PIN_MARKS)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
            node.GetDisplayNode().SetSelectedColor(0.95, 0.95, 0.3)
            node.GetDisplayNode().SetGlyphScale(1.5)
            node.GetDisplayNode().SetTextScale(2.5)
        kept = []                            # pins added or moved by hand survive a new row layout
        for i in range(node.GetNumberOfControlPoints()):
            if self._pin_kind(node, i)[1] not in ("auto", "autofree"):
                kept.append((list(node.GetNthControlPointPositionWorld(i)),
                             node.GetNthControlPointDescription(i), node.GetNthControlPointID(i)))
        selected_id = getattr(self, "_selected_pin_id", None)
        node.RemoveAllControlPoints()
        handles_at, problems, placed, shifted, replaced, candidates = {}, [], {}, {}, {}, []
        for segment in ("proximal", "distal"):
            if segment not in frames:
                problems.append(f"{segment}: no cut plane / fragment")
                continue
            origin, s_dir = frames[segment]
            shift = 0.0                      # extra distance so the saw guide accepts the pin (oblique cut)
            Minv = np.linalg.inv(M)
            for k in range(count):
                pin = None
                for _attempt in range(6):
                    level = origin + (distances[segment] + shift + k * spacing) * s_dir
                    pin = self._row_pin(segment, level, s_dir, fragments, direction)
                    if "error" in pin:
                        break
                    raw_entry = pin["entry"] if segment != moved else (Minv @ np.append(pin["entry"], 1.0))[:3]
                    ok_segment, reason = self._pin_segment(raw_entry)
                    found = re.search(r"\(([\d.]+) mm, needs ([\d.]+) mm\)", reason or "")
                    if ok_segment is not None or not found:
                        break
                    shift += float(found.group(2)) - float(found.group(1)) + 0.3
                if "error" in pin:
                    problems.append(f"{segment} pin {k + 1}: {pin['error']} - the row does not fit, "
                                    "reduce Pins / Spacing / distance")
                    break
                if k == 0:
                    handles_at[segment] = pin["entry"] + 4.0 * np.asarray(direction, float)
                    shifted[segment] = shift
                kind = segment
                if self._pin_row_staggered():
                    sign = (1.0 if k % 2 == 0 else -1.0) * (1.0 if segment == "proximal" else -1.0)
                    offset_pin = self._staggered_pin(pin, sign, s_dir, direction, fragments, segment, moved, Minv)
                    if offset_pin is not None:
                        pin, kind = offset_pin, f"{segment}|autofree"
                candidates.append((kind, np.asarray(pin["entry"], float)))
        # a pin added / moved by hand takes the place of the ONE automatic pin nearest to it
        a_dir = np.asarray(direction, float)

        def lateral(p, q):
            offset = np.asarray(p, float) - np.asarray(q, float)
            return float(np.linalg.norm(offset - float(np.dot(offset, a_dir)) * a_dir))
        claimed = set()
        for position, description, _old_id in kept:
            options = [(lateral(position, entry), c) for c, (kind, entry) in enumerate(candidates)
                       if kind.split("|")[0] == description.split("|")[0] and c not in claimed]
            options = [o for o in options if o[0] < 0.9 * spacing]
            if options:
                claimed.add(min(options)[1])
        for c, (kind, entry) in enumerate(candidates):
            segment = kind.split("|")[0]
            if c in claimed:
                replaced[segment] = replaced.get(segment, 0) + 1
                continue
            index = node.AddControlPointWorld(vtk.vtkVector3d(*entry))
            node.SetNthControlPointDescription(index, kind)
            placed[segment] = placed.get(segment, 0) + 1
        reselect = None
        for position, description, old_id in kept:
            index = node.AddControlPointWorld(vtk.vtkVector3d(*position))
            node.SetNthControlPointDescription(index, description)
            if old_id == selected_id:
                reselect = index
        self._relabel_pins(node)
        close = []                           # sleeves that would touch: pins of one fragment too close
        for i in range(node.GetNumberOfControlPoints()):
            for j in range(i + 1, node.GetNumberOfControlPoints()):
                if self._pin_kind(node, i)[0] != self._pin_kind(node, j)[0]:
                    continue
                gap_ij = lateral(node.GetNthControlPointPositionWorld(i), node.GetNthControlPointPositionWorld(j))
                if gap_ij < 0.9 * spacing:
                    close.append(f"{node.GetNthControlPointLabel(i)}-{node.GetNthControlPointLabel(j)} "
                                 f"{gap_ij:.1f} mm")
        self._update_pin_row_handles(handles_at, frames)
        self._show_reduction_pin_axes()
        self._select_reduction_pin(reselect)
        self._pins_dirty = True
        text = (f"Pin rows: " + ", ".join(f"{v} {k}" for k, v in placed.items()) +
                (f" (+{len(kept)} added by hand" + (", replacing " + ", ".join(f"{v} {k}" for k, v in replaced.items())
                                                     if replaced else "") + ")" if kept else "") +
                f"; first pin {distances['proximal']:.1f} mm (proximal) / {distances['distal']:.1f} mm (distal) "
                f"from the cut, spacing {spacing:.1f} mm - "
                + ("staggered to both sides of the bone centre" if self._pin_row_staggered() else "centred")
                + ", parallel to each other and to the cut plane(s). Drag the row handles along the bone to "
                "adjust, then 'Extrude guide body'.")
        moved_on = [f"{k} +{v:.1f} mm" for k, v in shifted.items() if v > 0.0]
        if moved_on:
            text += ("\nMoved further from the (oblique) cut so the saw-guide sleeves keep their distance: "
                     + ", ".join(moved_on) + ".")
        if problems:
            text += "\n" + "; ".join(problems)
        if close:
            text += (f"\nPins too close (< {0.9 * spacing:.1f} mm, sleeves touch): " + ", ".join(close)
                     + " - move one with the gizmo.")
        self.guide_status.setStyleSheet(STATUS_WARN_STYLE if close else "")
        self.guide_status.setText(text)

    def _pin_row_staggered(self):
        box = getattr(self, "pin_row_pattern", None)
        return box is not None and box.currentText == "Staggered"

    def _staggered_pin(self, pin, sign, s_dir, direction, fragments, segment, moved, Minv, max_offset=4.0):
        """The centred row pin shifted sideways (across the bone, perpendicular to shaft and pin) by as much
        as the bone allows: the pin must keep >= 80 % of its centred bone length and >= 1.5 mm bone around
        it. None if no offset of at least 1 mm fits (the pin then stays centred)."""
        z = guide_logic.unit(np.asarray(direction, float))
        across = np.cross(z, np.asarray(s_dir, float))
        if np.linalg.norm(across) < 1e-6:
            return None
        across = guide_logic.unit(across)
        _poly, obb, _loc = fragments[segment]
        centre = np.asarray(pin.get("centre", pin["entry"]), float)
        r_pin = 0.5 * float(pin.get("diameter", self._pin_diameter()))
        for offset in np.arange(max_offset, 0.99, -0.5):
            shifted = guide_logic.plan_pin(obb, centre + sign * offset * across, z, self._pin_diameter(),
                                           float(self.guide_gap.value), guide_logic.BLOCK_HEIGHT)
            if "error" in shifted or shifted["bone_length"] < 0.8 * float(pin["bone_length"]):
                continue
            # bone left beside the pin: the line r_pin + 1.5 mm further out must still hit the bone
            beside = guide_logic.plan_pin(obb, centre + sign * (offset + r_pin + 1.5) * across, z,
                                          self._pin_diameter(), float(self.guide_gap.value), guide_logic.BLOCK_HEIGHT)
            if "error" in beside or beside["bone_length"] < 0.5 * float(pin["bone_length"]):
                continue
            raw_entry = shifted["entry"] if segment != moved else (Minv @ np.append(shifted["entry"], 1.0))[:3]
            if self._pin_segment(raw_entry)[0] is None:
                continue
            shifted["segment"], shifted["centre"] = segment, centre + sign * offset * across
            return shifted
        return None

    def _update_pin_row_handles(self, handles_at, frames):
        node = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.PIN_ROW_HANDLES)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetSelectedColor(0.2, 0.8, 1.0)
            node.GetDisplayNode().SetGlyphScale(3.0)
            node.GetDisplayNode().SetTextScale(2.5)
            self._pin_row_handle_tag = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                                                        lambda caller, event: self._on_pin_row_handle_moved())
        elif getattr(self, "_pin_row_handle_tag", None) is None:
            self._pin_row_handle_tag = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                                                        lambda caller, event: self._on_pin_row_handle_moved())
        self._pin_row_frames = frames
        self._updating_handles = True
        try:
            node.RemoveAllControlPoints()
            for segment, position in handles_at.items():
                index = node.AddControlPointWorld(vtk.vtkVector3d(*position))
                node.SetNthControlPointLabel(index, f"{segment[0].upper()} row")
                node.SetNthControlPointDescription(index, segment)
            node.SetLocked(False)
            if node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(True)
        finally:
            self._updating_handles = False

    def _on_pin_row_handle_moved(self):
        if getattr(self, "_updating_handles", False):
            return
        node = slicer.mrmlScene.GetFirstNodeByName(self.PIN_ROW_HANDLES)
        frames = getattr(self, "_pin_row_frames", {})
        if node is None:
            return
        boxes = {"proximal": self.pin_row_distance_proximal, "distal": self.pin_row_distance_distal}
        changed = False
        for i in range(node.GetNumberOfControlPoints()):
            segment = node.GetNthControlPointDescription(i)
            if segment not in frames:
                continue
            origin, s_dir = frames[segment]
            distance = float(np.dot(np.array(node.GetNthControlPointPositionWorld(i)) - origin, s_dir))
            distance = max(0.5, round(distance * 2.0) / 2.0)
            box = boxes[segment]
            if abs(box.value - distance) > 1e-6:
                was = box.blockSignals(True)
                box.setValue(distance)
                box.blockSignals(was)
                changed = True
        if changed:
            qt.QTimer.singleShot(0, self._place_pin_rows)

    # --- pins set AFTER the reduction (on the corrected bone) --------------------------------
    REDUCTION_PIN_MARKS = "Doggybone guide pins (reduction)"

    def _correction_state(self):
        """(4x4 correction incl. fine shift, moved segment) or (None, None)."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        transform = slicer.mrmlScene.GetFirstNodeByName(names[2])
        if transform is None:
            return None, None
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToWorld(matrix)
        M = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        moved = None
        for segment, name in (("proximal", names[0]), ("distal", names[1])):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None and node.GetParentTransformNode() is not None \
                    and node.GetParentTransformNode().GetID() == transform.GetID():
                moved = segment
        return M, moved

    def _corrected_fragments(self):
        """{segment: (polydata in the corrected pose, OBB tree, cell locator)} - cached while neither the
        fragments nor the correction (incl. fine shift) changed."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        names = self._preview_node_names(side, bone)
        key = []
        for name in names[:2]:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None or node.GetPolyData() is None:
                key.append(None)
                continue
            matrix = vtk.vtkMatrix4x4()
            if node.GetParentTransformNode() is not None:
                node.GetParentTransformNode().GetMatrixTransformToWorld(matrix)
            key.append((node.GetID(), node.GetPolyData().GetMTime(),
                        tuple(round(matrix.GetElement(r, c), 9) for r in range(4) for c in range(4))))
        key = tuple(key)
        cached = getattr(self, "_corrected_fragments_cache", None)
        if cached is not None and cached[0] == key:
            return dict(cached[1])
        out = self._corrected_fragments_uncached(names)
        self._corrected_fragments_cache = (key, out)
        return dict(out)

    def _corrected_fragments_uncached(self, names):
        out = {}
        for segment, name in (("proximal", names[0]), ("distal", names[1])):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None or node.GetPolyData() is None or node.GetPolyData().GetNumberOfPoints() == 0:
                continue
            polydata = node.GetPolyData()
            parent = node.GetParentTransformNode()
            if parent is not None:
                matrix = vtk.vtkMatrix4x4()
                parent.GetMatrixTransformToWorld(matrix)
                transform = vtk.vtkTransform()
                transform.SetMatrix(matrix)
                moved = vtk.vtkTransformPolyDataFilter()
                moved.SetInputData(polydata)
                moved.SetTransform(transform)
                moved.Update()
                polydata = vtk.vtkPolyData()
                polydata.DeepCopy(moved.GetOutput())
            obb = vtk.vtkOBBTree()
            obb.SetDataSet(polydata)
            obb.BuildLocator()
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(polydata)
            locator.BuildLocator()
            out[segment] = (polydata, obb, locator)
        return out

    def _reduction_pin_frame(self, M, moved):
        """(common pin direction, {segment: shaft axis}, error) in the corrected pose.
        The pins lie PARALLEL TO THE CUT PLANE(S), so no pin can cross a cut anywhere along its length:
        CWO (faces in contact): in the contact plane and perpendicular to the condylar axis;
        OWO (faces apart): along the line both cut planes share. All pins parallel, towards the guide side."""
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        d, shaft, _across, error = self._guide_direction_frame(side, bone)
        if error:
            return None, None, error
        R = M[:3, :3] if M is not None else np.eye(3)
        condylar = "distal" if bone == "Femur" else "proximal"
        guide_side = guide_logic.unit(R @ d if moved == condylar else np.asarray(d, float))
        shafts = {segment: guide_logic.unit(R @ shaft if segment == moved else np.asarray(shaft, float))
                  for segment in ("proximal", "distal")}
        # the fragments' cut planes in the corrected pose
        planes = {"proximal": slicer.mrmlScene.GetFirstNodeByName(self._plane_node_name(side, bone)),
                  "distal": slicer.mrmlScene.GetFirstNodeByName(self._distal_plane_node_name(side, bone))}
        master = self.reference_plane_selector.currentText.lower()
        is_cwo = self.osteotomy_technique.currentText.startswith("CWO")
        normals = []
        for segment in ("proximal", "distal"):
            node = planes[segment] if is_cwo else planes.get(master)
            if node is None:
                continue
            _origin, normal = self._get_plane_origin_normal(node)
            normal = guide_logic.unit(np.asarray(normal, float))
            normals.append(guide_logic.unit(R @ normal) if segment == moved else normal)
        if not normals:
            return None, None, "Insert the cutting plane(s) first."
        ml_vector = self._get_ml_vector(side, bone)
        ml = None
        if ml_vector is not None and np.linalg.norm(ml_vector) > 1e-6:
            ml = guide_logic.unit(np.asarray(ml_vector, float))
            ml = guide_logic.unit(R @ ml) if moved == condylar else ml
        n0 = normals[0]
        n1 = normals[-1]
        common = np.cross(n0, n1)
        if np.linalg.norm(common) > np.sin(np.radians(1.0)):
            direction = guide_logic.unit(common)              # OWO: line shared by both cut planes
        elif ml is not None and np.linalg.norm(np.cross(n0, ml)) > 1e-6:
            direction = guide_logic.unit(np.cross(n0, ml))    # CWO: in the contact plane, perpendicular to condyles
        else:
            direction = guide_side - float(np.dot(guide_side, n0)) * n0
            if np.linalg.norm(direction) < 1e-6:
                return None, None, "Guide side is perpendicular to the cut - cannot orient the pins."
            direction = guide_logic.unit(direction)
        if float(np.dot(direction, guide_side)) < 0.0:
            direction = -direction
        return direction, shafts, None

    @staticmethod
    def _section_centroid(polydata, point, normal, largest=False):
        """Area centroid of the bone cross-section (outer contour piece nearest to point, or the
        largest loop of the whole section with largest=True)."""
        plane = vtk.vtkPlane()
        plane.SetOrigin(*[float(v) for v in point])
        plane.SetNormal(*[float(v) for v in normal])
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(polydata)
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputConnection(cutter.GetOutputPort())
        if largest:
            connectivity.SetExtractionModeToAllRegions()
        else:
            connectivity.SetExtractionModeToClosestPointRegion()
            connectivity.SetClosestPoint(*[float(v) for v in point])
        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(connectivity.GetOutputPort())
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        contour = stripper.GetOutput()
        if contour.GetNumberOfPoints() < 3:
            return None
        from vtk.util.numpy_support import vtk_to_numpy
        P = vtk_to_numpy(contour.GetPoints().GetData()).astype(float)
        n = guide_logic.unit(np.asarray(normal, float))
        e1, e2 = guide_logic._perp_frame(n)
        best = None
        lines = contour.GetLines()
        lines.InitTraversal()
        ids = vtk.vtkIdList()
        while lines.GetNextCell(ids):
            loop = np.array([P[ids.GetId(k)] for k in range(ids.GetNumberOfIds())])
            if len(loop) < 3:
                continue
            x, y = (loop - point) @ e1, (loop - point) @ e2
            area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            if abs(area) < 1e-9:
                continue
            cx = float(np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))) / (6.0 * area)
            cy = float(np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))) / (6.0 * area)
            if best is None or abs(area) > best[0]:
                best = (abs(area), np.asarray(point, float) + cx * e1 + cy * e2)
        return best[1] if best is not None else P.mean(axis=0)

    def _center_reduction_pin(self, point, fragments=None, frame=None):
        """Pin through the cross-section centre of the fragment at `point` (corrected pose),
        common direction. Returns pin dict (entry/far/top/axis/radii/segment) or dict(error=...)."""
        if fragments is None:
            fragments = self._corrected_fragments()
        if frame is None:
            M, moved = self._correction_state()
            if M is None:
                return {"error": "no correction preview"}
            direction, shafts, error = self._reduction_pin_frame(M, moved)
            if error:
                return {"error": error}
        else:
            direction, shafts = frame
        p = np.asarray(point, float)
        best = None
        for segment, (_poly, _obb, locator) in fragments.items():
            closest = [0.0, 0.0, 0.0]
            cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
            locator.FindClosestPoint(p.tolist(), closest, cell_id, sub_id, dist2)
            if best is None or float(dist2) < best[0]:
                best = (float(dist2), segment)
        if best is None:
            return {"error": "no corrected fragments"}
        segment = best[1]
        polydata, obb, _locator = fragments[segment]
        # section plane that CONTAINS the pin direction: re-centring from any point on the pin line
        # (e.g. its mark at the entry) gives the same pin, even with pins oblique to the shaft
        centre = self._section_centroid(polydata, p, self._pin_section_normal(shafts[segment], direction))
        if centre is None:
            return {"error": "no bone cross-section at that level"}
        pin = guide_logic.plan_pin(obb, centre, direction, self._pin_diameter(),
                                   float(self.guide_gap.value), guide_logic.BLOCK_HEIGHT)
        if "error" in pin:
            return pin
        pin["segment"] = segment
        pin["centre"] = centre
        return pin

    def _reduction_marked_pins(self):
        """Centred pins (corrected pose) for all marks set in reduction mode -> (pins, rejected)."""
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is None or node.GetNumberOfControlPoints() == 0:
            return [], []
        M, moved = self._correction_state()
        if M is None:
            return [], ["no correction preview"]
        direction, shafts, error = self._reduction_pin_frame(M, moved)
        if error:
            return [], [error]
        fragments = self._corrected_fragments()
        pins, rejected = [], []
        for i in range(node.GetNumberOfControlPoints()):
            label = node.GetNthControlPointLabel(i)
            point = np.array(node.GetNthControlPointPositionWorld(i))
            if self._pin_kind(node, i)[1] in ("free", "autofree"):
                pin = self._free_reduction_pin(point, fragments, direction)
            else:
                pin = self._center_reduction_pin(point, fragments, (direction, shafts))
            if "error" in pin:
                rejected.append(f"{label}: {pin['error']}")
                continue
            pin["label"] = label
            pins.append(pin)
        return pins, rejected

    def _reduction_guide_pins(self):
        """Pins for the reduction guide in the corrected pose: from the marks set after the reduction,
        else (older plans) the saw guide's pins moved into the corrected pose. -> (pins, moved, error)"""
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is not None:
            _M, moved = self._correction_state()
            if node.GetNumberOfControlPoints() == 0:
                return [], moved, "No pins - place the pin rows or set pins on the corrected bone."
            pins, rejected = self._reduction_marked_pins()
            if not pins:
                return [], moved, "No usable pins: " + "; ".join(rejected)
            return pins, moved, None
        # older plans without pins set after the reduction: the saw guide's pins moved into place
        return self._reduced_pins(self._load_planned_pins())

    # --- single pins: added by hand, moved with a small gizmo (always parallel) ------------------
    PIN_GIZMO = "Doggybone pin gizmo"
    PIN_GIZMO_LIFT = 10.0      # mm above the pin entry, on the pin axis - the pin itself stays clickable

    @staticmethod
    def _pin_kind(node, index):
        """(segment, kind) of a reduction pin mark; kind: 'auto' (pin row), 'manual' (added / moved by
        hand, centred in the bone) or 'free' (exactly where it was put)."""
        parts = (node.GetNthControlPointDescription(index) or "").split("|")
        return parts[0], (parts[1] if len(parts) > 1 and parts[1] else "auto")

    def _free_reduction_pin(self, point, fragments, direction):
        """Pin exactly through `point` (no centring), common direction, in the nearest fragment."""
        p = np.asarray(point, float)
        best = None
        for segment, (_poly, _obb, locator) in fragments.items():
            closest = [0.0, 0.0, 0.0]
            cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
            locator.FindClosestPoint(p.tolist(), closest, cell_id, sub_id, dist2)
            if best is None or float(dist2) < best[0]:
                best = (float(dist2), segment)
        if best is None:
            return {"error": "no corrected fragments"}
        segment = best[1]
        _poly, obb, _locator = fragments[segment]
        pin = guide_logic.plan_pin(obb, p, direction, self._pin_diameter(), float(self.guide_gap.value),
                                   guide_logic.BLOCK_HEIGHT)
        if "error" in pin:
            return pin
        pin["segment"], pin["centre"] = segment, p
        return pin

    def onDeleteSelectedPin(self):
        index = self._selected_pin_index()
        if index is None:
            self.guide_status.setStyleSheet("")
            self.guide_status.setText("No pin selected - switch 'Pins' on and click the pin you want to delete.")
            return
        self._remove_pin(index)
        self._select_reduction_pin(None)

    def _keep_clear_of_cut(self, pin, fragments, direction, centred=False):
        """A pin too close to its cut (the saw guide could not take it) is moved away from the cut along
        the bone - just far enough. -> (pin, shift in mm) or (pin with 'error', 0)."""
        M, moved = self._correction_state()
        if M is None or "error" in pin:
            return pin, 0.0
        frames = self._fragment_cut_frames(M, moved, fragments)
        segment = pin["segment"]
        if segment not in frames:
            return pin, 0.0
        _origin, s_dir = frames[segment]
        z = guide_logic.unit(np.asarray(direction, float))
        along = np.asarray(s_dir, float) - float(np.dot(s_dir, z)) * z      # along the bone, away from the cut
        if np.linalg.norm(along) < 1e-6:
            return pin, 0.0
        along = guide_logic.unit(along)
        Minv = np.linalg.inv(M)
        shift = 0.0
        for _attempt in range(6):
            raw = pin["entry"] if segment != moved else (Minv @ np.append(pin["entry"], 1.0))[:3]
            ok_segment, reason = self._pin_segment(raw)
            found = re.search(r"\(([\d.]+) mm, needs ([\d.]+) mm\)", reason or "")
            if ok_segment is not None or not found:
                return pin, shift
            step = float(found.group(2)) - float(found.group(1)) + 0.3
            shift += step
            point = np.asarray(pin.get("centre", pin["entry"]), float) + step * along
            new = (self._center_reduction_pin(point, fragments, self._reduction_pin_frame(M, moved)[:2]) if centred
                   else self._free_reduction_pin(point, fragments, z))
            if "error" in new:
                return pin, 0.0
            if not centred:
                new["centre"] = point
            pin = new
        return pin, shift

    def _selected_pin_index(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        pin_id = getattr(self, "_selected_pin_id", None)
        if node is None or pin_id is None:
            return None
        index = node.GetNthControlPointIndexByID(pin_id)
        return index if index is not None and index >= 0 else None

    def _select_reduction_pin(self, index, announce=False):
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is None or index is None or index < 0 or index >= node.GetNumberOfControlPoints():
            self._selected_pin_id = None
            self._remove_pin_gizmo()
            return
        self._selected_pin_id = node.GetNthControlPointID(index)
        self._place_pin_gizmo()
        if announce:
            self.guide_status.setStyleSheet("")
            self.guide_status.setText(
                f"Pin {node.GetNthControlPointLabel(index)} selected - move it with the gizmo above it (it "
                "stays parallel); a click on the pin deletes it.")

    def _pin_gizmo_axes(self, segment):
        M, moved = self._correction_state()
        if M is None:
            return None
        direction, shafts, error = self._reduction_pin_frame(M, moved)
        if error:
            return None
        z = guide_logic.unit(np.asarray(direction, float))
        s = np.asarray(shafts.get(segment, shafts["proximal"]), float)
        x = s - float(np.dot(s, z)) * z
        x = guide_logic.unit(x) if np.linalg.norm(x) > 1e-6 else guide_logic._perp_frame(z)[0]
        return x, np.cross(z, x), z

    def _remove_pin_gizmo(self):
        self._remove_pin_preview()
        gizmo = slicer.mrmlScene.GetFirstNodeByName(self.PIN_GIZMO)
        if gizmo is not None:
            slicer.mrmlScene.RemoveNode(gizmo)
        self._pin_gizmo_tags = None

    def _place_pin_gizmo(self):
        index = self._selected_pin_index()
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if index is None or node is None:
            self._remove_pin_gizmo()
            return
        segment, _kind = self._pin_kind(node, index)
        axes = self._pin_gizmo_axes(segment)
        if axes is None:
            self._remove_pin_gizmo()
            return
        gizmo = slicer.mrmlScene.GetFirstNodeByName(self.PIN_GIZMO)
        if gizmo is None:
            gizmo = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", self.PIN_GIZMO)
            gizmo.CreateDefaultDisplayNodes()
            display = gizmo.GetDisplayNode()
            display.SetSelectedColor(1.0, 0.55, 0.1)
            display.SetColor(1.0, 0.55, 0.1)
            display.SetFillOpacity(0.25)
            display.SetPropertiesLabelVisibility(False)
            display.SetPointLabelsVisibility(False)
            display.SetHandlesInteractive(True)
            display.SetTranslationHandleVisibility(True)
            display.SetRotationHandleVisibility(False)
            display.SetScaleHandleVisibility(False)
            # two arrows perpendicular to the pin + the centre ball for free dragging; any movement
            # along the pin is removed, so the pin always stays parallel
            display.SetTranslationHandleComponentVisibility([True, True, False, True])
            display.SetInteractionHandleScale(4.0)
            # no snapping onto visible surfaces (bone, sleeves, the preview): smooth, continuous drag
            display.SetSnapMode(slicer.vtkMRMLMarkupsDisplayNode.SnapModeUnconstrained)
            display.SetOccludedVisibility(True)
            display.SetOccludedOpacity(0.6)
            self._pin_gizmo_tags = None
        if getattr(self, "_pin_gizmo_tags", None) is None or self._pin_gizmo_tags[0] != gizmo.GetID():
            self._pin_gizmo_tags = (gizmo.GetID(), [
                gizmo.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._on_pin_gizmo_moved),
                gizmo.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_pin_gizmo_moved),
                gizmo.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self._on_pin_gizmo_released)])
        position = np.array(node.GetNthControlPointPositionWorld(index), float)
        lifted = position + self.PIN_GIZMO_LIFT * np.asarray(axes[2], float)
        self._pin_mark_origin = position
        self._pin_gizmo_updating = True
        try:
            gizmo.SetAxesWorld(*[[float(v) for v in a] for a in axes])
            gizmo.SetOriginWorld(*[float(v) for v in lifted])
            gizmo.SetSize(5.0, 5.0)
            gizmo.GetDisplayNode().SetVisibility(True)
        finally:
            self._pin_gizmo_updating = False
        self._pin_gizmo_origin, self._pin_gizmo_z = lifted, np.asarray(axes[2], float)

    PIN_PREVIEW = "Doggybone pin preview"

    def _pin_gizmo_hovered(self):
        gizmo = slicer.mrmlScene.GetFirstNodeByName(self.PIN_GIZMO)
        display = gizmo.GetDisplayNode() if gizmo is not None else None
        if display is None or not display.GetVisibility():
            return False
        try:
            return display.GetActiveComponentType() != slicer.vtkMRMLMarkupsDisplayNode.ComponentNone
        except Exception:
            return False

    def _show_pin_preview(self, point):
        """Thin line along the pin axis through `point` while the gizmo is dragged (no recalculation)."""
        z = getattr(self, "_pin_gizmo_z", None)
        if z is None:
            return
        node = slicer.mrmlScene.GetFirstNodeByName(self.PIN_PREVIEW)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.PIN_PREVIEW)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(1.0, 0.55, 0.1)
            node.GetDisplayNode().SetVisibility2D(False)
            node.SetSelectable(False)               # never picked / snapped onto while dragging
        line = vtk.vtkLineSource()
        p = np.asarray(point, float)
        line.SetPoint1(*[float(v) for v in p + 15.0 * z])
        line.SetPoint2(*[float(v) for v in p - 40.0 * z])
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(max(0.4, 0.5 * self._pin_diameter()))
        tube.SetNumberOfSides(12)
        tube.CappingOn()
        tube.Update()
        node.SetAndObservePolyData(tube.GetOutput())

    def _remove_pin_preview(self):
        node = slicer.mrmlScene.GetFirstNodeByName(self.PIN_PREVIEW)
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    def _on_pin_gizmo_moved(self, caller, event=None):
        if getattr(self, "_pin_gizmo_updating", False):
            return
        index = self._selected_pin_index()
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if index is None or node is None or getattr(self, "_pin_gizmo_origin", None) is None:
            return
        origin = [0.0, 0.0, 0.0]
        caller.GetOriginWorld(origin)
        z = self._pin_gizmo_z
        delta = np.asarray(origin, float) - self._pin_gizmo_origin
        delta = delta - float(np.dot(delta, z)) * z          # sideways only: the pin stays on its line
        target = getattr(self, "_pin_mark_origin", self._pin_gizmo_origin) + delta
        node.SetNthControlPointPositionWorld(index, *[float(v) for v in target])
        self._show_pin_preview(target)

    def _on_pin_gizmo_released(self, caller=None, event=None):
        if getattr(self, "_pin_gizmo_updating", False):
            return
        qt.QTimer.singleShot(0, self._finish_pin_move)

    def _finish_pin_move(self):
        self._remove_pin_preview()
        index = self._selected_pin_index()
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if index is None or node is None:
            return
        point = np.array(node.GetNthControlPointPositionWorld(index), float)
        M, moved = self._correction_state()
        if M is None:
            return
        direction, shafts, error = self._reduction_pin_frame(M, moved)
        if error:
            self.guide_status.setText(error)
            return
        fragments = self._corrected_fragments()
        centred = self.pin_centre_checkbox.isChecked()
        pin = (self._center_reduction_pin(point, fragments, (direction, shafts)) if centred
               else self._free_reduction_pin(point, fragments, direction))
        label = node.GetNthControlPointLabel(index)
        if "error" in pin:
            back = getattr(self, "_pin_mark_origin", self._pin_gizmo_origin)
            node.SetNthControlPointPositionWorld(index, *[float(v) for v in back])
            self._place_pin_gizmo()
            self._pin_feedback(f"Pin {label} not moved: {pin['error']}.", point, ok=False)
            return
        pin.setdefault("centre", point)
        pin, shift = self._keep_clear_of_cut(pin, fragments, direction, centred)
        node.SetNthControlPointPositionWorld(index, *[float(v) for v in pin["entry"]])
        node.SetNthControlPointDescription(index, f"{pin['segment']}|{'manual' if centred else 'free'}")
        self._relabel_pins(node)
        self._pins_dirty = True
        self._show_reduction_pin_axes()
        self._place_pin_gizmo()
        # would the saw guide accept it (distance to the cut in the as-scanned pose)?
        raw_entry = np.asarray(pin["entry"], float)
        if pin["segment"] == moved:
            raw_entry = (np.linalg.inv(M) @ np.append(raw_entry, 1.0))[:3]
        ok_segment, reason = self._pin_segment(raw_entry)
        label = node.GetNthControlPointLabel(index)
        text = (f"Pin {label} moved ({pin['segment']} fragment, {'centred' if centred else 'free'}), "
                f"{pin['bone_length']:.1f} mm in bone, parallel to the other pins. Rebuild the guide "
                "('Extrude guide body') when all pins are placed."
                + (f"\nPushed {shift:.1f} mm away from the cut so the saw guide can take it." if shift > 0 else ""))
        if ok_segment is None:
            self.guide_status.setStyleSheet(STATUS_WARN_STYLE)
            self.guide_status.setText(text + f"\nWarning: the saw guide cannot take this pin - {reason}.")
        else:
            self.guide_status.setStyleSheet("")
            self.guide_status.setText(text)

    def _show_reduction_pin_axes(self):
        """Axes of the pins set after the reduction (corrected pose) - ALWAYS into the reduction node,
        also when called while the saw guide is shown (they must never overwrite the saw guide's axes)."""
        pins, _rejected = self._reduction_marked_pins()
        saved = getattr(self, "_guide_mode", "saw")
        self._guide_mode = "reduction"
        try:
            self._set_pin_axes(guide_logic.pin_axis_polydata(pins) if pins else None)
            node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PIN_AXES_NAME)
        finally:
            self._guide_mode = saved
        if node is not None and node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(saved == "reduction")

    def _toggle_reduction_pin(self, hit):
        node = slicer.mrmlScene.GetFirstNodeByName(self.REDUCTION_PIN_MARKS)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.REDUCTION_PIN_MARKS)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
            node.GetDisplayNode().SetSelectedColor(0.95, 0.95, 0.3)
            node.GetDisplayNode().SetGlyphScale(1.5)
            node.GetDisplayNode().SetTextScale(2.5)
        p = np.asarray(hit, float)
        for i in range(node.GetNumberOfControlPoints()):
            if np.linalg.norm(np.array(node.GetNthControlPointPositionWorld(i)) - p) < 2.0:
                node.RemoveNthControlPoint(i)
                self._relabel_pins(node)
                self._pins_dirty = True
                self._show_reduction_pin_axes()
                self.guide_status.setText(f"Pin removed ({node.GetNumberOfControlPoints()} set).")
                return
        centred = self.pin_centre_checkbox.isChecked() if hasattr(self, "pin_centre_checkbox") else True
        if centred:
            pin = self._center_reduction_pin(p)
        else:
            M, moved = self._correction_state()
            direction, _shafts, error = self._reduction_pin_frame(M, moved) if M is not None \
                else (None, None, "no correction preview")
            pin = {"error": error} if error else self._free_reduction_pin(p, self._corrected_fragments(), direction)
        if "error" in pin:
            self._pin_feedback(f"No pin there: {pin['error']}.", p, ok=False)
            return
        M_, moved_ = self._correction_state()
        direction_ = self._reduction_pin_frame(M_, moved_)[0] if M_ is not None else None
        shift = 0.0
        if direction_ is not None:
            pin.setdefault("centre", p)
            pin, shift = self._keep_clear_of_cut(pin, self._corrected_fragments(), direction_, centred)
        index = node.AddControlPointWorld(vtk.vtkVector3d(*[float(v) for v in pin["entry"]]))
        node.SetNthControlPointDescription(index, f"{pin['segment']}|{'manual' if centred else 'free'}")
        self._relabel_pins(node)
        self._pins_dirty = True
        self._pin_feedback(None, p, ok=True)
        self._show_reduction_pin_axes()
        self._select_reduction_pin(index)
        self.guide_status.setText(
            f"Pin {node.GetNthControlPointLabel(index)} set ({pin['segment']} fragment) "
            f"{'through the bone centre' if centred else 'where you clicked'}, "
            f"{pin['bone_length']:.1f} mm in bone; {node.GetNumberOfControlPoints()} set. All pins parallel to "
            "each other and to the cut plane(s). Fine-tune it with the small gizmo; switch Pins off to "
            "build the reduction guide." + (f"\nMoved {shift:.1f} mm away from the cut so the saw guide can take it."
                                            if shift > 0 else ""))

    def _toggle_pin(self, hit):
        if self._guide_mode_is_reduction():
            self._toggle_reduction_pin(hit)
            return
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PINS_NAME)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.GUIDE_PINS_NAME)
            node.CreateDefaultDisplayNodes()
            node.SetLocked(True)
            node.GetDisplayNode().SetSelectedColor(0.95, 0.95, 0.3)
            node.GetDisplayNode().SetGlyphScale(1.5)
            node.GetDisplayNode().SetTextScale(2.5)
        p = np.asarray(hit, float)
        for i in range(node.GetNumberOfControlPoints()):
            q = np.array(node.GetNthControlPointPositionWorld(i))
            if np.linalg.norm(q - p) < 2.0:
                node.RemoveNthControlPoint(i)
                self._relabel_pins(node)
                self._pins_dirty = True
                self.guide_status.setText(f"Pin removed ({node.GetNumberOfControlPoints()} marked). "
                                          "Extrude guide body to update the sleeves.")
                return
        segment, reason = self._pin_segment(p)
        if segment is None:
            self._pin_feedback(f"No pin there: {reason}.", p, ok=False)
            return
        index = node.AddControlPointWorld(vtk.vtkVector3d(*p))
        node.SetNthControlPointDescription(index, segment)
        self._relabel_pins(node)
        self._pins_dirty = True
        self._pin_feedback(None, p, ok=True)
        self.guide_status.setText(f"Pin {node.GetNthControlPointLabel(index)} marked ({segment}); "
                                  f"{node.GetNumberOfControlPoints()} marked. Mark further pins, then "
                                  "Extrude guide body to build sleeves and bores.")

    @staticmethod
    def _relabel_pins(node):
        counts = {}
        for i in range(node.GetNumberOfControlPoints()):
            segment = (node.GetNthControlPointDescription(i) or "pin").split("|")[0] or "pin"
            counts[segment] = counts.get(segment, 0) + 1
            node.SetNthControlPointLabel(i, f"{segment[0].upper()}{counts[segment]}")

    def _set_pin_axes(self, polydata):
        node = slicer.mrmlScene.GetFirstNodeByName(self.GUIDE_PIN_AXES_NAME)
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
            return
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", self.GUIDE_PIN_AXES_NAME)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(0.75, 0.75, 0.78)
        node.SetAndObservePolyData(polydata)

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
        for a_index in range(len(pins)):
            for b_index in range(a_index + 1, len(pins)):
                a, b = pins[a_index], pins[b_index]
                if np.linalg.norm(a["entry"] - b["entry"]) < a["r_out"] + b["r_out"]:
                    rejected.append(f"{a['label']}/{b['label']}: sleeves overlap")
        note = ("Pin sleeves from the pins set after the reduction: " + ", ".join(f"{v} {k}" for k, v in counts.items())
                + (f", in bone {min(lengths):.1f}-{max(lengths):.1f} mm" if lengths else "")
                + " - pre-angled, parallel after the reduction.")
        if rejected:
            note += "\nNot built: " + "; ".join(rejected)
        return pins, note

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
        for a_index in range(len(pins)):
            for b_index in range(a_index + 1, len(pins)):
                a, b = pins[a_index], pins[b_index]
                if np.linalg.norm(a["entry"] - b["entry"]) < a["r_out"] + b["r_out"]:
                    rejected.append(f"{a['label']}/{b['label']}: sleeves overlap")
        note = (f"Pin sleeves: " + ", ".join(f"{v} {k}" for k, v in counts.items()) +
                (f", \u00d8 {diameter:.1f} mm, in bone {min(lengths):.1f}-{max(lengths):.1f} mm" if lengths else ""))
        group_angle = float(np.degrees(np.arccos(np.clip(float(np.dot(condylar_direction, opposite_direction)), -1.0, 1.0))))
        note += (f"\n{condylar.capitalize()} (condylar) pins perpendicular to the condylar axis; {opposite} pins "
                 f"pre-angled {group_angle:.1f}\N{DEGREE SIGN} - parallel after the reduction.")
        if rejected:
            note += "\nNot built: " + "; ".join(rejected)
        return pins, note

    # --- saw boxes: one gizmo per box, moving / turning it only WITHIN its cutting plane -------------
    def _saw_box_gizmo_name(self, label):
        return f"Doggybone saw box {label}"

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
        self.guide_status.setStyleSheet(STATUS_WARN_STYLE)
        self.guide_status.setText(
            f"Saw box {label} adjusted in its cutting plane: turned {np.degrees(theta):+.1f}\N{DEGREE SIGN}, "
            f"extended {du:+.1f} mm along the kerf, height {dv:+.1f} mm - press 'Extrude guide body' to "
            "rebuild the guide. The kerf stays exactly in the planned cut.")

    def _plan_saw_slots(self, patch, gap):
        """Slots for the cutting planes that exist. Returns (slots, notes, pin text)."""
        from vtk.util.numpy_support import vtk_to_numpy
        side = self.osteotomy_side.currentText.lower()
        bone = self.osteotomy_bone.currentText
        planes = []
        for label, name in (("proximal", self._plane_node_name(side, bone)),
                            ("distal", self._distal_plane_node_name(side, bone))):
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
                                         adjust=self._saw_box_adjustment(label))
            if "error" in slot:
                notes.append(f"{label}: {slot['error']}")
                continue
            slot["label"] = label
            slots.append(slot)
            notes.append(f"{label} {slot['length']:.1f} mm long (bone {slot['bone_width']:.1f} mm)")
        chosen = self.guide_pin.currentText
        d = min(diameters) if diameters else None
        self._last_bone_diameter = d
        if not chosen.startswith("Auto"):
            pin = chosen + (f" (chosen; bone \u00d8 {d:.1f} mm)" if d is not None else " (chosen)")
        elif d is not None:
            pin = (f"{guide_logic.auto_pin_diameter(d):.1f} mm auto (bone \u00d8 {d:.1f} mm; up to "
                   f"{guide_logic.PIN_THRESHOLD_MM:.0f} mm -> 1.2 mm, larger -> 1.6 mm)")
        else:
            pin = "1.6 mm auto (bone diameter not measurable)"
        return slots, notes, pin

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
        return f"Doggybone {self.side} {label} sphere"

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
            display_node.SetVisibility2D(True)          # circle in every slice view (MPR)
            display_node.SetSliceIntersectionThickness(2)
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
        return f"Doggybone {self.side} {label} axis"

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
            for landmark in self.landmarks:
                markups_node.RemoveAttribute(f"Doggybone.radius.{landmark.name}")

        # Remove the corrected-landmark overlay (spheres/axes on the corrected bone) of this side.
        prefix = f"doggybone {self.side} corrected".lower()
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            if node.GetName().lower().startswith(prefix):
                slicer.mrmlScene.RemoveNode(node)

        for index in range(self.measurement_stack.count):
            widget = self.measurement_stack.widget(index)
            try:
                widget.update_measurement()
            except Exception:
                pass

        if current_widget is not None:
            current_widget.enable()

        base = getattr(self, "base_widget", None)
        if base is not None and hasattr(base, "_on_target_values_changed"):
            try:
                base._on_target_values_changed()
            except Exception:
                pass

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
                    self.measurement_label.setStyleSheet(MEASUREMENT_VALUE_STYLE)
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
                    self.measurement_label.setStyleSheet(STATUS_WARN_STYLE)
                    self.axes_label.setText("")
                    for label in self.measurement.SPHERE_LABELS:
                        self.dialog.hide_sphere(label)
                    for label in self.measurement.AXIS_LABELS:
                        self.dialog.hide_axis(label)
            except Exception as e:
                self.measurement_label.setText(f"Error executing measurement: {e}")
                self.measurement_label.setStyleSheet(STATUS_ERROR_STYLE)
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
        """
        Delete one placed landmark per click: the selected one if it is placed, otherwise the
        most recently placed landmark of this measurement, otherwise the most recently placed
        landmark of this side. The control point is physically removed from the markups node
        and the indices of all later landmarks are shifted down, so the point list stays clean.
        """
        panel = self.dialog
        side_landmarks = list(getattr(panel, "landmarks", None) or self.measurement.get_landmarks())
        own = list(self.measurement.get_landmarks())
        placed = lambda seq: [l for l in seq if l.placed and l._id is not None]

        current_widget = self.landmark_stack.currentWidget()
        current = current_widget.landmark if current_widget is not None else None
        target = current if (current is not None and current.placed and current._id is not None) else None
        if target is None and placed(own):
            target = max(placed(own), key=lambda l: l._id)
        if target is None and placed(side_landmarks):
            target = max(placed(side_landmarks), key=lambda l: l._id)
        if target is None:
            # Nothing placed on this side: the visible points/labels may belong to the other side
            # (e.g. the side selector was switched after placing). Delete from there instead.
            base = getattr(panel, "base_widget", None)
            other = None
            if base is not None and not getattr(self, "_deleting_other_side", False):
                for candidate in (getattr(base, "left_panel", None), getattr(base, "right_panel", None)):
                    if candidate is not None and candidate is not panel:
                        other = candidate
            if other is not None and any(l.placed and l._id is not None for l in other.landmarks):
                other_widget = other.measurement_stack.currentWidget()
                if other_widget is not None:
                    other_widget._deleting_other_side = True
                    try:
                        other_widget._delete_current_landmark()
                    finally:
                        other_widget._deleting_other_side = False
                    if current is not None:
                        current.start_interaction()
                    return
            slicer.util.showStatusMessage(f"No placed landmarks left to delete.", 4000)
            return

        if current is not None:
            current.stop_interaction()
            current._private_observers = []
        target.stop_interaction()
        target._private_observers = []

        node = target._markups_node
        index = target._id
        if node is not None and 0 <= index < node.GetNumberOfControlPoints():
            node.RemoveNthControlPoint(index)
        for landmark in side_landmarks:
            if landmark is not target and landmark._id is not None and landmark._id > index:
                landmark._id -= 1
        target._id = None
        target.placed = False
        if node is not None and hasattr(target, "radius"):
            node.RemoveAttribute(f"Doggybone.radius.{target.name}")
        for callback in target.change_callbacks:
            try:
                callback()
            except Exception as exc:
                print(f"Doggybone: callback after delete failed: {exc}")

        remaining = len(placed(side_landmarks))
        slicer.util.showStatusMessage(f"Deleted '{target.name}' - {remaining} landmark(s) left on this side.", 4000)

        # Select the deleted landmark so the next click in the view places it again.
        if getattr(self, "_deleting_other_side", False):
            pass
        elif target in own:
            row = own.index(target)
            if self.landmark_list.currentRow == row:
                target.start_interaction()
            else:
                self.landmark_list.setCurrentRow(row)
        elif current is not None:
            current.start_interaction()

        base = getattr(panel, "base_widget", None)
        if base is not None and hasattr(base, "_on_target_values_changed"):
            try:
                base._on_target_values_changed()
            except Exception:
                pass

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
        self.radius_spinbox.setToolTip(
            "Sphere radius. Easier: drag the orange 'r' point on the circle in a slice view (MPR) - the "
            "circle shows the sphere in every slice.")
        landmark.add_change_callback(self._sync_radius)

    def _sync_radius(self):
        if abs(self.radius_spinbox.value - float(self.landmark.radius)) > 1e-3:
            was = self.radius_spinbox.blockSignals(True)
            self.radius_spinbox.setValue(float(self.landmark.radius))
            self.radius_spinbox.blockSignals(was)

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