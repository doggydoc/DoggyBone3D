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
import locale   # (kept for compatibility; the process-wide C locale is NOT changed any more)


def _csv_decimal():
    """Decimal separator of the user's system (German: ',') - only used for the CSV files, so they
    open correctly in Excel; Slicer / VTK keep the C locale ('.')."""
    try:
        point = qt.QLocale.system().decimalPoint()
        point = str(point) if point is not None else "."
        return point if point in (",", ".") else "."
    except Exception:
        return "."


def _csv_number(value):
    return repr(float(value)).replace(".", _csv_decimal())


def _csv_float(text):
    return float(str(text).strip().replace(",", "."))


def _csv_delimiter_for_file(handle):
    first = handle.readline()
    handle.seek(0)
    return ";" if ";" in first else ","

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

# UI parts of DoggyboneWidget, one file per workflow step (see DoggyboneUI/). Reloaded like the
# DoggyboneLib modules above so Slicer's "Reload" picks up edits.
import DoggyboneUI.common
importlib.reload(DoggyboneUI.common)
from DoggyboneUI.common import *   # MODULE_PATH and the STATUS_* / *_VALUE_STYLE colours
importlib.reload(importlib.import_module('DoggyboneUI.ui_planes'))
importlib.reload(importlib.import_module('DoggyboneUI.ui_correction'))
importlib.reload(importlib.import_module('DoggyboneUI.ui_segmentation'))
importlib.reload(importlib.import_module('DoggyboneUI.ui_guide_edit'))
importlib.reload(importlib.import_module('DoggyboneUI.ui_guide_build'))
importlib.reload(importlib.import_module('DoggyboneUI.ui_pins'))
importlib.reload(importlib.import_module('DoggyboneUI.ui_saw'))
importlib.reload(importlib.import_module('DoggyboneUI.ui_tibia'))
importlib.reload(importlib.import_module('DoggyboneUI.ui_guide_volume'))
from DoggyboneUI.ui_planes import PlanesMixin
from DoggyboneUI.ui_correction import CorrectionMixin
from DoggyboneUI.ui_segmentation import SegmentationMixin
from DoggyboneUI.ui_guide_edit import GuideEditMixin
from DoggyboneUI.ui_guide_build import GuideBuildMixin
from DoggyboneUI.ui_pins import PinsMixin
from DoggyboneUI.ui_saw import SawMixin
from DoggyboneUI.ui_tibia import TibiaTargetsMixin
from DoggyboneUI.ui_guide_volume import GuideVolumeMixin

#
# Doggybone
#

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


class DoggyboneWidget(GuideVolumeMixin, TibiaTargetsMixin, PlanesMixin, CorrectionMixin, SegmentationMixin, GuideEditMixin, GuideBuildMixin, PinsMixin, SawMixin,
                      ScriptedLoadableModuleWidget):
    """
    Base widget that manages the measurement groups
    """

    SEGMENTATION_NODE_NAME = "Doggybone Bone Segmentation"

    # Which existing angle measurement feeds the "suggest" button for each correction and bone -
    # see MeasurementsPanel/measurement_logic.py for where these names/descriptions come from.
    WEDGE_MEASUREMENT_NAME = {"Femur": "aLDFA (Varus/Valgus Femur)", "Tibia": "Varus Valgus Tibia"}
    ROTATION_MEASUREMENT_NAME = {"Femur": "Antetorsion", "Tibia": "Tibia Torsion"}
    # Default target ("Zielwert") each measurement should read after correction, per bone.
    # Femur: aLDFA target. Tibia: frontal angle between the two joint lines (mMPTA + mMDTA - 180,
    # torsion-free), target = canine reference mMPTA + mMDTA - 180 (see measurement_logic.py). The rotation targets are placeholders (0°) - this
    # module has no reliable breed/bone-specific reference for normal antetorsion/tibial torsion,
    # so these MUST be reviewed and set by the vet rather than trusted as clinical defaults.
    WEDGE_TARGET_DEFAULT = {"Femur": 94.0, "Tibia": round(measurement_logic.TIBIA_MPTA_REF_DEG + measurement_logic.TIBIA_MDTA_REF_DEG - 180.0, 1)}   # aLDFA target (canine reference ~94°)
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
        self.opacity_slider.setToolTip(BONE_OPACITY_TIP)
        self.opacity_slider.valueChanged.connect(self._set_bone_opacity)
        segmentation_form_layout.addRow("Bone opacity", self.opacity_slider)

        self.segmentation_status = qt.QLabel("no segmentation")
        self.segmentation_status.setWordWrap(True)
        segmentation_form_layout.addRow("Status", self.segmentation_status)

        # take over a segmentation made in the Segment Editor (only its VISIBLE segments)
        self.take_over_button = qt.QPushButton("Use Segment Editor segmentation")
        self.take_over_button.setToolTip(
            "Takes the segmentation you made in the Segment Editor as the Doggybone bone segmentation. "
            "Only the segments that are VISIBLE there (eye icon) are used and merged into one bone - "
            "hide a segment in the Segment Editor to leave it out. Replaces the current Doggybone "
            "segmentation (the editor's segmentation itself stays unchanged and is hidden).")
        self.take_over_button.clicked.connect(self.onTakeOverEditorSegmentation)
        segmentation_form_layout.addRow("From editor", self.take_over_button)

        # Clean-up of the bone segmentation with Segment Editor effects: a free-form lasso in the
        # 3D view keeps only the encircled region (through the full depth), then small islands
        # (noise, CT table fragments) are removed.
        self.erase_inside_button = qt.QPushButton("Erase inside")
        self.erase_inside_button.setCheckable(True)
        self.erase_inside_button.setToolTip(
            "Switch on, then draw around the part to REMOVE in the 3D view (hold the left mouse button): "
            "everything inside the lasso is removed through the full depth - rotate the view so that nothing "
            "you want to keep lies in front of or behind it. One stroke per click.")
        self.erase_inside_button.toggled.connect(self.onEraseInsideToggled)
        self.lasso_button = qt.QPushButton("Erase outside")
        self.lasso_button.setCheckable(True)
        self.lasso_button.setToolTip(
            "Switch on, then draw around the region to KEEP in the 3D view (hold the left mouse button): "
            "everything outside the lasso is removed through the full depth. One stroke per click.")
        self.lasso_button.toggled.connect(self.onLassoToggled)
        self.delete_cleanup_button = qt.QPushButton("Delete")
        self.delete_cleanup_button.setToolTip(
            "Delete all clean-up edits (erase inside / outside, islands, remove bone, fill cavities) - back to the "
            "segmentation as it was before the first clean-up step, to start again.")
        self.delete_cleanup_button.clicked.connect(self.onDeleteSegmentationEdits)
        lasso_row = qt.QHBoxLayout()
        lasso_row.setContentsMargins(0, 0, 0, 0)
        lasso_row.addWidget(self.erase_inside_button, 1)
        lasso_row.addWidget(self.lasso_button, 1)
        lasso_row.addWidget(self.delete_cleanup_button, 1)
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
        # A closed scene (File > Close Scene) deletes the landmark lists - the panels would then
        # place plain "F-1, F-2 ..." points. After closing or loading a scene the module is rebuilt,
        # which creates / re-links the landmark lists.
        self._scene_observers = [
            slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self._on_scene_reset),
            slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndImportEvent, self._on_scene_reset)]
        # the shown current values / calculated corrections follow the landmarks: refresh (debounced)
        # whenever a landmark is placed, moved or deleted - and once after the module is (re)loaded
        self._landmark_observers = []
        for panel in (self.left_panel, self.right_panel):
            node = slicer.mrmlScene.GetNodeByID(panel.markup_node_id)
            if node is None:
                continue
            for event in (slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                          slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                          slicer.vtkMRMLMarkupsNode.PointRemovedEvent):
                self._landmark_observers.append((node, node.AddObserver(event, self._schedule_values_refresh)))

        self.side_stack = qt.QStackedWidget()
        self.side_stack.addWidget(self.left_panel)
        self.side_stack.addWidget(self.right_panel)

        side_row = qt.QHBoxLayout()
        side_row.addWidget(qt.QLabel("Side"))
        side_row.addWidget(self.side_selector)
        side_row.addSpacing(16)
        side_row.addWidget(qt.QLabel("Bone opacity"))
        self.measurement_opacity_slider = self._new_bone_opacity_slider()
        side_row.addWidget(self.measurement_opacity_slider, 1)
        side_row_widget = qt.QWidget()
        side_row_widget.setLayout(side_row)

        measurements_layout.addWidget(side_row_widget)

        # slice views aligned to the BONE instead of the scanner (the limb is never lying exactly
        # along the scanner axes): red = perpendicular to the shaft axis, green = frontal plane
        # (shaft + condylar axis), yellow = sagittal plane
        align_row = qt.QHBoxLayout()
        align_row.addWidget(qt.QLabel("Slice views"))
        self.view_align_selector = qt.QComboBox()
        self.view_align_selector.addItems(["Scanner", "Aligned to femur", "Aligned to tibia"])
        self.view_align_selector.setToolTip(
            "Scanner: the usual axial / coronal / sagittal views of the CT.\n"
            "Aligned: red = perpendicular to the shaft axis, green = frontal plane (shaft axis + "
            "caudal condylar axis), yellow = sagittal plane - independent of how the limb lay in "
            "the scanner. Needs the two shaft midpoints (and the condyle points for the rotation).")
        self.view_align_selector.currentIndexChanged.connect(lambda _i: self.onAlignSliceViews())
        align_row.addWidget(self.view_align_selector, 1)
        self.view_align_update = qt.QPushButton("Update")
        self.view_align_update.setToolTip("Re-align after landmarks were moved, centred on the crosshair.")
        self.view_align_update.clicked.connect(self.onAlignSliceViews)
        align_row.addWidget(self.view_align_update)
        align_row_widget = qt.QWidget()
        align_row_widget.setLayout(align_row)
        measurements_layout.addWidget(align_row_widget)
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
        self.osteotomy_technique.currentIndexChanged.connect(lambda _i: self._on_technique_changed())

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
        self._osteotomy_form_layout = osteotomy_form_layout
        sagittal_field = self._build_tibia_targets(osteotomy_form_layout)   # tibia: table of current / target values

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
        osteotomy_form_layout.addRow("Sagittal (TPA)", sagittal_field)

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
        # ONE visible status line for the whole osteotomy section (planes and correction preview);
        # both names point to it (older code writes to either)
        self.correction_status = qt.QLabel()
        self.correction_status.setWordWrap(True)
        self.correction_status.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.osteotomy_status = self.correction_status

        #############################
        # Corrective osteotomy (wedge + rotation, with live preview)
        #############################
        self.rotation_invert = qt.QCheckBox()
        self.rotation_invert.setVisible(False)
        self.rotation_suggest_button = qt.QPushButton()
        self.rotation_suggest_button.setVisible(False)

        osteotomy_form_layout.addRow(qt.QLabel("<b>Preview</b>"))

        self.reduction_gap = qt.QDoubleSpinBox()
        self.reduction_gap.setRange(0.0, 0.5)
        self.reduction_gap.setDecimals(2)
        self.reduction_gap.setSingleStep(0.05)
        self.reduction_gap.setValue(0.2)
        self.reduction_gap.setSuffix(" mm")
        self.reduction_gap.setToolTip(
            "Small gap left between the cut faces after reduction (0.1-0.3 mm recommended) to avoid "
            "tension. The moved fragment is shifted along the cut normal; the reduction-guide pins "
            "follow, i.e. they sit this much farther apart than in the cutting guide. "
            "Applied with the next preview update.")
        osteotomy_form_layout.addRow("Reduction gap", self.reduction_gap)

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
        self.reset_all_button = qt.QPushButton("Reset planning and guides")
        self.reset_all_button.setToolTip(
            "Start the osteotomy over: removes the preview, both cutting planes AND everything of the saw "
            "guide and the reduction guide (footprints, guide bodies, pins, sleeves, saw boxes, labels). "
            "Landmarks, measurements, segmentation and target values are kept. Asks before deleting.")
        self.reset_all_button.setStyleSheet("QPushButton { color: #9b1c1c; }")
        self.reset_all_button.clicked.connect(self.onResetPlanningAndGuides)
        preview_grid.addWidget(self.reset_all_button, 2, 0, 1, 2)
        self.hide_annotations_button = qt.QCheckBox("Hide labels and cutting planes")
        self.hide_annotations_button.setToolTip(
            "One click hides all labels (landmark names, plane names, L/M markers) and the cutting "
            "planes - e.g. for a clean view or a screenshot. Click again to show them as before.")
        self.hide_annotations_button.toggled.connect(self.onHideAnnotationsToggled)
        preview_widget = qt.QWidget()
        preview_widget.setLayout(preview_grid)
        osteotomy_form_layout.addRow(preview_widget)


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
        osteotomy_form_layout.addRow(self.hide_annotations_button)

        self.segment_opacity_slider = self._new_bone_opacity_slider()
        osteotomy_form_layout.addRow("Bone opacity", self.segment_opacity_slider)
        osteotomy_form_layout.addRow("Status", self.correction_status)

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
        reduction_grid.addWidget(self.place_pin_rows_button, 5, 0, 1, 2)
        self.reduction_delete_pins_button = qt.QPushButton("Delete pins")
        self.reduction_delete_pins_button.setToolTip("Delete all pins (and the reduction guide built on them).")
        self.reduction_delete_pins_button.clicked.connect(self.onDeletePins)
        reduction_grid.addWidget(self.reduction_delete_pins_button, 5, 2, 1, 2)
        self.reduction_ring_checkbox = qt.QCheckBox("Ring (closed frame around all pins)")
        self.reduction_ring_checkbox.setChecked(True)
        self.reduction_ring_checkbox.setToolTip(
            "The webs run as a closed ring along the outline of all pins (plus links to inner pins) instead "
            "of a single chain - two bridges across the osteotomy, much stiffer against rotation and "
            "bending. Needs pins that do not all lie in one line.")
        reduction_grid.addWidget(self.reduction_ring_checkbox, 6, 0, 1, 4)
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
        reduction_grid.addWidget(self.pin_row_pattern, 4, 3)
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
        self.pin_tilt = row_spin(-60.0, 60.0, 5.0, "\N{DEGREE SIGN}",
                                 "Turns all pins WITHIN the cutting plane (about the cut normal) - for a better "
                                 "approach to the bone. The pins stay parallel to the cut (they can never meet the "
                                 "saw blade) and parallel to each other. + = pin heads towards proximal.", decimals=0)
        self.pin_tilt.setValue(0.0)
        reduction_grid.addWidget(qt.QLabel("Pin tilt"), 7, 0)
        reduction_grid.addWidget(self.pin_tilt, 7, 1)
        reduction_grid.addWidget(self.reduction_contact_only_checkbox, 8, 0, 1, 3)
        reduction_grid.addWidget(self.reduction_clearance, 8, 3)
        for box in (self.pin_row_count, self.pin_row_spacing, self.pin_row_distance_proximal,
                    self.pin_row_distance_distal, self.pin_tilt):
            box.valueChanged.connect(lambda _v: self._pin_rows_changed())
        self.reduction_type.currentIndexChanged.connect(lambda _i: self._update_guide_tool_visibility())
        # same look as the guide parameters below: label | field | label | field, two equal field
        # columns, every field / button as high as 'Extrude guide body'
        reduction_grid.setColumnStretch(1, 1)
        reduction_grid.setColumnStretch(3, 1)
        reduction_grid.setVerticalSpacing(2)
        row_height = qt.QPushButton("X").sizeHint.height()
        for widget in (self.pin_row_count, self.pin_row_spacing, self.pin_row_distance_proximal,
                       self.pin_row_distance_distal, self.pin_row_pattern, self.place_pin_rows_button,
                       self.reduction_delete_pins_button, self.pin_tilt, self.reduction_clearance,
                       self.reduction_standoff, self.reduction_bar_diameter):
            widget.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            widget.setMinimumWidth(60)
            widget.setMinimumHeight(row_height)
        self._reduction_row_height = row_height
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
        tools_row.setColumnStretch(0, 1)          # two equal columns
        tools_row.setColumnStretch(1, 1)
        tool_positions = {"brush_add": (0, 0), "brush_remove": (0, 1), "lasso_add": (1, 0),
                          "lasso_remove": (1, 1), "pins": (3, 0)}
        for key, label, tip in (("brush_add", "Brush +", "Colour the surface under the mouse (drag)."),
                                ("brush_remove", "Brush -", "Uncolour the surface under the mouse (drag) - removes footprint and Smooth marks."),
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
        # pins are selected by clicking them in the 3D view (reduction guide); the tool button is only
        # used internally for manual pin placement ('Pin distrib.: Manual' -> 'Place pins')
        self._guide_tool_buttons["pins"].setVisible(False)
        self.guide_delete_button = qt.QPushButton("Delete")
        self.guide_delete_button.setToolTip(
            "Delete the whole footprint and the guide body - then paint again "
            "or by painting with Brush + / Lasso +.")
        self.guide_delete_button.clicked.connect(self.onGuideDelete)
        self.guide_delete_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        # row 3: brush radius (under 'Lasso +') and 'Delete' next to it
        self.guide_brush_radius = spin(0.5, 15.0, 3.0, 0.5, " mm", "Brush radius.")
        self.guide_brush_radius.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        brush_radius_layout = qt.QHBoxLayout()
        brush_radius_layout.setContentsMargins(0, 0, 0, 0)
        brush_radius_layout.setSpacing(6)
        brush_radius_layout.addWidget(qt.QLabel("Brush size"))
        brush_radius_layout.addWidget(self.guide_brush_radius, 1)
        self._brush_radius_widget = qt.QWidget()
        self._brush_radius_widget.setLayout(brush_radius_layout)
        self._brush_radius_widget.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        tools_row.addWidget(self._brush_radius_widget, 2, 0)
        tools_row.addWidget(self.guide_delete_button, 2, 1)
        self.guide_pins_delete_button = qt.QPushButton("Pins delete")
        self.guide_pins_delete_button.setToolTip("Remove all pin marks, sleeves and bores (the guide is rebuilt without pins).")
        self.guide_pins_delete_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        self.guide_pins_delete_button.clicked.connect(self.onDeletePins)
        tools_row.addWidget(self.guide_pins_delete_button, 3, 0)
        self.guide_pin_delete_selected_button = qt.QPushButton("Delete selected pin")
        self.guide_pin_delete_selected_button.setToolTip(
            "Reduction guide: click a pin in the 3D view to select it (gizmo to move it), then delete it "
            "here or with the Delete key.")
        self.guide_pin_delete_selected_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        self.guide_pin_delete_selected_button.clicked.connect(self.onDeleteSelectedPin)
        tools_row.addWidget(self.guide_pin_delete_selected_button, 3, 1)
        tools_widget = qt.QWidget()
        tools_widget.setLayout(tools_row)
        guide_form.addRow(tools_widget)
        self._update_guide_tool_visibility()
        guide_fields = []
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
        # (not shown any more - fixed 1.0 mm: rounded rim of the volume body)
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
        # (not shown any more - fixed 3.0 mm: smoothing of the outer surface of the volume body)
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
        self.guide_slot_extent = combo(["Whole bone", "Guide side"], "Whole bone",
                                       "Length of the slots along the cut:\n"
                                       "Whole bone: the full width of the bone cross-section in the cutting plane "
                                       "(the far cortex can stick out further than the near one).\n"
                                       "Guide side: only the width of the half of the cross-section the guide "
                                       "sits on - narrower guide; the far part of the cut is finished along the "
                                       "plane without the slot.")
        guide_fields.append(("Slot length", self.guide_slot_extent))
        self.guide_slot_margin = spin(0.0, 3.0, 1.0, 0.25, " mm",
                                      "Slot longer than the bone on each side (then the 2 mm end caps).",
                                      decimals=2)
        self.guide_slot_margin.valueChanged.connect(lambda _v: self._refresh_guide_body_if_present())
        guide_fields.append(("Slot margin", self.guide_slot_margin))
        self.guide_saw_type = combo(["Slot", "Block"], "Slot",
                                    "Slot: the blade runs in a closed slot (guided on both sides).\n"
                                    "Block: no slot - the guide ends exactly at the cutting plane and the blade "
                                    "runs along its flat face. Closing wedge: the whole wedge between the two "
                                    "faces stays open. Smaller guide, easier to use on small bones.")
        guide_fields.append(("Saw guide", self.guide_saw_type))
        self.guide_body_method = combo(["Extrude", "Volume"], "Volume",
                                       "Extrude: the painted surface is lifted along the bone normals (classic).\n"
                                       "Volume: the guide is built from the distance field of the bone - it grows "
                                       "away from the bone in every direction, so it can lie on several sides of "
                                       "the bone at once. Inner surface = bone + fit gap (exact), outer surface = "
                                       "fit gap + wall thickness, rounded rim, always closed. Saw block and "
                                       "sleeves are fused with it.")
        # (not shown any more - always 'Volume')
        self.guide_bridges = qt.QCheckBox("4 at the cut")
        self.guide_bridges.checked = False
        self.guide_bridges.setToolTip(
            "Body: Volume - four round bridges across the cut (both ends of the kerf, close to the bone and "
            "near the top of the saw block), fused into the guide on both sides: the two halves cannot "
            "twist or bend at the cut. The kerfs stay open.")
        self.guide_bridges.toggled.connect(lambda _c: self._refresh_guide_body_if_present())
        # (not shown any more - off: the saw block holds the two halves together)
        self.guide_trim_undercut = qt.QCheckBox("trim")
        self.guide_trim_undercut.checked = True
        self.guide_trim_undercut.setToolTip(
            "Body: Volume - parts of the guide lying UNDER the bone as seen from the guide side (beyond the "
            "widest point of the bone) would block placing the guide; they are cut away.")
        self.guide_trim_undercut.toggled.connect(lambda _c: self._refresh_guide_body_if_present())
        # (not shown any more - always on: nothing below the bone as seen from the guide side)
        self.guide_min_feature = qt.QDoubleSpinBox()
        self.guide_min_feature.setRange(0.0, 1.0)
        self.guide_min_feature.setDecimals(2)
        self.guide_min_feature.setSingleStep(0.05)
        self.guide_min_feature.setValue(0.3)
        self.guide_min_feature.setSuffix(" mm")
        self.guide_min_feature.setSpecialValueText("off")
        self.guide_min_feature.setToolTip(
            "Body: Volume - parts of the guide thinner than this (thin edges, slivers, feathered rims) are "
            "removed - they would break off or print badly. Material is only removed, never added; the fit "
            "surface, kerfs and bores are not changed. 0 = off.")
        self.guide_min_feature.editingFinished.connect(lambda: self._refresh_guide_body_if_present())
        # (not shown any more - fixed 0.3 mm: thinner parts of the guide are removed)
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
        self._update_guide_tool_visibility()        # again, now that the parameter fields exist
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
        self.guide_opacity_slider = self._new_bone_opacity_slider()
        guide_form.addRow("Bone opacity", self.guide_opacity_slider)


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

    def _schedule_values_refresh(self, caller=None, event=None):
        if getattr(self, "_values_refresh_pending", False):
            return
        self._values_refresh_pending = True

        def run():
            self._values_refresh_pending = False
            try:
                self._on_target_values_changed()
            except Exception:
                logging.exception("Doggybone: refreshing the calculated corrections failed (continuing)")
        qt.QTimer.singleShot(300, run)

    def _on_scene_reset(self, caller=None, event=None):
        if getattr(self, "_scene_reset_pending", False):
            return
        self._scene_reset_pending = True
        qt.QTimer.singleShot(0, lambda: slicer.util.reloadScriptedModule("Doggybone"))

    def cleanup(self):
        for node, tag in getattr(self, "_landmark_observers", []):
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._landmark_observers = []
        for tag in getattr(self, "_scene_observers", []):
            slicer.mrmlScene.RemoveObserver(tag)
        self._scene_observers = []
        self._remove_fine_shift_observers()
        self._remove_pin_picker()
        self._release_widget_observers()
        self._stop_lasso()
        self._stop_guide_edit()
        # module reload: the old landmark objects must stop listening to the (still existing)
        # landmark node - otherwise e.g. an old sphere landmark draws its sphere on a new point
        for panel in (getattr(self, "left_panel", None), getattr(self, "right_panel", None)):
            for landmark in getattr(panel, "landmarks", []) or []:
                try:
                    landmark.stop_interaction()
                    node = landmark._markups_node
                    for tag in list(getattr(landmark, "_private_observers", [])):
                        if node is not None:
                            node.RemoveObserver(tag)
                    landmark._private_observers = []
                    if hasattr(landmark, "_remove_handle_observer"):
                        landmark._remove_handle_observer()
                    landmark.change_callbacks = []
                except Exception:
                    logging.exception("Doggybone: landmark clean-up failed (continuing)")
        for name in ("Doggybone sphere preview", "Doggybone sphere radius"):
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)

    def enter(self):
        # Called by Slicer whenever the user switches to this module's tab. Also call the base
        # class's own enter(), if it defines one, in case Slicer's module-switching machinery
        # relies on it.
        if hasattr(ScriptedLoadableModuleWidget, "enter"):
            ScriptedLoadableModuleWidget.enter(self)
        self._current_panel().activate()
        self._schedule_values_refresh()          # current values / calculated corrections up to date
        self._install_pin_picker()               # reduction guide: click a pin to select / move it
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
        self._remove_pin_picker()
        self._stop_remove_bone()
        editor = getattr(self, "_cleanup_editor_widget", None)
        if editor is not None:
            editor.setActiveEffect(None)
            editor.removeViewObservations()
        self._current_panel().deactivate()

    def _stop_remove_bone(self):
        """'Remove bone (click)' off - its 3D-view observer must never outlive the module tab / widget."""
        button = getattr(self, "remove_bone_button", None)
        if button is not None and button.checked:
            button.setChecked(False)                  # removes the observer (onRemoveBoneToggled)
        elif getattr(self, "_remove_bone_tag", None) is not None:
            try:
                slicer.app.layoutManager().threeDWidget(0).threeDView().interactor().RemoveObserver(
                    self._remove_bone_tag)
            except Exception:
                pass
            self._remove_bone_tag = None

    def _release_widget_observers(self):
        """Module reload / close: everything this widget instance observes on nodes that stay in the
        scene (gizmos, handles) or on the views is released - otherwise the OLD widget keeps reacting
        next to the new one (e.g. a pin or saw box moved twice)."""
        self._stop_remove_bone()
        timer = getattr(self, "_guide_refresh_timer", None)
        if timer is not None:
            timer.stop()                              # no delayed rebuild by the old widget
        try:
            self._remove_pin_gizmo()
        except Exception:
            pass
        # pin-row handles
        tag = getattr(self, "_pin_row_handle_tag", None)
        node = slicer.mrmlScene.GetNodeByID(getattr(self, "_pin_row_handle_node_id", "") or "")
        if tag is not None and node is not None:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._pin_row_handle_tag = None
        # saw-box gizmos
        for node_id, tag in list(getattr(self, "_saw_box_gizmo_tags", {}).values()):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                try:
                    node.RemoveObserver(tag)
                except Exception:
                    pass
        self._saw_box_gizmo_tags = {}
        # embedded segment editor: stop observing views / segmentation
        editor = getattr(self, "_cleanup_editor_widget", None)
        if editor is not None:
            try:
                editor.setActiveEffect(None)
                editor.removeViewObservations()
                editor.setSegmentationNode(None)
            except Exception:
                pass

    def _current_panel(self):
        return self.side_stack.currentWidget()

    def _on_side_changed(self, index):
        old_panel = self._current_panel()          # the stack still shows the PREVIOUS side here
        if old_panel is not self.side_stack.widget(index):
            old_panel.deactivate()
        self.side_stack.setCurrentIndex(index)
        self._current_panel().activate()

    def _on_osteotomy_bone_changed(self, index):
        bone = self.osteotomy_bone.currentText
        # Reset the target fields to this bone's default whenever the bone selection changes -
        # a target left over from the other bone (e.g. 90° carried over from Femur to Tibia)
        # would silently produce a wrong correction magnitude.
        # Targets are kept PER BONE: switching femur <-> tibia restores that bone's last targets
        # (first time: the bone's defaults) - never a value carried over from the other bone.
        stored = getattr(self, "_targets_per_bone", {})
        previous = getattr(self, "_target_bone", None)
        if previous is not None and previous != bone:
            stored[previous] = (float(self.target_wedge_value.value), float(self.target_rotation_value.value))
        self._targets_per_bone = stored
        self._target_bone = bone
        wedge, rotation = stored.get(bone, (self.WEDGE_TARGET_DEFAULT[bone], self.ROTATION_TARGET_DEFAULT[bone]))
        for box, value in ((self.target_wedge_value, wedge), (self.target_rotation_value, rotation)):
            was_blocked = box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(was_blocked)
        self._on_target_values_changed()
        self._set_tibia_mode_visible(bone == "Tibia")
        self._update_plane_handle_interactivity()
        if hasattr(self, "_on_guide_bone_switched") and hasattr(self, "guide_status"):
            self._on_guide_bone_switched()

    def _on_technique_changed(self):
        """Switched to CWO with two almost identical planes: recommend an OWO right away."""
        if not hasattr(self, "correction_status") or not hasattr(self, "_cwo_wedge_advice"):
            return
        side = self.osteotomy_side.currentText.lower()
        advice = self._cwo_wedge_advice(side, self.osteotomy_bone.currentText)
        if advice:
            self.correction_status.setText(advice)

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
        if bone == "Tibia" and hasattr(self, "_refresh_tibia_current"):
            self._refresh_tibia_current()

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

    # --- ONE bone opacity for the whole module: the slider in every section shows and sets the
    # same value (bone segmentation and the preview fragments / corrected bone alike) -------------
    def _new_bone_opacity_slider(self):
        slider = qt.QSlider(qt.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(int(getattr(self, "opacity_slider", None).value) if getattr(self, "opacity_slider", None) else 70)
        slider.setToolTip(BONE_OPACITY_TIP)
        slider.valueChanged.connect(self._set_bone_opacity)
        return slider

    def _bone_opacity_sliders(self):
        return [s for s in (getattr(self, name, None) for name in
                ("opacity_slider", "measurement_opacity_slider", "segment_opacity_slider", "guide_opacity_slider"))
                if s is not None]

    def _set_bone_opacity(self, value):
        value = int(value)
        for slider in self._bone_opacity_sliders():
            if slider.value != value:
                was = slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(was)
        self.onOpacityChanged(value)                 # bone segmentation(s)
        if hasattr(self, "_apply_segment_opacity"):
            self._apply_segment_opacity()            # preview fragments + corrected bone

    def onHideAnnotationsToggled(self, hide):
        """Hide / show all labels and the cutting planes; the previous state is restored exactly."""
        if hide:
            saved = []
            for node in slicer.util.getNodesByClass("vtkMRMLMarkupsNode"):
                name = node.GetName() or ""
                display = node.GetDisplayNode()
                if display is None or not name.startswith("Doggybone"):
                    continue
                is_plane = node.IsA("vtkMRMLMarkupsPlaneNode") or name.endswith(" marker")
                if is_plane:
                    saved.append((node.GetID(), "visibility", bool(display.GetVisibility())))
                    display.SetVisibility(False)
                else:
                    saved.append((node.GetID(), "point labels", bool(display.GetPointLabelsVisibility())))
                    display.SetPointLabelsVisibility(False)
                if hasattr(display, "GetPropertiesLabelVisibility"):
                    saved.append((node.GetID(), "properties label", bool(display.GetPropertiesLabelVisibility())))
                    display.SetPropertiesLabelVisibility(False)
            self._hidden_annotations = saved
            return
        for node_id, kind, value in getattr(self, "_hidden_annotations", []):
            node = slicer.mrmlScene.GetNodeByID(node_id)
            display = node.GetDisplayNode() if node is not None else None
            if display is None:
                continue
            if kind == "visibility":
                display.SetVisibility(value)
            elif kind == "point labels":
                display.SetPointLabelsVisibility(value)
            else:
                display.SetPropertiesLabelVisibility(value)
        self._hidden_annotations = []

    def onAlignSliceViews(self):
        choice = self.view_align_selector.currentText
        layout_manager = slicer.app.layoutManager()
        slice_nodes = {name: layout_manager.sliceWidget(name).mrmlSliceNode()
                       for name in ("Red", "Green", "Yellow") if layout_manager.sliceWidget(name) is not None}
        if choice == "Scanner":
            for name, node in slice_nodes.items():
                {"Red": node.SetOrientationToAxial, "Green": node.SetOrientationToCoronal,
                 "Yellow": node.SetOrientationToSagittal}[name]()
            return
        bone = "Femur" if "femur" in choice else "Tibia"
        side = self.side_selector.currentText.lower()
        origin, shaft, error = self._get_anatomical_axis(side, bone)
        if error:
            slicer.util.showStatusMessage(f"Slice alignment: {error}", 5000)
            self.view_align_selector.setToolTip(f"Not aligned: {error}")
            return
        shaft = osteotomy_logic.normalize(np.asarray(shaft, dtype=float))     # proximal -> distal
        ml_vector = self._get_ml_vector(side, bone)
        if ml_vector is None or np.linalg.norm(ml_vector) < 1e-6:
            ml_vector = np.array([1.0, 0.0, 0.0])                            # rotation unknown yet
        ml = np.asarray(ml_vector, dtype=float) - float(np.dot(ml_vector, shaft)) * shaft
        if np.linalg.norm(ml) < 1e-6:
            ml = np.array([0.0, 1.0, 0.0]) - shaft[1] * shaft
        ml = osteotomy_logic.normalize(ml)                                     # medial -> lateral
        if side == "right":
            ml = -ml                                                           # patient left = screen right
        up = -shaft                                                            # proximal
        ap = osteotomy_logic.normalize(np.cross(ml, up))                       # cranial
        # centre: on the shaft line, at the level of the crosshair (so 'Update' keeps the level you
        # scrolled to), limited to the bone between the two joints
        crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        level = 0.0
        if crosshair is not None:
            level = float(np.dot(np.array(crosshair.GetCrosshairRAS(), dtype=float) - origin, shaft))
        panel = self.left_panel if side == "left" else self.right_panel
        ends = [float(np.dot(np.asarray(l.get_position(), dtype=float) - origin, shaft))
                for l in panel.landmarks if l.placed]
        if ends:
            level = min(max(level, min(ends)), max(ends))
        centre = np.asarray(origin, dtype=float) + level * shaft
        # (normal, screen-right); screen-up = normal x screen-right. As in Slicer's standard views:
        # red  - screen up = cranial, right = patient left; green - up = proximal, right = patient
        # left; yellow - up = proximal, cranial on the left
        frames = {"Red": (-up, ml), "Green": (ap, ml), "Yellow": (ml, -ap)}
        for name, (normal, transverse) in frames.items():
            node = slice_nodes.get(name)
            if node is None:
                continue
            node.SetSliceToRASByNTP(*[float(v) for v in normal], *[float(v) for v in transverse],
                                    *[float(v) for v in centre], 0)
        if crosshair is not None:
            crosshair.SetCrosshairRAS(*[float(v) for v in centre])
        self.view_align_selector.setToolTip(
            f"Aligned to the {side} {bone.lower()}: red perpendicular to the shaft, green frontal, "
            "yellow sagittal. Scroll the red view to move along the bone.")
        slicer.util.showStatusMessage(f"Slice views aligned to the {side} {bone.lower()}.", 4000)

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
        # normal scene-generated ID (an ID with spaces breaks node references in saved scenes)
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", self.SEGMENTATION_NODE_NAME)
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
        self.AXIS_PALETTE = [(0.15, 0.85, 0.95), (0.95, 0.35, 0.85), (0.95, 0.65, 0.15), (0.55, 0.85, 0.25),
                             (1.0, 0.25, 0.25), (0.35, 0.45, 1.0)]

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
                                    if _csv_decimal() == "," else ",", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for landmark in self.landmarks:
                if landmark.placed:
                    position = landmark.get_position()
                    radius = getattr(landmark, "radius", None)
                    writer.writerow({"landmark name": landmark.name, 
                                     "x": _csv_number(position[0]), 
                                     "y": _csv_number(position[1]), 
                                     "z": _csv_number(position[2]),
                                     "radius": _csv_number(radius) if radius is not None else ""})


    def _import_landmarks(self):
        file_name = qt.QFileDialog.getOpenFileName(self, 'Import landmarks', '',"CSV File (*.csv)")
        if file_name == "":
            return
        landmark_dict = {lm.name: lm for lm in self.landmarks}
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=_csv_delimiter_for_file(csvfile),
                                    quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                try:
                    name = row['landmark name']
                    x = _csv_float(row['x'])
                    y = _csv_float(row['y'])
                    z = _csv_float(row['z'])
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
                        landmark.set_radius(_csv_float(radius_text))
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
                                    if _csv_decimal() == "," else ",", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for i in range(self.measurement_stack.count):
                measurement = self.measurement_stack.widget(i).measurement
                result_ready, result_value, result_string = measurement()
                if result_ready:
                    writer.writerow({"measurement": measurement.name, 
                                     "value": _csv_number(result_value), 
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
        if slicer.mrmlScene.GetNodeByID(self.markup_node_id) is None:
            # the landmark list was deleted with the scene - rebuild the module (new lists)
            if self.base_widget is not None and hasattr(self.base_widget, "_on_scene_reset"):
                self.base_widget._on_scene_reset()
            return
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