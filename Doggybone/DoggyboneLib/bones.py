# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""Bones of the hind limb that can be osteotomised (see bone_config.py)."""

from DoggyboneLib import measurement_logic
from DoggyboneLib.bone_config import BoneConfig, BoneSet

FEMUR = BoneConfig(
    name="Femur",
    node_tag="",                                   # plain node names (scenes from earlier versions)
    file_code="fem",
    label_code="FEM",
    shaft_axis_landmarks=("Femur midpoint 33%", "Femur midpoint 50%"),
    ml_landmarks=("medial femur condyle", "lateral femur condyle"),
    proximal_fallback_landmarks=("femur head", "femur neck", "Femur midpoint 33%"),
    wedge_measurement="aLDFA (Varus/Valgus Femur)",
    rotation_measurement="Antetorsion",
    wedge_target_default=94.0,                     # aLDFA target (canine reference ~94°)
    # Placeholder - no reliable breed-specific reference for normal antetorsion; the vet sets it.
    rotation_target_default=25.0,
    rotation_positive_prefix="Antetorsion",
    reference_end="distal",                        # condyles
    landmark_keyword="femur",
    sagittal_input=True,
    measurements=("Antetorsion", "aLDFA (Varus/Valgus Femur)", "Femorotibial Rotation"),
    # start the plane distal to the 50 % landmark, halfway to the condyles (same shaft line)
    start_plane_toward_ml_landmarks=0.5,
)

TIBIA = BoneConfig(
    name="Tibia",
    node_tag=" tibia",
    file_code="tib",
    label_code="TIB",
    shaft_axis_landmarks=("proximal tibia midpoint", "distal tibia midpoint"),
    ml_landmarks=("condylus medialis tibiae", "condylus lateralis tibiae"),
    proximal_fallback_landmarks=("intercondylar eminence", "proximal tibia midpoint", "condylus medialis tibiae"),
    wedge_measurement="Varus Valgus Tibia",
    rotation_measurement="Tibia Torsion",
    # frontal angle between the two joint lines (mMPTA + mMDTA - 180, torsion-free)
    wedge_target_default=round(measurement_logic.TIBIA_MPTA_REF_DEG + measurement_logic.TIBIA_MDTA_REF_DEG - 180.0, 1),
    # Placeholder - no reliable reference for normal tibial torsion; the vet sets it.
    rotation_target_default=0.0,
    rotation_positive_prefix="Internal",
    reference_end="proximal",                      # tibial condyles
    landmark_keyword="tibia",
    target_table=True,
    measurements=("Varus Valgus Tibia", "Tibial Plateau Angle", "Tibia Torsion", "Femorotibial Rotation",
                  "Tibiotalar Rotation", "Tibial Metatarsal Angle"),
)

BONES = BoneSet([FEMUR, TIBIA])
