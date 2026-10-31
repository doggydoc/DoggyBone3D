# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

from DoggyboneLib.measurement_logic import *

MEASUREMENTS = [
    AntetorsionMeasurement,
    VarusValgusFemurMeasurement,
    VarusValgusTibiaMeasurement,
    TibialPlateauAngleMeasurement,
    TibiaTorsionMeasurement,
    FemorotibialRotationMeasurement,
    TibiotalarRotationMeasurement,
    TibialMetatarsalAngleMeasurement
]
