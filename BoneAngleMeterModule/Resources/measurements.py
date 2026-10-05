# Measurement methods shared with Doggybone 3D (Matthias Frank, Kleintierklinik Frank & Partner, Freiburg)
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License.

from Resources.measurement_logic import *

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
