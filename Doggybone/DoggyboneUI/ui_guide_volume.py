# Doggybone 3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""GuideVolumeMixin: 'Body: Volume' - the guide body built from the distance field of the bone
(guide_logic.sdf_guide_body) instead of extruding the painted surface."""

from DoggyboneUI.common import *


class GuideVolumeMixin:

    def _guide_body_is_volume(self):
        box = getattr(self, "guide_body_method", None)
        return box is not None and box.currentText.startswith("Volume")
