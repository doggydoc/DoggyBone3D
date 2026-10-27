# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

"""Per-bone configuration.

Everything the planning, correction, pin and guide code needs to know about ONE bone that can be
osteotomised (which landmarks span its shaft axis, which measurements feed the correction, which
fragment is the anatomical reference, how its scene objects and export files are named ...).
The generic code asks a BoneConfig instead of testing for a bone name; each limb module supplies
its own list of bones (see bones.py)."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class BoneConfig:
    # Name shown in the bone selector; also the key stored on guide nodes ("Doggybone.GuideBone").
    name: str
    # Appended to every scene node name of this bone's guide objects. Exactly one bone per limb may
    # use "" (the plain names); keeps scenes saved with earlier versions loadable.
    node_tag: str
    # Bone part of exported file names ("fem") and of the label embossed on the guide ("FEM").
    file_code: str
    label_code: str
    # (proximal, distal) landmark spanning the anatomical shaft axis.
    shaft_axis_landmarks: Tuple[str, str]
    # (medial, lateral) landmark pair giving the medio-lateral reference direction at the cut.
    ml_landmarks: Tuple[str, str]
    # Landmarks that certainly lie proximal of any cut - fallback for the proximal direction.
    proximal_fallback_landmarks: Tuple[str, ...]
    # Measurements feeding the wedge (frontal) and rotation correction, by measurement name.
    wedge_measurement: str
    rotation_measurement: str
    # Default targets ("Zielwert") of these two measurements after correction.
    wedge_target_default: float
    rotation_target_default: float          # None: no reference value - start with the current value
    # Description prefix of the rotation measurement that counts as the POSITIVE direction
    # (e.g. "Antetorsion", "Internal").
    rotation_positive_prefix: str
    # Fragment that is the anatomical reference ("condylar" fragment): its pins stand
    # perpendicular to the condylar axis, the other fragment's pins are pre-angled. "proximal"/"distal".
    reference_end: str
    # Lower-case word identifying this bone's landmarks by name (landmark transfer between sides).
    landmark_keyword: str
    # Optional features of the generic UI.
    sagittal_input: bool = False          # manual sagittal correction field (femur procurvatum)
    target_table: bool = False            # per-angle target table instead of the two target fields
    start_plane_toward_ml_landmarks: Optional[float] = None   # initial plane level, fraction from the
                                                              # distal shaft landmark to the ML pair
    sagittal_target_measurement: str = ""   # sagittal field = TARGET of this measurement's value (else: manual angle)
    sagittal_target_key: str = ""           # key in that measurement's .values
    partial_axes: bool = False            # shaft landmarks span the whole bone (two partial axes) -
                                          # every landmark moves with ITS side of the cut (no split-axis rule)
    max_osteotomies: int = 1
    # Label and tooltip of the frontal target field (what the wedge measurement's value is).
    wedge_target_label: str = "Varus/valgus target"
    wedge_target_tooltip: str = ""
    # Measurements shown when this bone is selected (by measurement name); empty = all.
    measurements: Tuple[str, ...] = ()

    def signed_rotation(self, value, description):
        """Magnitude + description of the rotation measurement -> signed value (+ = positive direction)."""
        return value if (description or "").startswith(self.rotation_positive_prefix) else -value

    @property
    def opposite_end(self):
        return "proximal" if self.reference_end == "distal" else "distal"


class BoneSet:
    """The bones of one limb module, in selector order."""

    def __init__(self, bones):
        self.bones = tuple(bones)
        if not self.bones:
            raise ValueError("BoneSet needs at least one bone")
        names = [bone.name for bone in self.bones]
        if len(set(names)) != len(names):
            raise ValueError("bone names must be unique")
        plain = [bone for bone in self.bones if not bone.node_tag]
        if len(plain) > 1:
            raise ValueError("at most one bone may use the plain (untagged) node names")

    def __iter__(self):
        return iter(self.bones)

    def __len__(self):
        return len(self.bones)

    @property
    def names(self):
        return [bone.name for bone in self.bones]

    @property
    def default(self):
        return self.bones[0]

    def get(self, name):
        for bone in self.bones:
            if bone.name == name:
                return bone
        raise KeyError(f"unknown bone '{name}'")

    def order(self, name):
        """Sort key: position in the selector (unknown names last)."""
        for index, bone in enumerate(self.bones):
            if bone.name == name:
                return index
        return len(self.bones)

    def _fallback(self):
        # the bone with the plain names, else the last one
        for bone in self.bones:
            if not bone.node_tag:
                return bone
        return self.bones[-1]

    def for_node_name(self, node_name):
        """Bone whose guide objects carry this scene node name."""
        node_name = node_name or ""
        for bone in self.bones:
            if bone.node_tag and bone.node_tag in node_name:
                return bone
        return self._fallback()

    def for_tag(self, tag):
        """Bone for a node tag matched in a node name ('' or None = plain names)."""
        tag = tag or ""
        for bone in self.bones:
            if bone.node_tag == tag:
                return bone
        return self._fallback()

    def for_landmark(self, landmark_name):
        """Bone a landmark belongs to, by keyword; landmarks without a keyword go to the LAST bone
        (e.g. talus / metatarsal landmarks with the tibia)."""
        lowered = (landmark_name or "").lower()
        for bone in self.bones:
            if bone.landmark_keyword and bone.landmark_keyword in lowered:
                return bone
        return self.bones[-1]

    def for_text(self, text):
        """Bone named in a free text (e.g. 'Aligned to tibia'); the first bone if none is named."""
        lowered = (text or "").lower()
        for bone in self.bones:
            if bone.name.lower() in lowered:
                return bone
        return self.bones[0]

    def others(self, name):
        return [bone for bone in self.bones if bone.name != name]

    def tag_pattern(self):
        """Regex alternative matching any non-empty node tag, for use in a group: '( tibia)?'."""
        import re
        tags = [re.escape(bone.node_tag) for bone in self.bones if bone.node_tag]
        return "(" + "|".join(tags) + ")?" if tags else "()?"
