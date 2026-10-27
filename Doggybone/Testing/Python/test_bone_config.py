# DoggyBone3D - bone configuration tests (no GUI, no VTK):
#   Slicer:  exec(open(r"<path>/test_bone_config.py").read())      or
#   shell:   python3 -m unittest <path>/test_bone_config.py
#
# The hind-limb bone configuration replaced fixed "Femur" / "Tibia" branches in the widget. These
# tests pin the configuration to the behaviour of those former branches (written out as oracles
# below), so the hind limb keeps working exactly as before.
import os, re, sys, unittest

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from DoggyboneLib import measurement_logic
from DoggyboneLib.bone_config import BoneConfig, BoneSet
from DoggyboneLib.bones import BONES, FEMUR, TIBIA
from DoggyboneLib.measurements import MEASUREMENTS
try:                                   # landmark classes need Slicer
    from DoggyboneLib.landmarks import LANDMARKS
    LANDMARK_NAMES = [landmark.name for landmark in LANDMARKS]
except ImportError:
    LANDMARK_NAMES = None


# ---- the former fixed branches -------------------------------------------------------------------
def old_tag(bone):
    return " tibia" if bone == "Tibia" else ""


def old_signed(bone, value, description):
    if bone == "Femur":
        return value if description == "Antetorsion" else -value
    return value if description.startswith("Internal") else -value


def old_landmark_bone(name):
    return "Femur" if "femur" in (name or "").lower() else "Tibia"


OLD_PATTERN = re.compile(r"^Doggybone guide body( tibia)?( \(reduction\))?$")


class BoneConfigTest(unittest.TestCase):

    def test_selector_order_and_lookup(self):
        self.assertEqual(BONES.names, ["Femur", "Tibia"])
        self.assertIs(BONES.default, FEMUR)
        self.assertIs(BONES.get("Tibia"), TIBIA)
        with self.assertRaises(KeyError):
            BONES.get("Unknown bone")

    def test_former_constants(self):
        self.assertEqual(FEMUR.wedge_measurement, "aLDFA (Varus/Valgus Femur)")
        self.assertEqual(TIBIA.wedge_measurement, "Varus Valgus Tibia")
        self.assertEqual(FEMUR.rotation_measurement, "Antetorsion")
        self.assertEqual(TIBIA.rotation_measurement, "Tibia Torsion")
        self.assertEqual(FEMUR.wedge_target_default, 94.0)
        self.assertEqual(TIBIA.wedge_target_default, round(
            measurement_logic.TIBIA_MPTA_REF_DEG + measurement_logic.TIBIA_MDTA_REF_DEG - 180.0, 1))
        self.assertEqual((FEMUR.rotation_target_default, TIBIA.rotation_target_default), (25.0, 0.0))
        self.assertEqual(FEMUR.ml_landmarks, ("medial femur condyle", "lateral femur condyle"))
        self.assertEqual(TIBIA.ml_landmarks, ("condylus medialis tibiae", "condylus lateralis tibiae"))
        self.assertEqual(FEMUR.shaft_axis_landmarks, ("Femur midpoint 33%", "Femur midpoint 50%"))
        self.assertEqual(TIBIA.shaft_axis_landmarks, ("proximal tibia midpoint", "distal tibia midpoint"))
        self.assertEqual(FEMUR.proximal_fallback_landmarks, ("femur head", "femur neck", "Femur midpoint 33%"))
        self.assertEqual(TIBIA.proximal_fallback_landmarks,
                         ("intercondylar eminence", "proximal tibia midpoint", "condylus medialis tibiae"))
        self.assertEqual((FEMUR.reference_end, TIBIA.reference_end), ("distal", "proximal"))
        self.assertEqual((FEMUR.opposite_end, TIBIA.opposite_end), ("proximal", "distal"))
        self.assertEqual((FEMUR.file_code, TIBIA.file_code), ("fem", "tib"))
        self.assertEqual((FEMUR.label_code, TIBIA.label_code), ("FEM", "TIB"))
        self.assertEqual((FEMUR.sagittal_input, TIBIA.sagittal_input), (True, False))
        self.assertEqual((FEMUR.target_table, TIBIA.target_table), (False, True))
        self.assertEqual(FEMUR.start_plane_toward_ml_landmarks, 0.5)
        self.assertIsNone(TIBIA.start_plane_toward_ml_landmarks)
        self.assertEqual((FEMUR.max_osteotomies, TIBIA.max_osteotomies), (1, 1))

    def test_referenced_measurements_exist(self):
        measurement_names = {m().name for m in MEASUREMENTS}
        for bone in BONES:
            for name in (bone.wedge_measurement, bone.rotation_measurement) + tuple(bone.measurements):
                self.assertIn(name, measurement_names, f"{bone.name}: unknown measurement '{name}'")

    @unittest.skipIf(LANDMARK_NAMES is None, "needs Slicer")
    def test_referenced_landmarks_exist(self):
        for bone in BONES:
            for name in bone.shaft_axis_landmarks + bone.ml_landmarks + bone.proximal_fallback_landmarks:
                self.assertIn(name, LANDMARK_NAMES, f"{bone.name}: unknown landmark '{name}'")

    def test_every_measurement_belongs_to_a_bone(self):
        shown = {name for bone in BONES for name in bone.measurements}
        self.assertEqual(shown, {m().name for m in MEASUREMENTS})

    def test_node_tags(self):
        for bone in ("Femur", "Tibia"):
            self.assertEqual(BONES.get(bone).node_tag, old_tag(bone))
        names = ["Doggybone guide body", "Doggybone guide body tibia", "Doggybone guide pins",
                 "Doggybone guide pins tibia (reduction)", "Doggybone saw box tibia A", "Doggybone saw box B",
                 "Doggybone pin row handles tibia"]
        for name in names:
            self.assertEqual(BONES.for_node_name(name).name, "Tibia" if " tibia" in name else "Femur", name)

    def test_guide_body_pattern(self):
        pattern = re.compile(r"^Doggybone guide body" + BONES.tag_pattern() + r"( \(reduction\))?$")
        for name in ["Doggybone guide body", "Doggybone guide body tibia", "Doggybone guide body (reduction)",
                     "Doggybone guide body tibia (reduction)", "Doggybone guide body femur",
                     "Doggybone guide body tibia x", "Doggybone guide bodies"]:
            old, new = OLD_PATTERN.match(name), pattern.match(name)
            self.assertEqual(bool(old), bool(new), name)
            if old:
                self.assertEqual(old.groups(), new.groups(), name)
                old_bone = "Tibia" if old.group(1) else "Femur"
                self.assertEqual(BONES.for_tag(new.group(1)).name, old_bone, name)

    def test_signed_rotation(self):
        descriptions = ["Antetorsion", "Retrotorsion", "Internal torsion", "External torsion",
                        "Internal rotation", "External rotation", ""]
        for bone in ("Femur", "Tibia"):
            for description in descriptions:
                for value in (0.0, 7.5, -3.0):
                    self.assertEqual(BONES.get(bone).signed_rotation(value, description),
                                     old_signed(bone, value, description), (bone, description, value))

    def test_landmark_bone(self):
        names = list(LANDMARK_NAMES or [])
        names += list(FEMUR.shaft_axis_landmarks + FEMUR.ml_landmarks + FEMUR.proximal_fallback_landmarks
                      + TIBIA.shaft_axis_landmarks + TIBIA.ml_landmarks + TIBIA.proximal_fallback_landmarks)
        for name in names:
            self.assertEqual(BONES.for_landmark(name).name, old_landmark_bone(name), name)
        for name in ("", None, "Metatarsal centre", "Talus centre", "medial talus", "lateral cochlea"):
            self.assertEqual(BONES.for_landmark(name).name, old_landmark_bone(name))

    def test_view_align_choice(self):
        for choice in ("Aligned to femur", "Aligned to tibia"):
            self.assertEqual(BONES.for_text(choice).name, "Femur" if "femur" in choice else "Tibia")

    def test_sort_order(self):
        guides = [("Tibia", "reduction"), ("Femur", "reduction"), ("Tibia", "saw"), ("Femur", "saw")]
        old = sorted(guides, key=lambda item: (item[0] != "Femur", item[1] != "saw"))
        new = sorted(guides, key=lambda item: (BONES.order(item[0]), item[1] != "saw"))
        self.assertEqual(old, new)

    def test_boneset_validation(self):
        plain = BoneConfig("A", "", "a", "A", ("p", "d"), ("m", "l"), ("p",), "w", "r", 0.0, 0.0,
                           "Internal", "proximal", "a")
        with self.assertRaises(ValueError):
            BoneSet([])
        with self.assertRaises(ValueError):
            BoneSet([plain, plain])
        other_plain = BoneConfig("B", "", "b", "B", ("p", "d"), ("m", "l"), ("p",), "w", "r", 0.0, 0.0,
                                 "Internal", "proximal", "b")
        with self.assertRaises(ValueError):
            BoneSet([plain, other_plain])


if __name__ == "__main__":
    unittest.main()
