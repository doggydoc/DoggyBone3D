import math
import numpy as np

from Resources.helpers import *

def sphere_landmark(point_dict, radius_dict, label):
    '''
    Returns (center, radius) for a manually-placed sphere landmark: the landmark's own placed
    position as the center, and its user-adjusted radius. Used wherever visually matching a
    sphere against the bone surface (femur head, femoral condyles) is more practical than an
    automatic best-fit from several points.
    '''
    return point_dict[label], radius_dict[label]

class BaseMeasurement:
    '''
    Base class for all measurements. Child classes need to implement the _measure method that takes a
    dictionary of landmark positions and returns a float value and a string description. Calls to the
    measurement should be done via ().
    '''
    # Names of any fitted spheres this measurement computes (e.g. "femur head"), for subclasses
    # that fit one. Used by the GUI to know which spheres to show/hide, independent of whether
    # a fit has actually been computed yet.
    SPHERE_LABELS = []

    # Descriptive labels of the two anatomical axes this measurement compares (e.g. "Tibial
    # axis", "Talar axis"), in the same order they're compared. Used by the GUI to know which
    # axis lines to show/hide, independent of whether they've actually been computed yet.
    AXIS_LABELS = []

    # Optional display/placement order of this measurement's landmarks in the GUI. Names not
    # listed keep their natural (first-queried) order after the listed ones. Does not affect the
    # measurement itself.
    LANDMARK_ORDER = []

    def __init__(self, name):
        self.name = name
        self.side = None
        self.landmarks = []
        self.description = ""
        self.spheres = {}  # label -> (center, radius), populated by _measure on success
        self.axes = {}  # label -> (start_point, end_point), populated by _measure on success

    def get_spheres(self):
        return self.spheres

    def get_axes(self):
        return self.axes

    def set_side(self, side):
        self.side = side

    def register_landmarks(self, landmarks):
        # Hacky way to find all used landmark names
        class FakeDict:
            def __init__(self):
                self.queried_keys = []

            def __getitem__(self, key):
                self.queried_keys.append(key)
                return np.random.rand(3)

        fake_dict = FakeDict()
        self._measure(fake_dict, fake_dict)
        # _measure legitimately reads some landmarks more than once (e.g. once to build a
        # reference "normal" vector, again for the actual comparison vector), so dedupe while
        # preserving first-seen order - otherwise a landmark referenced N times would be
        # registered and listed N times.
        used_landmark_names = list(dict.fromkeys(fake_dict.queried_keys))
        if self.LANDMARK_ORDER:
            ordered = [name for name in self.LANDMARK_ORDER if name in used_landmark_names]
            used_landmark_names = ordered + [name for name in used_landmark_names if name not in ordered]

        # Register landmark objects
        for name in used_landmark_names:
            landmark_found = False
            for landmark in landmarks:
                if landmark.name == name:
                    self.landmarks.append(landmark)
                    landmark_found = True
                    break
            if not landmark_found:
                raise ValueError(f"Could not register landmark with name {name}")

        # Register update callback for all landmarks
        for landmark in self.landmarks:
            landmark.add_change_callback(self.maybe_update)

    def get_landmarks(self):
        return self.landmarks

    def maybe_update(self):
        pass

    def __call__(self):
        # Check if all landmarks were placed
        for landmark in self.landmarks:
            if not landmark.placed:
                return False, None, "Not all landmarks defined"
        point_dict = {l.name: l.get_position() for l in self.landmarks}
        radius_dict = {l.name: getattr(l, "radius", None) for l in self.landmarks}
        angle, message = self._measure(point_dict, radius_dict)
        return True, angle, message

class TibiaTorsionMeasurement(BaseMeasurement):
    AXIS_LABELS = ["Distal cochlear axis", "Proximal tibial condylar axis"]

    def __init__(self):
        super().__init__('Tibia Torsion')
    
    def _measure(self, point_dict, radius_dict):
        normal = vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])
        vector_distal = vector_with_two_points(point_dict["medial cochlea"], point_dict["lateral cochlea"])
        vector_proximal = vector_with_two_points(point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"])
        self.axes = {
            "Distal cochlear axis": (point_dict["medial cochlea"], point_dict["lateral cochlea"]),
            "Proximal tibial condylar axis": (point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"]),
        }
        
        vector_distal_proj = project_vector_to_plane_from_normal(normal, vector_distal)
        vector_proximal_proj = project_vector_to_plane_from_normal(normal, vector_proximal)
        
        a = angle(vector_distal_proj, vector_proximal_proj)*180/math.pi
        
        t = np.cross(vector_proximal_proj, vector_distal_proj)
        q = np.dot(t,normal)
        
        if self.side.lower() == "right":
            return a, "Innenrotation" if q > 0 else "Aussenrotation"
        elif self.side.lower() == "left":
            return a, "Aussenrotation" if q > 0 else "Innenrotation"
        else:
            raise ValueError(f"Unknown side {self.side}")

class VarusValgusTibiaMeasurement(BaseMeasurement):
    AXIS_LABELS = ["Tibial shaft axis", "Proximal tibial condylar axis", "Cochlear articulation axis", "Condylar articulation axis"]

    def __init__(self):
        super().__init__('Varus Valgus Tibia')
        self.description = "TEST"
    
    def _measure(self, point_dict, radius_dict):
        normal = np.cross((vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])),(vector_with_two_points(point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"])))
        vector_proximal_tibia_vv = vector_with_two_points(point_dict["lateral cochlea articulation point tibia"], point_dict["medial cochlea articulation point tibia"])
        vector_distal_tibia_vv = vector_with_two_points(point_dict["lateral condyle articulation point tibia"], point_dict["medial condyle articulation point tibia"])
        self.axes = {
            "Tibial shaft axis": (point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"]),
            "Proximal tibial condylar axis": (point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"]),
            "Cochlear articulation axis": (point_dict["lateral cochlea articulation point tibia"], point_dict["medial cochlea articulation point tibia"]),
            "Condylar articulation axis": (point_dict["lateral condyle articulation point tibia"], point_dict["medial condyle articulation point tibia"]),
        }

        # This is a bending angle (not a rotation/torsion), so - same reasoning as the femur
        # varus/valgus fix - the raw 3D angle is used directly rather than first projecting onto
        # a plane. normal is still used below for the side-direction check.
        a = angle(vector_proximal_tibia_vv, vector_distal_tibia_vv)*180/math.pi
        
        t = np.cross(vector_proximal_tibia_vv, vector_distal_tibia_vv)
        q = np.dot(t,normal)
        if self.side.lower() == "right":
            return a, "Varus" if q > 0 else "Valgus"
        elif self.side.lower() == "left":
            return a, "Valgus" if q > 0 else "Varus"
        else:
            raise ValueError(f"Unknown side {self.side}")

class TibiotalarRotationMeasurement(BaseMeasurement):
    AXIS_LABELS = ["Distal tibial axis", "Talar axis"]

    def __init__(self):
        super().__init__("Tibiotalar Rotation")
    
    def _measure(self, point_dict, radius_dict):
        normal = vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])
        vector_distaltibia = vector_with_two_points(point_dict["medial cochlea"], point_dict["lateral cochlea"])
        vector_talus = vector_with_two_points(point_dict["medial talus"], point_dict["lateral talus"])
        self.axes = {
            "Distal tibial axis": (point_dict["medial cochlea"], point_dict["lateral cochlea"]),
            "Talar axis": (point_dict["medial talus"], point_dict["lateral talus"]),
        }
        
        vector_distaltibia_proj = project_vector_to_plane_from_normal(normal, vector_distaltibia)
        vector_talus_proj = project_vector_to_plane_from_normal(normal, vector_talus)
        
        
        a = angle(vector_distaltibia_proj, vector_talus_proj)*180/math.pi
        
        t = np.cross(vector_distaltibia_proj, vector_talus_proj)
        q = np.dot(t,normal)
        
        if self.side.lower() == "right":
            return a, "Innenrotation" if q > 0 else "Aussenrotation"
        elif self.side.lower() == "left":
            return a, "Aussenrotation" if q > 0 else "Innenrotation"
        else:
            raise ValueError(f"Unknown side {self.side}")

class FemorotibialRotationMeasurement(BaseMeasurement):
    SPHERE_LABELS = ["medial femur condyle", "lateral femur condyle"]
    AXIS_LABELS = ["Femoral condylar axis", "Proximal tibial axis"]

    def __init__(self):
        super().__init__("Femorotibial Rotation")

    def _measure(self, point_dict, radius_dict):
        normal = vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])

        medial_condyle, medial_radius = sphere_landmark(point_dict, radius_dict, "medial femur condyle")
        lateral_condyle, lateral_radius = sphere_landmark(point_dict, radius_dict, "lateral femur condyle")
        self.spheres = {
            "medial femur condyle": (medial_condyle, medial_radius),
            "lateral femur condyle": (lateral_condyle, lateral_radius),
        }
        self.axes = {
            "Femoral condylar axis": (medial_condyle, lateral_condyle),
            "Proximal tibial axis": (point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"]),
        }
        vector_distal_femur = vector_with_two_points(medial_condyle, lateral_condyle)
        vector_prox_tibia = vector_with_two_points(point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"])
        
        vector_distal_femur_proj = project_vector_to_plane_from_normal(normal, vector_distal_femur)
        vector_prox_tibia_proj = project_vector_to_plane_from_normal(normal, vector_prox_tibia)
        
        a = angle(vector_distal_femur_proj, vector_prox_tibia_proj)*180/math.pi
        
        t = np.cross(vector_distal_femur_proj, vector_prox_tibia_proj)
        q = np.dot(t,normal)
        
        if self.side.lower() == "right":
            return a, "Innenrotation" if q > 0 else "Aussenrotation"
        elif self.side.lower() == "left":
            return a, "Aussenrotation" if q > 0 else "Innenrotation"
        else:
            raise ValueError(f"Unknown side {self.side}")

class VarusValgusFemurMeasurement(BaseMeasurement):
    SPHERE_LABELS = ["medial femur condyle", "lateral femur condyle"]
    AXIS_LABELS = ["Femoral shaft axis", "Femoral condylar axis"]

    def __init__(self):
        super().__init__("aLDFA (Varus/Valgus Femur)")

    def _measure(self, point_dict, radius_dict):
        medial_condyle, medial_radius = sphere_landmark(point_dict, radius_dict, "medial femur condyle")
        lateral_condyle, lateral_radius = sphere_landmark(point_dict, radius_dict, "lateral femur condyle")
        self.spheres = {
            "medial femur condyle": (medial_condyle, medial_radius),
            "lateral femur condyle": (lateral_condyle, lateral_radius),
        }
        self.axes = {
            "Femoral shaft axis": (point_dict["Femur midpoint 50%"], point_dict["Femur midpoint 30%"]),
            "Femoral condylar axis": (medial_condyle, lateral_condyle),
        }

        # aLDFA is traditionally read on an AP (anteroposterior) radiograph, which is why this
        # used to project both axes onto a coronal plane assumed from a fixed anteroposterior
        # direction in the scanner's frame. That assumption doesn't hold for every scan - a
        # limb tilted well out of that assumed plane produces a materially distorted angle. Since
        # this is a full 3D CT-derived measurement rather than a 2D radiograph, the raw 3D angle
        # between the two axes is used directly instead - it needs no assumption about how the
        # limb was positioned in the scanner.
        # The shaft vector points proximally (50% -> 30%): verified against a deliberately
        # constructed severe-varus test case (moving the proximal landmark medially). Flipping it
        # gives the supplementary angle (180 - angle), which read Valgus for that known-varus
        # case - the earlier "point distally" convention was calibrated against the now-removed
        # plane projection and doesn't carry over to the raw 3D angle.
        vector_axis_femur_vv = vector_with_two_points(point_dict["Femur midpoint 50%"], point_dict["Femur midpoint 30%"])
        vector_dist_femur_vv = vector_with_two_points(medial_condyle, lateral_condyle)

        # aLDFA: the femoral (anatomical) axis stands at 90 deg on the condylar axis. Varus
        # increases this lateral angle above 90 deg, valgus decreases it below 90 deg - this is a
        # magnitude comparison, not a handedness/chirality question, so it needs no side-dependent
        # sign logic (unlike the rotation measurements elsewhere in this module).
        aldfa = angle(vector_axis_femur_vv, vector_dist_femur_vv)*180/math.pi
        is_varus = aldfa > 90
        return aldfa, "aLDFA - Varus" if is_varus else "aLDFA - Valgus"

class AntetorsionMeasurement(BaseMeasurement):
    SPHERE_LABELS = ["femur head", "medial femur condyle", "lateral femur condyle"]
    AXIS_LABELS = ["Femoral condylar axis", "Femoral neck axis", "Femoral shaft axis"]
    LANDMARK_ORDER = ["femur head", "femur neck", "Femur midpoint 30%", "Femur midpoint 50%",
                      "medial femur condyle", "lateral femur condyle"]

    def __init__(self):
        super().__init__("Antetorsion")

    def _measure(self, point_dict, radius_dict):
        center, head_radius = sphere_landmark(point_dict, radius_dict, "femur head")
        medial_condyle, medial_radius = sphere_landmark(point_dict, radius_dict, "medial femur condyle")
        lateral_condyle, lateral_radius = sphere_landmark(point_dict, radius_dict, "lateral femur condyle")
        self.spheres = {
            "femur head": (center, head_radius),
            "medial femur condyle": (medial_condyle, medial_radius),
            "lateral femur condyle": (lateral_condyle, lateral_radius),
        }
        # Femoral (anatomical) shaft axis through the 30 % / 50 % midpoints - the reference axis of
        # this measurement - drawn over the whole femur: from the level of the femoral head to the
        # level of the condyles (both projected onto the axis line).
        shaft_a = np.asarray(point_dict["Femur midpoint 30%"], dtype=float)
        shaft_b = np.asarray(point_dict["Femur midpoint 50%"], dtype=float)
        shaft_dir = shaft_b - shaft_a
        shaft_dir = shaft_dir / np.linalg.norm(shaft_dir)
        condyle_mid = 0.5 * (np.asarray(medial_condyle, dtype=float) + np.asarray(lateral_condyle, dtype=float))
        shaft_start = shaft_a + np.dot(np.asarray(center, dtype=float) - shaft_a, shaft_dir) * shaft_dir
        shaft_end = shaft_a + np.dot(condyle_mid - shaft_a, shaft_dir) * shaft_dir
        self.axes = {
            "Femoral condylar axis": (medial_condyle, lateral_condyle),
            "Femoral neck axis": (center, point_dict["femur neck"]),
            "Femoral shaft axis": (shaft_start, shaft_end),
        }
        normal = vector_with_two_points(point_dict["Femur midpoint 50%"], point_dict["Femur midpoint 30%"])
        vector_distal_femur = vector_with_two_points(medial_condyle, lateral_condyle)
        vector_femur_neck = vector_with_two_points(center, point_dict["femur neck"])
        
        vector_distal_femur_proj = project_vector_to_plane_from_normal(normal, vector_distal_femur)
        vector_femur_neck_proj = project_vector_to_plane_from_normal(normal, vector_femur_neck)
        
        a = angle(vector_femur_neck_proj, vector_distal_femur_proj)*180/math.pi
        
        t = np.cross(vector_distal_femur_proj, vector_femur_neck_proj)
        q = np.dot(t,normal)
        
        if self.side.lower() == "right":
            return a, "no Antetorsion" if q > 0 else "Antetorsion"
        elif self.side.lower() == "left":
            return a, "Antetorsion" if q > 0 else "no Antetorsion"
        else:
            raise ValueError(f"Unknown side {self.side}")
        

class TibialMetatarsalAngleMeasurement(BaseMeasurement):
    '''
    Angle between the tibial condylar axis (proximal reference) and the talus-metatarsal axis
    (distal reference), both projected onto the plane perpendicular to the tibial long axis. In
    a normal limb the foot sits roughly in that same plane as the condylar axis, so this isolates
    rotational deviation of the foot relative to the knee - robust to how much the foot happens
    to be flexed/extended, since flexion/extension mostly changes the metatarsal vector's
    component along the tibial axis, which the projection removes. Same pattern as Tibia
    Torsion/Tibiotalar Rotation, which are also projected onto this same tibial-axis plane.
    '''
    AXIS_LABELS = ["Tibial long axis", "Tibial condylar axis", "Talus-metatarsal axis"]

    def __init__(self):
        super().__init__("Tibial Metatarsal Angle")

    def _measure(self, point_dict, radius_dict):
        normal = vector_with_two_points(point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"])
        vector_proximal = vector_with_two_points(point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"])
        vector_distal = vector_with_two_points(point_dict["Talus centre"], point_dict["Metatarsal centre"])
        self.axes = {
            "Tibial long axis": (point_dict["distal tibia midpoint"], point_dict["proximal tibia midpoint"]),
            "Tibial condylar axis": (point_dict["condylus medialis tibiae"], point_dict["condylus lateralis tibiae"]),
            "Talus-metatarsal axis": (point_dict["Talus centre"], point_dict["Metatarsal centre"]),
        }

        vector_proximal_proj = project_vector_to_plane_from_normal(normal, vector_proximal)
        vector_distal_proj = project_vector_to_plane_from_normal(normal, vector_distal)
        tma = angle(vector_proximal_proj, vector_distal_proj) * 180 / math.pi
        # Below 90 deg: the metatarsal axis has rotated toward the condylar axis's own lateral
        # direction (outward). Above 90 deg: rotated the other way (inward). Verified via
        # controlled rotation-based synthetic geometry - needs no side-dependent sign logic,
        # since "medial -> lateral" is already a correctly side-oriented vector by construction,
        # same reasoning as the aLDFA magnitude-only comparison.
        is_outward = tma < 90
        deviation = abs(tma - 90)
        return deviation, "Outward rotation" if is_outward else "Inward rotation"


class ExampleMeasurement(BaseMeasurement):
    '''
    This class SHOULD NOT BE USED, it is just provided as a example that shows how
    child classes should be implemented.
    '''
    def __init__(self, display_name):
        super().__init__(display_name)
    
    def _measure(self, point_dict, radius_dict):
        p = point_dict['landmark name'] # access landmarks
        m = 42 # calculate float measurement
        
        # Potentially return different descriptions for each side
        if self.side.lower() == "right":
            return m, "Some description"
        elif self.side.lower() == "left":
            return m, "Another description"
        else:
            raise ValueError(f"Unknown side {self.side}")

