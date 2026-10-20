# DoggyBone3D - canine osteotomy planning and patient-specific cutting guides for 3D Slicer
# Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany
# Based on BoneAngleMeter, Copyright (c) 2022 Juliette Burg Personnaz, Andreas Voigt
# (CTK, LMU Munich) - https://github.com/jburgp/BoneAngleMeter_public
# Licensed under the MIT License, see LICENSE in the repository root.

from DoggyboneLib.landmark_logic import SimpleLandmark, SphereLandmark, CentroidLandmark

# first guess of the bone axis for the diaphysis centres (proximal names, distal names)
TIBIA_AXIS = (("intercondylar eminence",), ("Talus centre",))
FEMUR_AXIS = (("femur head",), ("medial femur condyle", "lateral femur condyle"))

LANDMARKS = [
    CentroidLandmark('distal tibia midpoint',
                     "Choose the most distal midpoint of the diaphysis in a circular tibia transversal plane. "
                     "A click at that height is enough: the point is set to the area centroid of the cross-section.",
                     r"Resources/descriptions/disttibiamiddpoint.png", axis_from=TIBIA_AXIS),
    CentroidLandmark('proximal tibia midpoint',
                     "Choose the point in the middle of the diaphysis at the height of the foramen nutricum. "
                     "A click at that height is enough: the point is set to the area centroid of the cross-section.",
                     r"Resources/descriptions/proxtibiamidpoint.png", axis_from=TIBIA_AXIS),
    SimpleLandmark('lateral cochlea',
                   "Choose the most cranial point on the cochlea tibiae lateralis.",
                   r"Resources/descriptions/lateralcochlea.png"),
    SimpleLandmark('medial cochlea',
                   "Choose the most cranial point on the cochlea tibiae medialis.",
                   r"Resources/descriptions/medialcochlea.png"),
    SimpleLandmark('condylus medialis tibiae',
                   "Choose the most caudal point on the convex condyle surface.",
                   r"Resources/descriptions/condylusmedtibiae.png"),
    SimpleLandmark('condylus lateralis tibiae',
                   "Choose the most caudal point on the convex condyle surface.",
                   r"Resources/descriptions/condyluslateralistibiae.png"),
    SimpleLandmark('lateral condyle articulation point tibia',
                   "Choose the lowest midpoint of the condylus tibialis lateralis articulation groove.",
                   r"Resources/descriptions/latcondylearticulationpoint.png"),
    SimpleLandmark('medial condyle articulation point tibia',
                   "Choose the lowest midpoint of the condylus tibialis medialis articulation groove.",
                   r"Resources/descriptions/medialcondylearticulationpoint.png"),
    SimpleLandmark('medial cochlea articulation point tibia',
                   "Choose the lowest midpoint of the cochlea tibialis medialis articulation groove.",
                   r"Resources/descriptions/medcochleaarticulationpoint.png"),
    SimpleLandmark('lateral cochlea articulation point tibia',
                   "Choose the lowest midpoint of the cochlea tibialis lateralis articulation groove.",
                   r"Resources/descriptions/lateralcochleaarticulationpoint.png"),
    SimpleLandmark('medial talus',
                   "Choose the most dorsal point on the trochlea tali.",
                   r"Resources/descriptions/medialtalus.png"),
    SimpleLandmark('lateral talus',
                   "Choose the most dorsal point on the trochlea tali.",
                   r"Resources/descriptions/lateraltalus.png"),
    SimpleLandmark('intercondylar eminence',
                   "Choose the centre between the medial and lateral intercondylar tubercles (proximal point of the tibial mechanical / functional axis).",
                   r""),
    SimpleLandmark('cranial medial tibial plateau',
                   "Choose the most cranial point of the articular surface of the medial tibial condyle (cranial end of the plateau line).",
                   r""),
    SimpleLandmark('caudal medial tibial plateau',
                   "Choose the most caudal point of the articular surface of the medial tibial condyle (caudal end of the plateau line).",
                   r""),
    SimpleLandmark('Talus centre',
                   "Choose the center of the talus.",
                   r"Resources/descriptions/taluscentre.png"),
    SimpleLandmark('Metatarsal centre',
                   "Choose the center of the metatarsal bone, at the level used for the distal reference point of the tibio-metatarsal rotation.",
                   r"Resources/descriptions/metatarsalcentre.png"),
    SphereLandmark('medial femur condyle',
                   "Place the center inside the medial condyle, then adjust the sphere's radius until it best matches the condyle surface.",
                   r"Resources/descriptions/medfemurcondyle.png"),
    SphereLandmark('lateral femur condyle',
                   "Place the center inside the lateral condyle, then adjust the sphere's radius until it best matches the condyle surface.",
                   r"Resources/descriptions/lateralfemurcondyle.png"),
    CentroidLandmark('Femur midpoint 33%',
                     "Choose the point at 33 % of the femoral length (one third), measured from the proximal end, in the "
                     "middle of the diaphysis. 'Place at 33 %' sets it from the femur head and condyle spheres; a click "
                     "at any height is centred in the cross-section.",
                     r"Resources/descriptions/proxfemurmidpoint.png", axis_from=FEMUR_AXIS, fraction=0.33),
    CentroidLandmark('Femur midpoint 50%',
                     "Choose the point at 50 % of the femoral length (mid-femur), in the middle of the diaphysis. "
                     "'Place at 50 %' sets it from the femur head and condyle spheres; a click at any height is "
                     "centred in the cross-section.",
                     r"Resources/descriptions/distfemurmidpoint.png", axis_from=FEMUR_AXIS, fraction=0.50),
    SimpleLandmark('femur neck', 
                   "Choose the center of the proximal femoral metaphysis on the height of the highest elevation of the lesser trochanter.", 
                   r"Resources/descriptions/femurneck.png"),
    SphereLandmark('femur head',
                   "Place the center inside the femoral head, then adjust the sphere's radius until it best matches the articular surface.",
                   r"Resources/descriptions/point on femur head.png")
]

# Earlier names of landmarks -> current name. Saved scenes and exported landmark files that still use
# an old name are recognised (see DoggyboneWidget._relink_existing_landmarks and the CSV import).
LEGACY_LANDMARK_NAMES = {"Femur midpoint 30%": "Femur midpoint 33%"}


def current_landmark_name(name):
    return LEGACY_LANDMARK_NAMES.get(name, name)

