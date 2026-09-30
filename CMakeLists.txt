cmake_minimum_required(VERSION 3.13.4)

project(BoneAngleMeter)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/BoneAngleMeter")
set(EXTENSION_CATEGORY "Orthopedics")
set(EXTENSION_CONTRIBUTORS "Juliette Burg Personnaz, Andreas Voigt (CTK, LMU Munich), Matthias Frank (Kleintierklinik Frank & Partner, Freiburg)")
set(EXTENSION_DESCRIPTION "Canine hind-limb angle measurement (BoneAngleMeter) and corrective osteotomy planning with patient-specific cutting guides (Doggybone 3D). For research use only.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/BoneAngleMeter.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/BoneAngleMeter/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BoneAngleMeterModule)
add_subdirectory(Doggybone)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
