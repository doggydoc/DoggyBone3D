cmake_minimum_required(VERSION 3.13.4)

project(Doggybone3D)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/doggydoc/Doggybone3D")
set(EXTENSION_CATEGORY "Orthopedics")
set(EXTENSION_CONTRIBUTORS "Juliette Burg Personnaz, Andreas Voigt (CTK, LMU Munich), Matthias Frank (Kleintierklinik Frank & Partner, Freiburg)")
set(EXTENSION_DESCRIPTION "Canine hind-limb angle measurement, corrective osteotomy planning and patient-specific cutting guides. Based on BoneAngleMeter (Burg Personnaz, Voigt). For research use only.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/doggydoc/Doggybone3D/main/Doggybone/Resources/Icons/Doggybone.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Doggybone)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
