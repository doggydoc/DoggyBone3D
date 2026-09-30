#-----------------------------------------------------------------------------
set(MODULE_NAME BoneAngleMeterModule)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  Resources/helpers.py
  Resources/landmark_logic.py
  Resources/landmarks.py
  Resources/measurement_logic.py
  Resources/osteotomy_logic.py
  Resources/measurements.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/descriptions/condyluslateralistibiae.png
  Resources/descriptions/condylusmedtibiae.png
  Resources/descriptions/distfemurmidpoint.png
  Resources/descriptions/disttibiamiddpoint.png
  Resources/descriptions/femurneck.png
  Resources/descriptions/latcondylearticulationpoint.png
  Resources/descriptions/lateralcochlea.png
  Resources/descriptions/lateralcochleaarticulationpoint.png
  Resources/descriptions/lateralfemurcondyle.png
  Resources/descriptions/lateraltalus.png
  Resources/descriptions/medcochleaarticulationpoint.png
  Resources/descriptions/medfemurcondyle.png
  Resources/descriptions/medialcochlea.png
  Resources/descriptions/medialcondylearticulationpoint.png
  Resources/descriptions/medialtalus.png
  Resources/descriptions/taluscentre.png
  Resources/descriptions/metatarsalcentre.png
  Resources/descriptions/proxfemurmidpoint.png
  "Resources/descriptions/point on femur head.png"
  Resources/descriptions/proxtibiamidpoint.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
