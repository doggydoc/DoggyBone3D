#-----------------------------------------------------------------------------
set(MODULE_NAME Doggybone)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  DoggyboneLib/__init__.py
  DoggyboneLib/guide_logic.py
  DoggyboneLib/helpers.py
  DoggyboneLib/landmark_logic.py
  DoggyboneLib/landmarks.py
  DoggyboneLib/measurement_logic.py
  DoggyboneLib/measurements.py
  DoggyboneLib/osteotomy_logic.py
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
  Resources/descriptions/metatarsalcentre.png
  "Resources/descriptions/point on femur head.png"
  Resources/descriptions/proxfemurmidpoint.png
  Resources/descriptions/proxtibiamidpoint.png
  Resources/descriptions/taluscentre.png
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
  add_subdirectory(Testing)
endif()
