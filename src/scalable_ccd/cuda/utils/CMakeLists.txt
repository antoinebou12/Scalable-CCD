set(SOURCES
  assert.cuh
  limits.cuh
  record.cuh
  timer.cuh
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################