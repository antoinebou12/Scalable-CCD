set(SOURCES
  ipc_ccd_strategy.cpp
  ipc_ccd_strategy.hpp
  memory_handler.cuh
  scalar.cu
  scalar.cuh
  types.cuh
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(stq)
add_subdirectory(tight_inclusion)
add_subdirectory(utils)