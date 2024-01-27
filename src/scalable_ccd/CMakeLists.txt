set(SOURCES
  config.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(stq)
add_subdirectory(tight_inclusion)