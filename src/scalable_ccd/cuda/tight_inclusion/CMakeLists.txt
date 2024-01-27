set(SOURCES
    helper.cu
    helper.cuh
    record.hpp
    root_finder.cu
    root_finder.cuh
    type.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################