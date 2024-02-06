set(SOURCES
    ccd_buffer.cuh
    ccd_config.cuh
    ccd_data.cuh
    interval.cuh
    narrow_phase.cu
    narrow_phase.cuh
    root_finder.cu
    root_finder.cuh
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

# ==============================================================================
# Subfolders
# ==============================================================================