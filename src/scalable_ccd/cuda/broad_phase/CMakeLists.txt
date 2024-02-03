set(SOURCES
    aabb.cu
    aabb.cuh
    broad_phase_multigpu.cu
    broad_phase_multigpu.cuh
    broad_phase.cu
    broad_phase.cuh
    collision.cu
    collision.cuh
    queue.cu
    queue.cuh
    sweep.cu
    sweep.cuh
    util.cu
    util.cuh
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

# ==============================================================================
# Subfolders
# ==============================================================================