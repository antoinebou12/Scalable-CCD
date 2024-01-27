function(create_app APP_NAME APP_MAIN_FILE)
    # executable
    add_executable(${APP_NAME}
        ${APP_MAIN_FILE}
        # Utils
        ground_truth.cpp
        ground_truth.hpp
        io.cpp
        io.hpp
    )

    set_target_properties(${APP_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${APP_NAME} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

    # Link libraries
    target_link_libraries(${APP_NAME} PRIVATE scalable_ccd::scalable_ccd)

    # libigl
    include(libigl)
    target_link_libraries(${APP_NAME} PRIVATE igl::core)

    # Logger
    include(spdlog)
    target_link_libraries(${APP_NAME} PRIVATE spdlog::spdlog)

    # JSON
    include(json)
    target_link_libraries(${APP_NAME} PRIVATE nlohmann_json::nlohmann_json)

    # Logger
    include(cli11)
    target_link_libraries(${APP_NAME} PRIVATE CLI11::CLI11)

    if(APPLE)
      # We need to add the path to the driver (libcuda.dylib) as an rpath,
      # so that the static cuda runtime can find it at runtime.
      set_property(TARGET ${APP_NAME}
                   PROPERTY
                   BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()

    target_compile_options(${APP_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      --generate-line-info
      --use_fast_math
      --relocatable-device-code=true
      --ptxas-options=-v

      # --maxrregcount=7
      -fPIC # https://stackoverflow.com/questions/5311515/gcc-fpic-option
    >)
endfunction()

create_app("stq_cpu_bin" "${CMAKE_CURRENT_SOURCE_DIR}/main_stq.cpp")
if (SCALABLE_CCD_WITH_CUDA)
  create_app("stq_cuda_bin" "${CMAKE_CURRENT_SOURCE_DIR}/main_stq.cu")
  create_app("tight_inclusion_cuda_bin" "${CMAKE_CURRENT_SOURCE_DIR}/main_tight_inclusion.cu")
endif()
