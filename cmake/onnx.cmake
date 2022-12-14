function(configure_onnx)
    message(STATUS "Configuring onnx...")
    set(ONNX_BUILD_MAIN_LIB ON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/onnx)
    target_compile_definitions(onnx_proto PRIVATE ONNX_BUILD_MAIN_LIB)
endfunction()
