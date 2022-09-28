find_package(FlatBuffers REQUIRED)

function(add_fbs_header name)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${name}.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.fbs
    COMMAND flatc ${CMAKE_CURRENT_SOURCE_DIR}/${name}.fbs -cpp -o ${CMAKE_CURRENT_BINARY_DIR}/${name}.h
    )
endfunction(add_fbs_header name)

function(compile_flatbuffers_schema_to_cpp SRC_FBS)
  get_filename_component(SRC_FBS_DIR ${SRC_FBS} PATH)
  string(REGEX REPLACE "\\.fbs$" "_generated.h" GEN_HEADER ${SRC_FBS})
  add_custom_command(
    OUTPUT ${GEN_HEADER}
    COMMAND "${FLATBUFFERS_FLATC_EXECUTABLE}" -cpp -o "${SRC_FBS_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FBS}"
    DEPENDS flatc)
endfunction()

compile_flatbuffers_schema_to_cpp(gnt)