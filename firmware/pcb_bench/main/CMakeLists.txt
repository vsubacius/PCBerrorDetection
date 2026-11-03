# Embed the quantized student in .rodata. The ESP-DL component directory has to
# be looked up because its name depends on whether esp-dl was pulled in by the
# component manager (espressif__esp-dl) or vendored locally (esp-dl).
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
else()
    message(FATAL_ERROR "esp-dl component not found; run 'idf.py reconfigure'")
endif()
include(${espdl_dir}/fbs_loader/cmake/utilities.cmake)

# Must match the asm symbol in main.cpp: _binary_student_c1_espdl_start
set(embed_files ${CMAKE_CURRENT_LIST_DIR}/../models/student_c1.espdl)

idf_component_register(
    SRCS "main.cpp" "gate.cpp" "board.cpp"
    INCLUDE_DIRS "."
)

target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)
