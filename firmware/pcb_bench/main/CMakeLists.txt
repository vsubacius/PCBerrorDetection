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
# utilities.cmake reads `cmake_dir` to find data_file_embed_asm_aligned.cmake.
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# Must match the asm symbols in main.cpp: _binary_student_c2_espdl_start and
# _binary_boards_bin_start/_end. boards.bin comes from tools/make_boards.py.
set(model_file ${CMAKE_CURRENT_LIST_DIR}/../models/student_c2.espdl)
set(boards_file ${CMAKE_CURRENT_LIST_DIR}/../models/boards.bin)

idf_component_register(
    SRCS "main.cpp" "gate.cpp" "board.cpp" "boards.cpp" "live.cpp" "camera.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp-dl esp_timer esp32-camera esp_driver_usb_serial_jtag
)

target_add_aligned_binary_data(${COMPONENT_LIB} ${model_file} BINARY)
target_add_aligned_binary_data(${COMPONENT_LIB} ${boards_file} BINARY)
