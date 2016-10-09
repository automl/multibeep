cmake_minimum_required(VERSION 3.3)
project(multibeep)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")


IF(NOT CMAKE_BUILD_TYPE)
 SET(CMAKE_BUILD_TYPE "DEBUG")
 #SET(CMAKE_BUILD_TYPE "RELEASE")
 #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
 #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()


find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Doxygen)
find_package(PythonInterp)


enable_testing()

include_directories ("${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/lib" "${Boost_INCLUDE_DIR}")




# the documentation
if(DOXYGEN_FOUND)
	configure_file(${PROJECT_SOURCE_DIR}/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)


add_library(gauss_legendre ${PROJECT_SOURCE_DIR}/lib/gauss_legendre/gauss_legendre.c)
link_libraries(gauss_legendre ${Boost_LIBRARIES}) # Deprecated but so convenient!

add_subdirectory("src/test")


if(PYTHONINTERP_FOUND)
	add_custom_target(python_bindings ALL
			${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/setup.py build_ext --force --inplace
			WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	add_subdirectory("src/python_tests")
else()
	message("No Python interpreter found!")
endif(PYTHONINTERP_FOUND)






