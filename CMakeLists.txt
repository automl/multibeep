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

#add_executable(ut_arm "${PROJECT_SOURCE_DIR}/src/test/unit_test_arm.cpp")
#set_property(TARGET ut_arm PROPERTY CXX_STANDARD 11)
#set_property(TARGET ut_arm PROPERTY CXX_STANDARD_REQUIRED ON)
#add_test (UTArm ut_arm)


#add_executable(ut_bandit "${PROJECT_SOURCE_DIR}/src/test/test_bandit.cpp")
#set_property(TARGET ut_bandit PROPERTY CXX_STANDARD 11)
#set_property(TARGET ut_bandit PROPERTY CXX_STANDARD_REQUIRED ON)
#add_test (UTBandit ut_bandit)


#add_executable(ut_policy "${PROJECT_SOURCE_DIR}/src/test/test_policy.cpp")
#set_property(TARGET ut_policy PROPERTY CXX_STANDARD 11)
#set_property(TARGET ut_policy PROPERTY CXX_STANDARD_REQUIRED ON)
#add_test (UTPolicy ut_policy)
#set_tests_properties (UTPolicy PROPERTIES PASS_REGULAR_EXPRESSION "No errors detected")

