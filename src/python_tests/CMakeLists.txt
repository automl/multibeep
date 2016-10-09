file(GLOB TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *_test.py)

#colon separated list for tests to skip
set (SKIP_TESTS "")

# Build all the unit tests
foreach(TEST_SOURCE ${TESTS})
	# construct target name from file name
	string(REPLACE ".py" "" TMP "${TEST_SOURCE}")

	# only add tests that should not be skipped
	list(FIND SKIP_TESTS "${TEST_SOURCE}" TO_BE_SKIPPED)
	if (TO_BE_SKIPPED EQUAL -1)
		add_test(${TMP} ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/python_tests/${TEST_SOURCE}
				WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
	else()
		message("Skipping ${TEST_SOURCE}")
	endif()
endforeach()
