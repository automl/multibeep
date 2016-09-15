import cython

from libcpp cimport bool
#from libcpp.pair cimport pair
#from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr


# TODO: check for const methods in the c++ code and add the keyword here!
#       also, check for exceptions


from typedefs cimport *


cimport util_cpp



cdef extern from "multibeep/arm/arm.hpp" namespace "multibeep::arms":
	cdef cppclass base[num_t, rng_t]:
		num_t pull()
		num_t real_mean()
		num_t real_variance()
		string get_ident()
		bool provides_posterior()
		shared_ptr[util_cpp.base[num_t, rng_t] ] posterior()
		void deactivate()


cdef extern from "multibeep/arm/bernoulli.hpp" namespace "multibeep::arms":
	cdef cppclass bernoulli_arm[num_t, rng_t] (base[num_t,rng_t]):
		bernoulli_arm(num_t, shared_ptr[rng_t])

cdef extern from "multibeep/arm/exponential.hpp" namespace "multibeep::arms":
	cdef cppclass exponential_arm[num_t,rng_t] (base[num_t, rng_t]):
		exponential_arm(num_t, shared_ptr[rng_t])

cdef extern from "multibeep/arm/normal.hpp" namespace "multibeep::arms":
	cdef cppclass normal_arm[num_t, rng_t] (base[num_t, rng_t]):
		normal_arm(num_t, num_t, shared_ptr[rng_t])


cdef extern from "multibeep/arm/data.hpp" namespace "multibeep::arms":
	cdef cppclass data_arm_bootstrap[num_t, rng_t](base[ num_t, rng_t]):
		data_arm_bootstrap (num_t *, unsigned int, string, shared_ptr[rng_t])
	
	cdef cppclass data_arm_sequential[num_t, rng_t](base[num_t, rng_t]):
		data_arm_sequential(num_t *, unsigned int, string, shared_ptr[rng_t])


ctypedef float_t (*python_pull)(void*)
ctypedef void (*python_deactivate)(void*)
ctypedef shared_ptr[util_cpp.base[float_t, rand_t] ] (*python_posterior)(void*)

cdef extern from "multibeep/arm/python_arm.hpp" namespace "multibeep::arms":
	cdef cppclass python_arm[num_t, rng_t] (base[num_t, rng_t]):
		python_arm(string, void*, python_pull, python_pull, python_pull, python_posterior, python_deactivate)
