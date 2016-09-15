import cython
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr


cimport bandits_cpp


from typedefs cimport *

from util cimport posterior_class


cdef class arm_info:
	cdef public string name

	cdef public unsigned int index
	cdef public unsigned int identifier
	cdef public bool is_active
	cdef public long double num_pulls
	cdef public float_t estimated_mean
	cdef public float_t estimated_variance

	cdef public float_t real_mean
	cdef public float_t real_variance

	cdef public vector[float_t] rewards

	cdef public float_t p_max

	cdef public posterior_class posterior

	cdef fill_attributes(self, const bandits_cpp.arm_info[float_t, rand_t] * tmpptr, unsigned int i)



cdef class base:
	cdef shared_ptr[bandits_cpp.base[float_t, rand_t] ] thisptr
	# temporary pointer, as a workaround for struggeling to write __init__ with a shared_ptr
	cdef  bandits_cpp.base[float_t, rand_t] * tmpptr


cdef class empirical_bandit(base):
	pass

cdef class posterior_bandit(base):
	pass

cdef class last_n_pulls_bandit(base):
	pass
