from libcpp.memory cimport shared_ptr

from typedefs cimport *
cimport arms_cpp
from util cimport rng_class



cdef class base:
	cdef shared_ptr[ arms_cpp.base[float_t, rand_t]] thisptr
	# hack to make shared pointers work: instantiate a temporary pointer first
	cdef arms_cpp.base[float_t, rand_t] * tmpptr

	cdef shared_ptr[arms_cpp.base[float_t, rand_t] ] get_arm_ptr (self)



cdef class bernoulli(base):
	pass

cdef class exponential(base):
	pass

cdef class normal(base):
	pass

cdef class data(base):
	pass

cdef class python(base):
	pass
