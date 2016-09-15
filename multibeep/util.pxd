import cython
from libcpp.memory cimport shared_ptr


import numpy as np
cimport numpy as np


from typedefs cimport *
cimport util_cpp

cdef class rng_class:
	cdef shared_ptr[rand_t] thisptr
	cdef shared_ptr[rand_t] get_shared_ptr(self)

cdef class posterior_class:
	cdef shared_ptr[util_cpp.base[float_t, rand_t]] thisptr
	cdef util_cpp.base[float_t, rand_t]* tmpptr
	cpdef valid(self)
	cdef shared_ptr[util_cpp.base[float_t, rand_t]] get_shared_ptr(self)


cdef class gaussian_posterior(posterior_class):
	pass
