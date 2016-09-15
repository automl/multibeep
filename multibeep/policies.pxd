import cython
#from cython.operator cimport dereference as deref
#from libcpp cimport bool
#from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr


cimport bandits_cpp
cimport policies_cpp

from typedefs cimport *



cdef class base:
	cdef policies_cpp.base[float_t, rand_t]* thisptr



cdef class random(base):
	pass
cdef class UCB_p(base):
	pass
cdef class prob_match(base):
	pass
cdef class successive_halving(base):
	pass

