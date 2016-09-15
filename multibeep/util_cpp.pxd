import cython

from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr


# TODO: check for const methods in the c++ code and add the keyword here!
#       also, check for exceptions


from typedefs cimport *


cdef extern from *:
	ctypedef void* false "false"
	ctypedef void* true "true"


cdef extern from "multibeep/util/posteriors.hpp" namespace "multibeep::util::posteriors":
	cdef cppclass base[num_t, rng_t]:
		num_t mean() const
		num_t variance() const
		num_t pdf(num_t) const
		num_t cdf(num_t) const
		num_t quantile (num_t) const
		pair[num_t, num_t] support(num_t) const
		num_t predictive_posterior_sample (rng_t) const
		void add_observation(num_t)
	
	cdef cppclass gaussian_posterior[num_t, rng_t] (base[num_t, rng_t]):
		gaussian_posterior (num_t, num_t) except+
