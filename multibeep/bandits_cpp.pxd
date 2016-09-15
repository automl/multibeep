import cython

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr


# TODO: check for const methods in the c++ code and add the keyword here!
#       also, check for exceptions


from typedefs cimport *


cimport arms_cpp
cimport util_cpp


####################################################################
# bandit section
####################################################################

cdef extern from "multibeep/bandit/arm_info.hpp" namespace "multibeep::bandits":
	cdef cppclass arm_info[num_t, rng_t]:
		unsigned int    identifier
		bool            is_active
		long double     num_pulls
		# TODO: add reward_stats
		vector[num_t]   rewards
		num_t           p_max
		num_t           p_min
		shared_ptr[util_cpp.base[num_t, rng_t] ] posterior
		num_t estimated_mean
		num_t estimated_variance
		shared_ptr[const arms_cpp.base[num_t, rng_t] ] get_arm_ptr()

cdef extern from "multibeep/bandit/bandit.hpp" namespace "multibeep::bandits":
	cdef cppclass base[num_t, rng_t]:
		base                                ()
		unsigned int add_arm                (shared_ptr[arms_cpp.base])
		void deactivate_by_index            (unsigned int)
		void deactivate_by_identifier       (unsigned int)
		void deactivate_by_confidence_gap   (num_t delta, bool)
		void deactivate_by_pmax_threshold   (num_t delta)
		void deactivate_n_worst             (unsigned int)
		void reactivate_by_index            (unsigned int)
		void reactivate_by_identifier       (unsigned int)
		num_t pull_by_index                 (unsigned int)
		num_t pull_by_identifier            (int)
		void min_pull_arms                  (unsigned int)
		unsigned int number_of_arms         ()
		unsigned int number_of_active_arms  ()
		unsigned int number_of_pulls        ()
		unsigned int number_of_pulled_arms  ()
		const arm_info & operator[]         (unsigned int)
		void update_arm_info                (unsigned int)
		void sort_active_arms_by_mean       ()
		void update_p_max					(bool, num_t, unsigned int)


cdef extern from "multibeep/bandit/empirical_bandits.hpp" namespace "multibeep::bandits":
	cdef cppclass empirical[num_t, rng_t] (base[num_t, rng_t]):
		empirical()

	cdef cppclass last_n_pulls[num_t, rng_t] (base[num_t, rng_t]):
		last_n_pulls( unsigned int)


cdef extern from "multibeep/bandit/posterior_bandit.hpp" namespace "multibeep::bandits":
	cdef cppclass posterior[num_t, rng_t] (base[num_t, rng_t]):
		posterior()
