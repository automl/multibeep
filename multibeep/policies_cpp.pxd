import cython

from libcpp.memory cimport shared_ptr

# TODO: check for const methods in the c++ code and add the keyword here!
#       also, check for exceptions

from typedefs cimport *
cimport bandits_cpp


####################################################################
# policy section
####################################################################

cdef extern from "multibeep/policy/policy.hpp" namespace "multibeep::policies":
	cdef cppclass base[num_t, rng_t]:
		policy_base (shared_ptr[bandits_cpp.base[num_t, rng_t] ])
		unsigned int select_next_arm()
		void play_n_rounds (unsigned int)

cdef extern from "multibeep/policy/random.hpp" namespace "multibeep::policies":
	cdef cppclass random[num_t, rng_t] (base[num_t, rng_t]):
		random(shared_ptr[bandits_cpp.base[num_t, rng_t] ], shared_ptr[rng_t])

cdef extern from "multibeep/policy/ucb.hpp" namespace "multibeep::policies":
	cdef cppclass UCB_base[num_t, rng_t] (base[num_t, rng_t]):
		UCB_base(shared_ptr[bandits_cpp.base[num_t, rng_t] ], shared_ptr[rng_t])

cdef extern from "multibeep/policy/ucbp.hpp" namespace "multibeep::policies":
	cdef cppclass UCB_p[num_t, rng_t] (UCB_base[num_t, rng_t]):
		UCB_p(shared_ptr[bandits_cpp.base[num_t, rng_t] ], shared_ptr[rng_t], num_t)

cdef extern from "multibeep/policy/prob_match.hpp" namespace "multibeep::policies":
	cdef cppclass prob_match[num_t, rng_t] (base[num_t, rng_t]):
		prob_match(shared_ptr[bandits_cpp.base[num_t, rng_t] ], shared_ptr[rng_t])

cdef extern from "multibeep/policy/successive_halving.hpp" namespace "multibeep::policies":
	cdef cppclass successive_halving[num_t, rng_t] (base[num_t, rng_t]):
		successive_halving(shared_ptr[bandits_cpp.base[num_t, rng_t] ], unsigned int, num_t, num_t)
		successive_halving(shared_ptr[bandits_cpp.base[num_t, rng_t] ], unsigned int, num_t)

