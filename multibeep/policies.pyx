import cython
from libcpp.memory cimport shared_ptr


cimport policies_cpp

from typedefs cimport *

cimport policies
cimport bandits
from util cimport rng_class


cdef class base:
	def __dealloc__(self):
		del self.thisptr
	def select_next_arm(self):
		"""
		policy suggests the next arm to pull
		
		Returns
		-------
		unsigned int
			the current *index* of the arm to pull
		"""
		return(self.thisptr.select_next_arm())

	def play_n_rounds(self, cython.uint n):
		"""
		automatically pull multiple times.
		
		If not specified otherwise, the number of rounds to play equals
		the number of pulls.
		
		Parameters
		----------
		n : unsigned int
			number of round to be played
		
		"""
		self.thisptr.play_n_rounds(n)
	

cdef class random(base):
	""" the random policy just picks an arm uniformly at random among all active arms
	
	Parameters
	----------
	b : multibeep.bandits.bandit
		the bandit to be played
	rng: multibeep.util.rng_class
		a valid random number generator
	"""
	def __init__ (self, bandits.base b, rng_class rng):
		self.thisptr = new policies_cpp.random[float_t, rand_t] (b.thisptr, rng.thisptr)

cdef class UCB_p(base):
	""" UCB_p picks the arm with the largest Upper Confidence Bound $\mu_i + p \cdot \sigma_i$
	
	Parameters
	----------
	b : multibeep.bandits.bandit
		the bandit to be played
	rng : multibeep.util.rng_class
		a valid random number generator
	p : double
		prefactor to the standard deviation in the equation above. Controlls exploration vs. exploitation.
	"""
	def __init__ (self, bandits.base b, rng_class rng, float_t p):
		self.thisptr = new policies_cpp.UCB_p[float_t, rand_t] (b.thisptr, rng.thisptr, p)

cdef class prob_match(base):
	""" Probability Match pulls arms randomly where the probability is proportional to having the highest mean.
	
	Parameters
	----------
	b : multibeep.bandits.bandit
		the bandit to be played
	rng: multibeep.util.rng_class
		a valid random number generator
	"""
	def __init__ (self, bandits.base b, rng_class rng):
		self.thisptr = new policies_cpp.prob_match[float_t, rand_t] (b.thisptr, rng.thisptr)

cdef class successive_halving(base):
	""" pulls 
	
	Parameters
	----------
	b : multibeep.bandits.bandit
		the bandit to be played
	min_num_pulls : unsigned int
		number of pulls for every arm in the first round
	frac_arms : double
		after each round, only frac_arms * b.number_of_active_arms() remain active, all others are deactivated.
	factor_pulls : double
		every round the number of pull is factor_pull times the value of the previous round.
		Default is 0, which means that 1/frac arms is used. With that choice the total number
		of pulls per round is constant.
	"""
	def __init__ (self, bandits.base b, unsigned int min_num_pulls, float_t frac_arms, float_t factor_pulls = 0):
		if factor_pulls <= 0 :
			self.thisptr = new policies_cpp.successive_halving[float_t, rand_t] (b.thisptr, min_num_pulls, frac_arms)
		else:
			self.thisptr = new policies_cpp.successive_halving[float_t, rand_t] (b.thisptr, min_num_pulls, frac_arms, factor_pulls)
