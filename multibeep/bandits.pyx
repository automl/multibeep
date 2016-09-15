import cython
from cython.operator cimport dereference as deref
from libcpp cimport bool


cimport bandits_cpp
cimport bandits


from typedefs cimport *

cimport arms
from util import posterior_class



cdef class arm_info:

	cdef fill_attributes(self, const bandits_cpp.arm_info[float_t, rand_t] * tmpptr, unsigned int i):
		self.identifier = tmpptr.identifier
		self.is_active = tmpptr.is_active
		self.num_pulls = tmpptr.num_pulls
		self.estimated_mean = tmpptr.estimated_mean
		self.estimated_variance = tmpptr.estimated_variance
		self.real_mean = deref(tmpptr.get_arm_ptr()).real_mean()
		self.real_variance = deref(tmpptr.get_arm_ptr()).real_variance()
		self.p_max = tmpptr.p_max
		self.posterior = posterior_class()
		self.posterior.thisptr = deref(tmpptr).posterior
		self.rewards = (tmpptr.rewards)[:]
		self.name = deref(tmpptr.get_arm_ptr()).get_ident()


cdef class base:
	""" Base class for all bandits.
	
	It contains the functionality common to all subclasses. To access the 
	arm_info objects of the contained arms, use the index operator '[]'
	
	
	"""
	def add_arm(self, arms.base arm):
		""" adds an arm to the bandit
		
		Parameters
		----------
		arm : multibeep.arms.base
			an instantiated arm
		
		Returns
		-------
		unsigned int
			the unique identifier associated with the arm just added
		"""
		return(self.thisptr.get().add_arm(arm.get_arm_ptr()))

	def deactivate_by_index(self, unsigned int index):
		""" deactivates an arm based on its current index
		
		Parameters
		----------
		index : unsigned int
			the index that should be deactivated. Note the indices of
			other arms might change, so don't use this in succession!
			See deactivate_by_identifier for deactivating multiple arms.
		"""
		self.thisptr.get().deactivate_by_index(index)

	def deactivate_by_identifier(self, unsigned int ident):
		""" deactivates an arm based on its unique identifier
		
		Parameters
		----------
		ident : unsigned int
			the identifier that should be deactivated. In contrast to indices,
			the identifiers are constant over the lifetime of a bandit.
		"""
		self.thisptr.get().deactivate_by_identifier(ident)

	def deactivate_by_confidence_gap(self, float_t delta, bool consider_inactive_arms = True):
		""" deactivates arms based on the posteriors by comparing the confidence bounds computed from the posteriors
		
		Parameters
		----------
		delta : double
			determines the size of the used confidence bounds. They are computed as the
			delta/2, and 1-delta/2 quantiles respectively
		consider_inactive_arms : bool
			If True, the bounds of the inactive arms are also considered to find the largest lower bound.
		"""
		self.thisptr.get().deactivate_by_confidence_gap(delta, consider_inactive_arms)

	def deactivate_n_worst(self, unsigned int n):
		""" deactivates arms solely based on their estimated mean
		
		Parameters
		----------
		n : unsigned int
			the number of arms to deactivate
		"""
		self.tihsptr.get().deactivate_n_worst(n)

	def reactivate_by_index(self, unsigned int index):
		self.thisptr.get().reactivate_by_index(index)
	def reactivate_by_identifier(self, unsigned int ident):
		self.thisptr.get().reactivate_by_identifier(ident)
	def pull_by_index(self, unsigned int index):
		"""
		use this function to pull an arm. Note the index of an arm might
		change when an arm is deactivated.
		
		Parameters
		----------
		index : unsigned int
			the index of the arm to pull
		"""
		return(self.thisptr.get().pull_by_index(index))

	def pull_by_identifier(self, unsigned int ident):
		"""
		use this function to pull an arm.
		
		Parameters
		----------
		ident : unsigned int
			the identifier of the arm to pull
		"""
		return(self.thisptr.get().pull_by_identifier(ident))

	def min_pull_arms(self, unsigned int min_num_pulls):
		"""
		ensures that each arm has been at least pulled a given number of times
		
		Parameters
		----------
		min_num_pulls : unsigned int
			minimum number of pull required for every active arm
		"""
		self.thisptr.get().min_pull_arms(min_num_pulls)
	
	def number_of_arms(self):
		"""
		Returns
		-------
		unsigned int
			total number of arms associated with the bandit
		"""
		return(self.thisptr.get().number_of_arms())
	def number_of_active_arms(self):
		return(self.thisptr.get().number_of_active_arms())
	def number_of_pulls(self):
		return(self.thisptr.get().number_of_pulls())
	def number_of_pulled_arms(self):
		return(self.thisptr.get().number_of_pulled_arms())

	def sort_active_arms_by_mean(self):
		"""
		Sorts the remaining active arms by the estimated mean. The values
		are in descending order such that the 'best' arms has index zero.
		"""
		self.thisptr.get().sort_active_arms_by_mean()


	def update_p_max(self, bool consider_inactive=False, float_t delta = 0.01, unsigned int GL_num_points = 64 ):
		"""
		Updates the p_max values for all arms, available in the arm_info objects

		Parameters
		----------
		consider_inactive : bool
			whether or not to consider all arms during the computation
		delta : float
			controlls the confidence interval. See multibeep.util.posterior.base.support
			for more detail
		GL_num_points : unsigned int
			number of point used during the Gauss-Legendre integration.
		"""
		self.thisptr.get().update_p_max(consider_inactive, delta, GL_num_points)

	def __getitem__( self, int index):
		ai = arm_info()
		ai.fill_attributes(&(deref(self.thisptr)[index]), index)
		return(ai)

cdef class empirical(base):
	"""
	This bandit automatically provides a gaussian posterior based on the returned rewards.
	It does not provide a predictive posterior at the moment.
	"""
	def __init__(self):
		self.tmpptr = new bandits_cpp.empirical[float_t, rand_t]()
		self.thisptr = shared_ptr[bandits_cpp.base[float_t, rand_t] ] (self.tmpptr)
		self.tmpptr = NULL


cdef class posterior(base):
	"""
	This bandit requires every arm added to provide a posterior!
	"""
	def __init__(self):
		self.tmpptr = new bandits_cpp.posterior[float_t, rand_t]()
		self.thisptr = shared_ptr[bandits_cpp.base[float_t, rand_t] ] (self.tmpptr)
		self.tmpptr = NULL

cdef class last_n_pulls(base):
	"""
	Similar to the empirical_bandit, but only the last few rults are used to compute
	the empirical posterior
	
	Parameters
	----------
	n : unsigned int
		number of previous rewards used to compute the posterior
	
	"""
	def __init__(self, unsigned int n = 1):
		self.tmpptr = new bandits_cpp.last_n_pulls[float_t, rand_t](n)
		self.thisptr = shared_ptr[bandits_cpp.base[float_t, rand_t] ] (self.tmpptr)
		self.tmpptr = NULL
