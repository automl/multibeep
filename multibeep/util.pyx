cimport numpy as np
import numpy as np
from typedefs cimport *


cimport util_cpp
cimport util


cdef class rng_class:
	""" a pseudo random number generator
	
	Parameters
	----------
	
	seed : int
		to initialize the PRNG in a specific state
	
	"""
	def __init__(self, cython.int seed):
		self.thisptr = shared_ptr[rand_t] (new rand_t(seed))

	cdef shared_ptr[rand_t] get_shared_ptr(self):
		return(self.thisptr)


cdef class posterior_class:
	""" base class for all posteriors"""
	def __init__(self):
		self.thisptr = shared_ptr[util_cpp.base[float_t, rand_t]] (NULL)
		self.tmpptr = NULL

	cdef shared_ptr[util_cpp.base[float_t, rand_t]] get_shared_ptr(self):
		return(self.thisptr)

	cpdef valid(self):
		""" checks if the posterior is useful
		
		Returns
		-------
		bool
			False, if the posterior is not valid, e.g. when not enough
			pulls have been performed.
		"""
		if (self.thisptr):
			return(True)
		else:
			return(False)

	def mean(self):
		""" posterior's mean
		
		Returns
		-------
		float
			the mean of the posterior or None if the posterior is invalid
		"""
		if self.valid():
			return(self.thisptr.get().mean())
		return(None)

	def variance(self):
		""" variance of the posterior over the mean
		
		Returns
		-------
		float
			the variance of the posterior or None if the posterior is invalid
		"""
		if self.valid():
			return(self.thisptr.get().variance())
		return(None)

	def pdf(self, float_t x):
		""" evaluates the posterior for the mean
		
		Parameters
		----------
		x : float
			where to evaluate the pdf
		
		Returns
		-------
		float
			 the pdf value at x or None if the posterior is invalid
		"""
		if self.valid():
			return(self.thisptr.get().pdf(x))
		return(None)

	def cdf(self, float_t x):
		if self.valid():
			return(self.thisptr.get().cdf(x))
		return(None)

	def quantile(self, float_t x):
		if self.valid():
			return(self.thisptr.get().quantile(x))
		return(None)

	def support(self, float_t delta):
		if self.valid():
			return(self.thisptr.get().support(delta))
		return(None,None)


cdef class gaussian_posterior(posterior_class):
	def __init__ (self, float_t mean, float_t variance):
		super().__init__()

		try:
			self.tmpptr = new util_cpp.gaussian_posterior[float_t, rand_t](mean, variance)
			self.thisptr = shared_ptr[util_cpp.base[float_t, rand_t]] (self.tmpptr)
			self.tmpptr = NULL
		except:
			self.thisptr = shared_ptr[util_cpp.base[float_t, rand_t]] (NULL)
			self.tmpptr = NULL
			
