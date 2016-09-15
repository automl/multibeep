from cython.operator cimport dereference as deref

import numpy as np
cimport numpy as np

cimport util_cpp
cimport arms_cpp

cimport arms


from typedefs cimport *
from util cimport rng_class, posterior_class


cdef class base:
	def pull(self):
		""" pulls the arm 
		
		Returns
		-------
		float
			recieved reward
		
		"""
		return(self.thisptr.get().pull())

	def real_mean(self):
		""" the mean of the underlying distribution
		
		Returns
		-------
		float
			mean of the underlying distribution, NaN if N/A
		"""
		return(self.thisptr.get().real_mean())

	def real_variance(self):
		""" the variance of the underlying distribution
		
		Returns
		-------
		float
			variance of the underlying distribution, NaN if N/A
		"""
		return(self.thisptr.get().real_variance())
	
	def get_ident(self):
		""" access to the possibly non-unique name of an arm
		
		Returns
		-------
		string
			name of the arm
		"""
		return(self.thisptr.get().get_ident())
		
	def provides_posterior(self):
		""" to query if an arm provides a posterior
		
		Returns
		-------
		bool
			True if the arm has a posterior, False if not
		"""
		return(self.thisptr.get().provides_posterior())

	def posterior(self):
		""" access to the arm's posterior
		
		Returns
		-------
		posterior_class
			a (valid) posterior
		"""
		p = posterior_class()
		p.thisptr = deref(self.thisptr.get()).posterior()
		return(p)
		
	cdef shared_ptr[arms_cpp.base[float_t, rand_t] ] get_arm_ptr (self):
		return(self.thisptr)



cdef class bernoulli(base):
	""" The classic Bernoulli arm.
	
	Provides the 'default' posterior using a Beta prior.

	Parameters
	----------
	p :	float
		the p parameter of the Bernoulli distribution
	rng : multibeep.util.rng_class
		a random number generator object
	"""
	def __init__(self, float_t p, rng_class rng):
		self.tmpptr = new arms_cpp.bernoulli_arm[float_t, rand_t] (p, rng.get_shared_ptr())
		self.thisptr = shared_ptr[ arms_cpp.base[float_t, rand_t] ] (self.tmpptr)
		self.tmpptr = NULL

cdef class exponential(base):
	""" An arm with the exponential reward distribution.
	
	Provides a posterior using an inverse-Gamma prior.

	Parameters
	----------
	l :	float
		the lambda parameter of the exponential distribution
	rng : multibeep.util.rng_class
		a random number generator object
	"""
	def __init__(self, float_t l, rng_class rng):
		self.tmpptr = new arms_cpp.exponential_arm[float_t, rand_t] (l, rng.get_shared_ptr())
		self.thisptr = shared_ptr[ arms_cpp.base[float_t, rand_t] ] (self.tmpptr)
		self.tmpptr = NULL

cdef class normal(base):
	""" An arm with a normal reward distribution
	
	Provides a Bayesian posterior where the mean and variance are unknown!

	Parameters
	----------
	mean : float
		the mean of the normal distribution
	variance : float
		the variance of the normal distribution
	rng : multibeep.util.rng_class
		a random number generator object

	"""
	def __init__(self, float_t mean, float_t variance, rng_class rng):
		self.tmpptr = new arms_cpp.normal_arm[float_t, rand_t] (mean, variance, rng.get_shared_ptr())
		self.thisptr = shared_ptr[ arms_cpp.base[float_t, rand_t] ] (self.tmpptr)
		self.tmpptr = NULL

cdef class data(base):
	"""		
	Parameters
	----------
	data : numpy.ndarray (1d)
		The data for this arme. The data is copied at least once, but if the data is not in
		C order, a second temporary copy is made.
	
	name : string
		the name associated with this arm
	
	rng : multibeep.util.rng_class
		a random number generator object
	
	bootstrap : bool
		If false, the rewards are returned in sequential order
		starting with the first when the end is reached.
		
		If true, an entry is chosen uniformly at random (with replacement)
		
		Default is False.
		
	"""
	def __init__(self, np.ndarray[float_t, ndim=1] data, name, rng_class rng, bootstrap=False):
		data = np.ascontiguousarray(data)
		if bootstrap:
			self.tmpptr = new arms_cpp.data_arm_bootstrap[float_t, rand_t] (&data[0], data.shape[0], name, rng.get_shared_ptr())
		else:
			self.tmpptr = new arms_cpp.data_arm_sequential[float_t, rand_t] (&data[0], data.shape[0], name, rng.get_shared_ptr())
		self.thisptr = shared_ptr[ arms_cpp.base[float_t, rand_t] ] (self.tmpptr)
		self.tmpptr = NULL




# moderator functions between C++ and python

cdef float_t pull_wrapper(void *obj):
	# recover python object from the C++ pointer to the python pull function
	o = <object> obj
	# call it and cast the result to be a float_t
	return (<float_t> o.pull())

cdef float_t mean_wrapper(void *obj):
	o = <object> obj
	return (<float_t> o.real_mean())

cdef float_t var_wrapper(void *obj):
	o = <object> obj
	return (<float_t> o.real_variance())

cdef shared_ptr[util_cpp.base[float_t, rand_t] ] posterior_wrapper (void *obj):
	o = <object> obj
	p = <posterior_class> o.posterior()
	return (p.get_shared_ptr())

cdef void deactivate_wrapper(void *obj):
	o = <object> obj
	o.deactivate()


cdef class python(base):
	def __init__(self, obj, name="custom python arm"):
		self.tmpptr = new arms_cpp.python_arm[float_t,rand_t](name, <void*> obj, &pull_wrapper, &mean_wrapper, &var_wrapper, &posterior_wrapper, &deactivate_wrapper)
		self.thisptr = shared_ptr[ arms_cpp.base[float_t, rand_t] ] (self.tmpptr)
		self.tmpptr = NULL	
	def pull(self):
		raise ("To use a python_arm, you have to override its pull method such that it returns a reward")
	def real_mean(self):
		return(float('NaN'))
	def real_variance(self):
		return(float('NaN'))
	def posterior(self):
		raise ("This arm does not provide a posterior. So you either use a bandit that provides one or provide a posterior for the arm.")
	def deactivate(self):
		pass
