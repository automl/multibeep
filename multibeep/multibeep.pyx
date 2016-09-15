import cython
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr


import numpy as np
cimport numpy as np

ctypedef double float_t

cimport cpp_classes as c

ctypedef c.default_random_engine rng_t




####################################################################
# utils section
####################################################################


cdef class rng_class:
	cdef shared_ptr[rng_t] thisptr
	def __cinit__(self, cython.int seed):
		self.thisptr = shared_ptr[rng_t] (new rng_t(seed))

	cdef shared_ptr[rng_t] get_shared_ptr(self):
		return(self.thisptr)



cdef class posterior_class:
	#cdef c.posterior_base[float_t] *thisptr
	cdef shared_ptr[c.posterior_base[float_t]] thisptr

	def __init__(self):
		pass

	cpdef valid(self):
		if (self.thisptr):
			return(True)
		else:
			return(False)

	def mean(self):
		if self.valid():
			return(self.thisptr.get().mean())
		return(None)

	def variance(self):
		if self.valid():
			return(self.thisptr.get().variance())
		return(None)

	def pdf(self, float_t x):
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

####################################################################
# arm section
####################################################################
cdef class base:
	cdef shared_ptr[ c.base[float_t]] thisptr
	cdef c.base[float_t] * tmpptr

	def pull(self):
		return(self.thisptr.get().pull())
	def real_mean(self):
		return(self.thisptr.get().real_mean())
	def real_variance(self):
		return(self.thisptr.get().real_variance())
	def get_ident(self):
		return(self.thisptr.get().get_ident())
	def provides_posterior(self):
		return(self.thisptr.get().provides_posterior())
	def posterior(self):
		p = posterior_class()
		p.thisptr = deref(self.thisptr.get()).posterior()
		return(p)
	#cdef shared_ptr[c.posterior_base[float_t] ] get_posterior_pointer(self):
	#	return(deref(self.thisptr).posterior())
	cdef shared_ptr[c.base[float_t] ] get_arm_ptr (self):
		return(self.thisptr)



cdef class bernoulli_arm(base):
	def __cinit__(self, float_t p, rng_class rng):
		self.tmpptr = new c.bernoulli_arm[rng_t, float_t] (p, rng.get_shared_ptr())
		self.thisptr = shared_ptr[ c.base[float_t] ] (self.tmpptr)
		self.tmpptr = NULL

cdef class exponential_arm(base):
	def __init__(self, float_t l, rng_class rng):
		self.tmpptr = new c.exponential_arm[rng_t, float_t] (l, rng.get_shared_ptr())
		self.thisptr = shared_ptr[ c.base[float_t] ] (self.tmpptr)
		self.tmpptr = NULL

cdef class normal_arm(base):
	def __init__(self, float_t mean, float_t variance, rng_class rng):
		self.tmpptr = new c.normal_arm[rng_t,float_t] (mean, variance, rng.get_shared_ptr())
		self.thisptr = shared_ptr[ c.base[float_t] ] (self.tmpptr)
		self.tmpptr = NULL

cdef class data_arm(base):
	def __init__(self, np.ndarray[float_t, ndim=1] data, name, rng_class rng, bootstrap=False):
		if bootstrap:
			self.tmpptr = new c.data_arm_bootstrap[rng_t,float_t] (&data[0], data.shape[0], name, rng.get_shared_ptr())
		else:
			self.tmpptr = new c.data_arm_sequential[rng_t,float_t] (&data[0], data.shape[0], name, rng.get_shared_ptr())

		self.thisptr = shared_ptr[ c.base[float_t] ] (self.tmpptr)
		self.tmpptr = NULL


# moderator function between C++ and python
cdef float_t pull_wrapper(void *obj):
	# recover python object from the C++ pointer to the python pull function
	o = <object> obj
	# call it and cast the result to be of float_t
	return (<float_t> o.pull())


cdef class python_base(base):
	def __init__(self, pyfunc, float_t mean, float_t variance):
		self.tmpptr = new c.python_arm[float_t](<void*> pyfunc, pull_wrapper, mean, variance)
		self.thisptr = shared_ptr[ c.base[float_t] ] (self.tmpptr)
		self.tmpptr = NULL	
	def pull(self):
		raise ("To use a python_arm, you have to override its pull method such that it returns a reward")
		

####################################################################
# bandit section
####################################################################
cdef class arm_info:
	cdef public unsigned int index
	cdef public unsigned int identifier
	cdef public bool is_active
	cdef public long double num_pulls
	cdef public float_t estimated_mean
	cdef public float_t estimated_variance

	cdef public vector[float_t] rewards


	cdef public float_t p_min
	cdef public float_t p_max


	cdef public posterior_class posterior

	cdef fill_attributes(self, const c.arm_info[float_t] * tmpptr, unsigned int i):
		self.identifier = tmpptr.identifier
		self.is_active = tmpptr.is_active
		self.num_pulls = tmpptr.num_pulls
		self.estimated_mean = tmpptr.estimated_mean
		self.estimated_variance = tmpptr.estimated_variance
		self.p_min = tmpptr.p_min
		self.p_max = tmpptr.p_max
		self.posterior = posterior_class()
		self.posterior.thisptr = deref(tmpptr).posterior
		self.rewards = (tmpptr.rewards)[:]

		#shared_ptr[const base[num_t] ] get_arm_ptr()




cdef class bandit:
	cdef shared_ptr[c.default_bandit[float_t] ] thisptr
	# temporary pointer, as a workaround for struggeling to write __init__ with a shared_ptr
	cdef  c.default_bandit[float_t] * tmpptr

	def add_arm(self, base arm):
		self.thisptr.get().add_arm(arm.get_arm_ptr())
	def deactivate_by_index(self, unsigned int index):
		self.thisptr.get().deactivate_by_index(index)
	def deactivate_by_identifier(self, unsigned int ident):
		self.thisptr.get().deactivate_by_identifier(ident)
	def reactivate_by_index(self, unsigned int index):
		self.thisptr.get().reactivate_by_index(index)
	def reactivate_by_identifier(self, unsigned int ident):
		self.thisptr.get().reactivate_by_identifier(ident)
	def pull_by_index(self, unsigned int index):
		return(self.thisptr.get().pull_by_index(index))
	def pull_by_identifier(self, unsigned int ident):
		return(self.thisptr.get().pull_by_identifier(ident))
	def number_of_arms(self):
		return(self.thisptr.get().number_of_arms())
	def number_of_active_arms(self):
		return(self.thisptr.get().number_of_active_arms())
	def number_of_pulls(self):
		return(self.thisptr.get().number_of_pulls())
	def number_of_pulled_arms(self):
		return(self.thisptr.get().number_of_pulled_arms())

	def __getitem__( self, int index):
		ai = arm_info()
		ai.fill_attributes(&(deref(self.thisptr)[index]), index)
		return(ai)

cdef class empirical_bandit(bandit):
	def __init__(self):
		self.tmpptr = new c.empirical_bandit[float_t]()
		self.thisptr = shared_ptr[c.default_bandit[float_t] ] (self.tmpptr)
		self.tmpptr = NULL


cdef class posterior_bandit(bandit):
	def __init__(self):
		self.tmpptr = new c.posterior_bandit[float_t]()
		self.thisptr = shared_ptr[c.default_bandit[float_t] ] (self.tmpptr)
		self.tmpptr = NULL



####################################################################
# policy section
####################################################################

cdef class policy_base:
	cdef c.policy_base[float_t]* thisptr

	def __dealloc__(self):
		del self.thisptr
	def select_next_arm(self):
		return(self.thisptr.select_next_arm())
	def play_n_rounds(self, cython.uint n):
		self.thisptr.play_n_rounds(n)


cdef class random_policy(policy_base):
	def __init__ (self, bandit b, rng_class rng):
		self.thisptr = new c.random[rng_t, float_t] (b.thisptr, rng.thisptr)

cdef class UCB_p(policy_base):
	def __init__ (self, bandit b, rng_class rng, float_t p):
		self.thisptr = new c.UCB_p[rng_t, float_t] (b.thisptr, rng.thisptr, p)

cdef class prob_match(policy_base):
	def __init__ (self, bandit b, rng_class rng):
		self.thisptr = new c.prob_match[rng_t, float_t] (b.thisptr, rng.thisptr)


cdef class successive_halving(policy_base):
	def __init__ (self, bandit b, unsigned int min_num_pulls, frac_arms, frac_pulls = None):
		if frac_pulls is None:
			self.thisptr = new c.successive_halving[float_t] (b.thisptr, min_num_pulls, frac_arms)
		else:
			self.thisptr = new c.successive_halving[float_t] (b.thisptr, min_num_pulls, frac_arms, frac_pulls)
