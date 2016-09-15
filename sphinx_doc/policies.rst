.. multibeep documentation master file, created by
   sphinx-quickstart on Fri Sep  2 14:22:32 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The policies submodule
=====================================

Contents:

.. automodule:: multibeep.policies
	:members:
	:undoc-members:
	:show-inheritance:




TODO
=========

Policies that need to be coded:

1.	UCB1, UCB1-NORMAL
		P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite time analysis of the multiarmed bandit problem.
		Machine Learning, 47(2-3):235–256, 2002
2.	lil-UCB
		Jamieson et al.
		lil’ UCB : An Optimal Exploration Algorithm for Multi-Armed Bandits
		JMLR: Workshop and Conference Proceedings vol 35:1–17, 2014
3.	KL-UCB
		Aurélien Garivier, Olivier Cappé: The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond. COLT 2011: 359-376
4.	MOSS
		J-Y. Audibert and S. Bubeck: Minimax Policies for Adversarial and Stochastic Bandits.  Proceedings of the 22nd Annual Conference on Learning Theory 2009
5.	Epsilon-Greedy
		http://cs.mcgill.ca/~vkules/bandits.pdf
6.	SoftMax/Boltzmann Exploration
		http://cs.mcgill.ca/~vkules/bandits.pdf
7.	Poker:
		Vermorel, Joannes, and Mehryar Mohri.
		"Multi-armed bandit algorithms and empirical evaluation."
		European conference on machine learning. Springer Berlin Heidelberg, 2005.
8.	f-race
		Birattari, Mauro, et al.
		"F-Race and iterated F-Race: An overview."
		Experimental methods for the analysis of optimization algorithms. Springer Berlin Heidelberg, 2010. 311-336.
9.	Entropy search with different update rules
		a.	proper predictive posterior to compute expected change in H(pmax)
		
		c.	Sampling from the reward history to approximate expected change in H(pmax)
		
		b.	Joel's sampling from the posterior and pretending the mean is fixed (expected 'information of one mean')
		

