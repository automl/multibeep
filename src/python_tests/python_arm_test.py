
import os
print(os.getcwd())

import sys
sys.path.append("../../")


import multibeep as mb

import numpy as np
import matplotlib.pyplot as plt


rng = mb.util.rng_class(125)



a = mb.arms.normal(0,1,rng)

print(a.pull())



N = 10000
delta = 0.1
bandit = mb.bandits.last_n_pulls(n=2)

means = [0.5, 0.6, 0.7, 0.8]
variances = [0.2, 0.5, 1, 0.5] 
bootstrap = [False, True, False, True]
arms = []


# fill bandit with some arms
for m,v,b in zip(means, variances, bootstrap):
	data = np.sqrt(v) * np.random.randn(N) + m
	a = mb.arms.data(data, bytes("%f"%m,'UTF-8'), rng, bootstrap = b)
	arms.append(a)

names = [a.get_ident() for a in arms]

[bandit.add_arm(a) for a in arms]

print([a.real_mean() for a in arms])
print([a.real_variance() for a in arms])
# delete all python handles of the arms (so you cannot mess with them anymore
# outside of the bandit!
arms=[]



# or just add an arm without storing a python object holding it in the first place
#bandit.add_arm(mb.bernoulli_arm(0.7, rng))
#names.append("2nd bernoulli arm")


# create a policy that will play the bandit
#policy = mb.policies.random(bandit, rng)
policy = mb.policies.UCB_p(bandit, rng, 2)
#policy = mb.policies.prob_match(bandit, rng)


policy.play_n_rounds(N)


