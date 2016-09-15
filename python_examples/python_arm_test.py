import sys
sys.path.append("..")


from IPython import embed
import multibeep as mb

import numpy as np
import matplotlib.pyplot as plt


rng = mb.util.rng_class(125)



a = mb.arms.normal(0,1,rng)

print(a.pull())



N = 10000
delta = 0.1
bandit = mb.bandits.empirical()

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

for i in range(bandit.number_of_arms()):
	print("="*50)
	col = plt.cm.rainbow(i/(len(names)-1))
	#rewards = np.array([bandit.pull_by_identifier(i) for _ in range(N)])
	
	print(i,(bandit[i]).is_active,(bandit[i]).num_pulls)
	
	#print(rewards)
	p = bandit[i].posterior

	if p.valid():
		xmin, xmax = p.support(delta)
		print(bandit[i].identifier, bandit[i].num_pulls, xmin, xmax, bandit[i].real_mean, p.mean(), np.sqrt(p.variance()))

		xs = np.linspace(xmin, xmax, 100)
		ys = [p.pdf(x) for x in xs]

		print((xs[1]-xs[0]) * np.sum(ys[:-1] + ys[1:])/2 , " ?= %f"%(1-delta))

		plt.errorbar(p.mean(), np.max(ys)/2, xerr = np.sqrt(p.variance()), c=col)

		plt.plot(xs,ys, label = "pdf %s"%names[i], c=col)
		ys = [p.cdf(x) for x in xs]
		plt.plot(xs,ys, label = "cdf %s"%names[i], c=col)

bandit.update_p_max(False, 0.01, 128)
print ([bandit[i].p_max for i in range(bandit.number_of_arms())])

plt.legend()
plt.show()


embed()


