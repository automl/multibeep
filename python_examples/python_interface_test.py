import sys
sys.path.append("..")


from IPython import embed
import multibeep as mb

import numpy as np
import matplotlib.pyplot as plt


rng = mb.util.rng_class(np.random.randint(4096))
N = 1000
delta = 0.001
min_pulls=5



#bandit = mb.bandits.empirical_bandit()
bandit = mb.bandits.posterior_bandit()



# fill bandit with some arms
arms = [mb.arms.bernoulli(0.64, rng), mb.arms.exponential(5/3, rng),
		mb.arms.normal(0.8,0.25, rng), mb.arms.bernoulli(0.7, rng)]
names = [a.get_ident() for a in arms]
[bandit.add_arm(a) for a in arms]


# delete all python handles of the arms, so you cannot mess with them anymore
# outside of the bandit!
arms=[]



# or just add an arm without storing a python object holding it in the first place
bandit.add_arm(mb.arms.bernoulli(0.82, rng))
names.append("2nd bernoulli arm")


# create a policy that will play the bandit
#policy = mb.policies.random(bandit, rng)
#policy = mb.policies.UCB_p(bandit, rng, 2)
policy = mb.policies.prob_match(bandit, rng)


real_means = [];

for i in range(bandit.number_of_arms()):
	real_means.append(bandit[i].real_mean)
	for j in range(min_pulls):
		bandit.pull_by_index(i)
print(real_means)

policy.play_n_rounds(N)

for i in range(bandit.number_of_arms()):
	print("="*50)
	col = plt.cm.rainbow(i/(len(names)-1))
	#rewards = np.array([bandit.pull_by_identifier(i) for _ in range(N)])
	
	print(i,(bandit[i]).is_active,(bandit[i]).num_pulls)

	plt.axvline( bandit[i].real_mean, c=col)
	
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
		


plt.legend()
plt.show()


