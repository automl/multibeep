import sys
sys.path.append("..")


from IPython import embed
import multibeep as mb

import numpy as np
import matplotlib.pyplot as plt


rng = mb.util.rng_class(0)

max_pulls = 1024
budged = 10 * max_pulls

bandit = mb.bandits.last_n_pulls(3)



means = np.random.rand(1024)
variances = np.random.rand(1024)/10

print(means,variances)

arms = [mb.arms.normal(m,v,rng) for (m,v) in zip(means, variances)]
[bandit.add_arm(a) for a in arms]


# create a policy that will play the bandit
#policy = mb.policies.random(bandit, rng)
#policy = mb.policies.UCB_p(bandit, rng, 2)
#policy = mb.policies.prob_match(bandit, rng)
policy = mb.policies.successive_halving(bandit,1,2)
print("start playing")
policy.play_n_rounds(6)
print("finished playing")

print([ bandit[i].num_pulls for i in range(bandit.number_of_active_arms())])
print([ bandit[i].is_active for i in range(bandit.number_of_active_arms())])

exit(0)

for i in range(bandit.number_of_arms()):
	print("="*50)
	col = plt.cm.rainbow(bandit[i].identifier/(len(names)-1))

	p = bandit[i].posterior

	if p.valid():
		xmin, xmax = p.support(delta)
		print(bandit[i].name, bandit[i].num_pulls, bandit[i].real_mean, p.mean(), np.sqrt(bandit[i].real_variance), np.sqrt(p.variance()))

		xs = np.linspace(xmin, xmax, 100)
		ys = [p.pdf(x) for x in xs]

		print((xs[1]-xs[0]) * np.sum(ys[:-1] + ys[1:])/2 , " ?= %f"%(1-delta))

		plt.errorbar(-p.mean(), np.max(ys)/2, xerr = np.sqrt(p.variance()), c=col)

		plt.plot(-xs,ys, label = "pdf %s"%bandit[i].name, c=col)
		ys = [p.cdf(x) for x in xs]
		#plt.plot(-xs,ys, label = "cdf %s"%names[i], c=col)
		plt.axvline(-bandit[i].real_mean, c=col)
		
		

plt.legend(loc=2)
plt.show()


