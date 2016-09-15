import sys
sys.path.append("..")


from IPython import embed
import multibeep as mb

import numpy as np
import matplotlib.pyplot as plt


rng = mb.util.rng_class(0)

num_rounds = 8



bandit = mb.bandits.last_n_pulls()




# fill bandit with some arms
arms = [mb.arms.bernoulli(0.75, rng), mb.arms.exponential(4/3, rng),
		mb.arms.normal(0.8,0.25, rng), mb.arms.bernoulli(0.7, rng)]
names = [a.get_ident() for a in arms]
[bandit.add_arm(a) for a in arms]

print([a.real_mean() for a in arms])


# delete all python handles of the arms (so you cannot mess with them anymore
# outside of the bandit!
#arms=[]



# or just add an arm without storing a python object holding it in the first place
#bandit.add_arm(mb.bernoulli_arm(0.7, rng))
#names.append("2nd bernoulli arm")


# create a policy that will play the bandit
#policy = mb.policies.random(bandit, rng)
#policy = mb.UCB_p(bandit, rng, 2)
#policy = mb.prob_match(bandit, rng)

policy = mb.policies.successive_halving(bandit, 1, 3)


print(policy.select_next_arm())

#policy.play_n_rounds(num_rounds)

for i in range(bandit.number_of_arms()):
	print("="*50)
	col = plt.cm.rainbow(i/(len(names)-1))
	#rewards = np.array([bandit.pull_by_identifier(i) for _ in range(N)])
	
	print(i,(bandit[i]).is_active,(bandit[i]).num_pulls)
	
	#print(rewards)
	p = bandit[i].posterior

	if p.valid():
		xmin, xmax = p.support(delta)
		print(bandit[i].identifier, bandit[i].num_pulls, xmin, xmax, arms[i].real_mean(), p.mean(), np.sqrt(p.variance()))

		xs = np.linspace(xmin, xmax, 100)
		ys = [p.pdf(x) for x in xs]

		print((xs[1]-xs[0]) * np.sum(ys[:-1] + ys[1:])/2 , " ?= %f"%(1-delta))

		plt.errorbar(p.mean(), np.max(ys)/2, xerr = np.sqrt(p.variance()), c=col)

		plt.plot(xs,ys, label = "pdf %s"%names[i], c=col)
		ys = [p.cdf(x) for x in xs]
		plt.plot(xs,ys, label = "cdf %s"%names[i], c=col)

plt.legend()
plt.show()


