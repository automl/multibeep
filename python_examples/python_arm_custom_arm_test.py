import sys
sys.path.append("..")


from IPython import embed
import multibeep as mb

import numpy as np
import matplotlib.pyplot as plt


delta = 0.01
rng = mb.util.rng_class(0)


class python_normal_arm_known_variance(mb.arms.python):
	def __init__(self, mu, sigma):
		self.mu = mu
		self.sigma = sigma
		# you need to call the base class' init method with these arguments:
		# a reference to this object, the true mean and variance
		super().__init__(self, bytes('example arm {} {}'.format(mu,sigma), 'UTF-8'))
		
		self.rewards = []
		
	def pull(self):
		r = (self.sigma*np.random.randn() + self.mu)
		self.rewards.append(r)
		return(r)

	def real_mean(self):
		return(self.mu)
	def real_variance(self):
		return(self.sigma**2)

	def posterior(self):
		r=self.rewards
		return(mb.util.gaussian_posterior( np.mean(r), np.std(r)/len(r)))


arms = list(map(lambda t: python_normal_arm_known_variance(*t), [(0.5,0.25), (0.6,0.25),(0.55,0.25)]))

#bandit = mb.bandits.posterior_bandit()
bandit = mb.bandits.empirical_bandit()
bandit = mb.bandits.last_n_pulls(n=2)
[bandit.add_arm(a) for a in arms]


print([a.real_mean() for a in arms])

policy = mb.policies.prob_match(bandit, rng)
policy.play_n_rounds(10000)


for i in range(bandit.number_of_arms()):
	print("="*50)
	col = plt.cm.rainbow(i/(len(arms)-1))
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

		plt.plot(xs,ys, c=col, label = bandit[i].name)
		ys = [p.cdf(x) for x in xs]
		plt.plot(xs,ys, c=col)

plt.legend(loc=2)
plt.show()
embed()
