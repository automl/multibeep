import sys
sys.path.append("..")


from IPython import embed
import multibeep as mb

import numpy as np
import matplotlib.pyplot as plt


rng = mb.util.rng_class(51)

N_init=40
N = 2000
delta = 0.001
bandit = mb.bandits.empirical_bandit()



with open('../data/autofolio.csv', 'r') as f:
    first_line = f.readline()


data = -np.loadtxt('../data/autofolio.csv', skiprows=1, delimiter=',', converters= {0: lambda s: 0})[:,1:]


np.random.shuffle(data)

print(data.shape)
data[data == -12000] = -1200

names = first_line.strip().split(',')[1:]
names = list(map(lambda s: bytes(s, 'utf-8'), names))


arms = []

for i in range(data.shape[1]):
	arms.append(mb.arms.data(data[:,i], names[i], rng, True))






[bandit.add_arm(a) for a in  arms]
print( list(bandit.pull_by_index(i) for i in range(len(arms))))

bandit.min_pull_arms(N_init)

print([ bandit[i].num_pulls for i in range(bandit.number_of_active_arms())])
print([ bandit[i].is_active for i in range(bandit.number_of_active_arms())])

# create a policy that will play the bandit
#policy = mb.policies.random(bandit, rng)
#policy = mb.policies.UCB_p(bandit, rng, 2)
policy = mb.policies.prob_match(bandit, rng)




print([ bandit[i].num_pulls for i in range(bandit.number_of_active_arms())])
print([ bandit[i].is_active for i in range(bandit.number_of_active_arms())])


for i in range(N//10):
	policy.play_n_rounds(10)
	bandit.deactivate_by_confidence_gap(delta)
	if bandit.number_of_active_arms() == 1:
		break

print("finished playing")

print([ bandit[i].num_pulls for i in range(bandit.number_of_active_arms())])
print([ bandit[i].is_active for i in range(bandit.number_of_active_arms())])


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


