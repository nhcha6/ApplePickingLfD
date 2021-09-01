from tensorflow_probability import distributions as tfd

mvn_1 = tfd.MultivariateNormalDiag(loc=[0,0,0], scale_diag=[1,1,1])
mvn_2 = tfd.MultivariateNormalDiag(loc=[2,2,2], scale_diag=[1,1,1])

gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.2, 0.8]),
    components_distribution = tfd.MultivariateNormalDiag(loc=[[0,0,0],[2,2,2]], scale_diag = [[1, 1, 1], [1,1,1]]))

sample = [1.5,1.5,1.5]

print(gm.prob(sample))

print(mvn_1.prob(sample))
print(mvn_2.prob(sample))