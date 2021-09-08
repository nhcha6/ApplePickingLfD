from tensorflow_probability import distributions as tfd
import tensorflow as tf

mvn_1 = tfd.MultivariateNormalDiag(loc=[0,0,0], scale_diag=[1,1,1])
mvn_2 = tfd.MultivariateNormalDiag(loc=[2,2,2], scale_diag=[1,1,1])

gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[[0.2, 0.8], [0.2, 0.8]]),
    # mixture_distribution=tfd.Categorical(probs=[[0.2, 0.8]]),
    # components_distribution = tfd.MultivariateNormalDiag(loc=[[[0,0,0],[2,2,2]]], scale_diag = [[[1, 1, 1], [1,1,1]]]))
    components_distribution = tfd.MultivariateNormalDiag(loc=[[[0,0,0],[2,2,2]], [[0,0,0],[2,2,2]]], scale_diag = [[[1, 1, 1], [1,1,1]], [[1, 1, 1], [1,1,1]]]))

sample = [[1.5,1.5,1.5], [2,2,2]]

log_prob = gm.log_prob(sample)
print(log_prob)
error = -tf.reduce_mean(log_prob, axis=-1)
print(error)

print(mvn_1.log_prob(sample))
print(mvn_2.log_prob(sample))