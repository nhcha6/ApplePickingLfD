from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Activation
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import pandas as pd
import pickle
from create_model_data import *
from matplotlib import pyplot as plt

components = 4  # Number of components in the mixture
no_parameters = 3  # Paramters of the mixture (alpha, mu, sigma)
no_dimensions = 3  # Dimensions of output distribution (x, y, z)

def read_data(file_name_train):
    with open('model data/x_' + file_name_train + '.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open('model data/y_' + file_name_train + '.pkl', 'rb') as f:
        y_train = pickle.load(f)

    return x_train, y_train

def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def gnll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha = parameter_vector[:,0:components]
    mu = parameter_vector[:, components:components*(1+no_dimensions)]
    mu = tf.reshape(mu, (-1,components, no_dimensions))
    sigma = parameter_vector[:,components*(1+no_dimensions):components*(2+2*no_dimensions)]
    sigma = tf.reshape(sigma, (-1, components, no_dimensions))

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.MultivariateNormalDiag(loc=mu,
                                                           scale_diag=sigma))

    log_likelihood = gm.log_prob(y)  # Evaluate log-probability of y
    error = -tf.reduce_mean(log_likelihood, axis=-1)

    # tf.print('\n')
    # tf.print(alpha)
    # tf.print(mu)
    # tf.print(sigma)
    # tf.print(y)
    # tf.print(log_likelihood)
    # tf.print(error)

    return error

def build_model(x_train, y_train, epochs, batch_size, neurons):
    # add custom nnelu function as an activation function
    tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

    n_samples = x_train.shape[0]
    n_timesteps = x_train.shape[1]
    n_features = x_train.shape[2]

    inputs = Input(shape=(n_timesteps, n_features))

    lstm_1 = LSTM(neurons, activation='relu', kernel_initializer=Orthogonal())
    h1 = lstm_1(inputs)

    alphas = Dense(components, activation="softmax", name="alphas")(h1)
    mus = Dense(components*no_dimensions, name="mus")(h1)
    sigmas = Dense(components*no_dimensions, activation="nnelu", name="sigmas")(h1)
    pvector = Concatenate(name="output")([alphas, mus, sigmas])

    model = Model(inputs=inputs, outputs=pvector, name="model")
    print(model.summary())

    model.compile(loss=gnll_loss, optimizer=Adam())
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model

def calulate_input_vector(ee_pos, point_cloud, goal):
    [repulsive_ee, repulsive_closest] = calculate_weighted_centre(ee_pos, point_cloud, goal, goal_flag=False)
    [repulsive_goal, closest_point] = calculate_weighted_centre(ee_pos, point_cloud, goal, goal_flag=True)

    # calculate vectors between goal/repulsive points
    attractive = np.subtract(goal, ee_pos)
    repulsive_ee = np.subtract(ee_pos, repulsive_ee)
    repulsive_goal = np.subtract(ee_pos, repulsive_goal)
    repulsive_closest = np.subtract(ee_pos, repulsive_closest)

    # normalise each one
    attractive_norm = norm_vector(attractive)
    repulsive_ee_norm = norm_vector(repulsive_ee)
    repulsive_goal_norm = norm_vector(repulsive_goal)
    repulsive_closest_norm = norm_vector(repulsive_closest)

    args = (attractive_norm, repulsive_ee_norm, repulsive_goal_norm, repulsive_closest_norm)
    input_vector = np.concatenate(args)
    return input_vector

def train_test_model():
    ############ train model ###########
    train_x, train_y = read_data('train_single_sample')
    print(train_x.shape)
    print(train_y.shape)

    model = build_model(train_x, train_y, epochs = 15, batch_size=8, neurons=64)

    ############ TEST MODEL ##############
    for file_name in ['test_2']:
        # import trajectory and environment information
        traj_df = pd.read_pickle('trajectory data/' + file_name + '.pkl')
        goal_df = pd.read_pickle('trajectory data/' + file_name + '_goal.pkl')
        point_cloud_df = pd.read_pickle('trajectory data/' + file_name + '_point_cloud.pkl')

        # convert to usable form
        point_cloud = []
        for index, row in point_cloud_df.iterrows():
            point_cloud.append([row[0], row[1], row[2]])
        goal = [goal_df.iloc[0, 0], goal_df.iloc[1, 0], goal_df.iloc[2, 0]]
        current_pos = [-traj_df.iloc[0]['traj'][3], -traj_df.iloc[0]['traj'][7], traj_df.iloc[0]['traj'][11]]

        # while True:
        input_data = np.zeros((1, 20, train_x.shape[2]))
        distances = []
        while True:
            # calculate input vector for this timestep
            input_vector = calulate_input_vector(current_pos, point_cloud, goal)

            # update input vector
            shape = input_data.shape
            input_data = np.delete(input_data, [i for i in range(12)])
            input_data = np.append(input_data, input_vector)
            input_data = input_data.reshape((shape[0], shape[1], shape[2]))

            # make prediction
            pred = model.predict(input_data)
            parameter_vector = pred[0]

            # extract sample from generated probability distribution
            alpha = parameter_vector[0:components]
            mu = parameter_vector[components:components * (1 + no_dimensions)]
            mu = tf.reshape(mu, (components, no_dimensions))
            sigma = parameter_vector[components * (1 + no_dimensions):components * (2 + 2 * no_dimensions)]
            sigma = tf.reshape(sigma, (components, no_dimensions))

            gm = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.MultivariateNormalDiag(loc=mu,
                                                                   scale_diag=sigma))

            delta_pos = gm.sample([1])[0].numpy()
            new_pos = np.add(current_pos, delta_pos)
            print('\n')
            print(current_pos)
            print(new_pos)
            print(goal)

            dist = np.linalg.norm(np.subtract(new_pos, goal))
            print(dist)
            distances.append(dist)

            # need better smoothing and termination criteria
            if len(distances) > 5:
                if dist - distances[-5] < 0.2:
                    current_pos = new_pos
                if dist < 0.3:
                    break

        # analyse results
        plt.figure()
        plt.plot(distances)
        plt.show()


