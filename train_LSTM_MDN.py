from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Activation
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import pickle
from create_model_data import *
from matplotlib import pyplot as plt

components = 4  # Number of components in the mixture
no_parameters = 3  # Paramters of the mixture (alpha, mu, sigma)
no_dimensions = 3  # Dimensions of output distribution (x, y, z)

max_movement = 0.4
stop_distance = 0.4

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

def swept_error_area(real_traj, generated_traj):
    closest_points = []
    distances = []
    for generated_point in generated_traj:
        closest_point = None
        min_dist = 100
        for real_point in real_traj:
            real_point = [-real_point[3], -real_point[7], real_point[11]]
            dist = np.linalg.norm(np.subtract(real_point, generated_point))
            if dist<min_dist:
                closest_point = real_point
                min_dist = dist
        closest_points.append(closest_point)
        distances.append(dist)
        # print(closest_point)
        # print(generated_point)
        # print('\n')

    total_area = 0
    for i in range(len(generated_traj)-1):

        triag_1 = [generated_traj[i], generated_traj[i+1], closest_points[i]]
        triag_2 = [generated_traj[i+1], closest_points[i+1], closest_points[i]]

        # print('\n')
        # print(triag_2)
        # print(triag_1)

        for triag in [triag_1, triag_2]:
            v_ab = np.subtract(triag[1], triag[0])
            v_ac = np.subtract(triag[2], triag[0])
            cross = np.cross(v_ab, v_ac)
            # print(cross)
            area = 0.5*np.linalg.norm(cross)
            # print(area)
            total_area += area
        #     print(area)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.plot([p[0] for p in generated_traj], [p[1] for p in generated_traj], [p[2] for p in generated_traj])
        # ax.plot([p[0] for p in closest_points], [p[1] for p in closest_points], [p[2] for p in closest_points])
        # ax.plot([generated_traj[i][0], generated_traj[i + 1][0], closest_points[i][0], closest_points[i + 1][0]],
        #         [generated_traj[i][1], generated_traj[i + 1][1], closest_points[i][1], closest_points[i + 1][1]],
        #         [generated_traj[i][2], generated_traj[i + 1][2], closest_points[i][2], closest_points[i + 1][2]], 'o')
        # plt.legend()
        # plt.show()

    # print(total_area)
    # print(np.mean(dist))

    return total_area

def train_test_model():
    ############ train model ###########
    train_x, train_y = read_data('train_single_sample')
    print(train_x.shape)
    print(train_y.shape)

    model = build_model(train_x, train_y, epochs = 30, batch_size=8, neurons=50)

    ############ TEST MODEL ##############
    for file_name in ['test_2']:
        # import trajectory and environment information
        traj_df = pd.read_pickle('trajectory data/' + file_name + '.pkl')
        goal_df = pd.read_pickle('trajectory data/' + file_name + '_goal.pkl')
        point_cloud_df = pd.read_pickle('trajectory data/' + file_name + '_point_cloud.pkl')

        swept_error_area(traj_df['traj'].values, [[-0.010148664005100727, 0.2371896654367447, 0.8040667772293091], [-0.035649736411869526, 0.20717388950288296, 0.7706105895340443], [-0.06376057211309671, 0.20026960968971252, 0.7584957601502538], [-0.09912641439586878, 0.18019162118434906, 0.7405795166268945], [-0.09966458415146917, 0.11886832118034363, 0.6899319188669324], [-0.06884158251341432, 0.14230557531118393, 0.6528893345966935], [-0.04095121507998556, 0.04624660313129425, 0.6590271648019552], [-0.012194389826618135, -0.022728487849235535, 0.6512554995715618], [0.014337816857732832, -0.024133998900651932, 0.6630082409828901], [0.026575917028822005, -0.04136800393462181, 0.6809714715927839], [0.055449640029110014, -0.07100438885390759, 0.6672618836164474], [0.06397608353290707, -0.06987248547375202, 0.7011527009308338], [0.10822953714523464, -0.10554221458733082, 0.7032382795587182], [0.1528710393467918, -0.1082378439605236, 0.6706166164949536], [0.18061221262905747, -0.09485810715705156, 0.6850726818665862], [0.21182763820979744, -0.13104575965553522, 0.7155462997034192]])

        # convert to usable form
        point_cloud = []
        for index, row in point_cloud_df.iterrows():
            point_cloud.append([row[0], row[1], row[2]])
        goal = [goal_df.iloc[0, 0], goal_df.iloc[1, 0], goal_df.iloc[2, 0]]

        point_list = []
        for i in range(3):
            current_pos = [-traj_df.iloc[0]['traj'][3], -traj_df.iloc[0]['traj'][7], traj_df.iloc[0]['traj'][11]]
            input_data = np.zeros((1, 20, train_x.shape[2]))
            distances = []
            points = [current_pos]
            # while True:
            for i in range(30):
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
                print(delta_pos)
                print(current_pos)
                print(new_pos)
                print(goal)

                dist = np.linalg.norm(np.subtract(new_pos, goal))
                print(dist)
                distances.append(dist)

                # restart criteria for a bad run
                # if np.linalg.norm(delta_pos)>0.2:
                # if dist > 1.5:
                #     current_pos = [-traj_df.iloc[0]['traj'][3], -traj_df.iloc[0]['traj'][7],
                #                    traj_df.iloc[0]['traj'][11]]
                #     points = [current_pos]
                #     continue

                # do not allow excessively large changes in ee pos
                if np.linalg.norm(delta_pos) > max_movement:
                    continue

                current_pos = new_pos
                points.append(list(current_pos))

                # termination criteria: distance to goal
                if dist < stop_distance:
                    break

            point_list.append(points)
        # analyse results
        # plt.figure()
        # plt.plot(distances)


        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for points in point_list:
            error = swept_error_area(traj_df['traj'].values, points)
            ax.plot([p[0] for p in points], [p[1] for p in points], [p[2] for p in points], label = str(error))
        ax.plot([-p[3] for p in traj_df['traj']], [-p[7] for p in traj_df['traj']], [p[11] for p in traj_df['traj']], 'y', linewidth = 4, label='demonstration')
        ax.plot([goal[0]], [goal[1]], [goal[2]], 'go')
        plt.legend()
        plt.show()


