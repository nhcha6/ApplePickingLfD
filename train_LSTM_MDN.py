from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Activation
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import models
from random import randint
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import pickle
from create_model_data import *
from matplotlib import pyplot as plt
import csv
import time

no_parameters = 3  # Paramters of the mixture (alpha, mu, sigma)
no_dimensions = 3  # Dimensions of output distribution (x, y, z)

max_movement = 0.4
stop_distance = 0.35

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

def build_model(x_train, y_train, epochs, batch_size, neurons, layers, test_name):
    # add custom nnelu function as an activation function
    tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

    filepath = "model tests/current test/" + test_name + "-batch_size" + str(batch_size) + '-neurons' + str(neurons) + '-layers' + str(layers) + '-components' + str(components) + "-{epoch:2d}.hdf5"
    callback_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)

    n_samples = x_train.shape[0]
    n_timesteps = x_train.shape[1]
    n_features = x_train.shape[2]

    inputs = Input(shape=(n_timesteps, n_features))

    if layers == 2:
        lstm_1 = LSTM(neurons, activation='relu', return_sequences = True, kernel_initializer=Orthogonal())
        h1 = lstm_1(inputs)
        lstm_2 = LSTM(neurons, activation='relu', kernel_initializer=Orthogonal())
        h2 = lstm_2(h1)
        distribution_input = h2
    elif layers == 3:
        lstm_1 = LSTM(neurons, activation='relu', return_sequences = True, kernel_initializer=Orthogonal())
        h1 = lstm_1(inputs)
        lstm_2 = LSTM(neurons, activation='relu', return_sequences=True, kernel_initializer=Orthogonal())
        h2 = lstm_2(h1)
        lstm_3 = LSTM(neurons, activation='relu', kernel_initializer=Orthogonal())
        h3 = lstm_3(h2)
        distribution_input = h3
    elif layers == 6:
        lstm_1 = LSTM(neurons, activation='relu', return_sequences = True, kernel_initializer=Orthogonal())
        h1 = lstm_1(inputs)
        lstm_2 = LSTM(neurons, activation='relu', return_sequences=True, kernel_initializer=Orthogonal())
        h2 = lstm_2(h1)
        lstm_3 = LSTM(neurons, activation='relu', return_sequences=True, kernel_initializer=Orthogonal())
        h3 = lstm_3(h2)
        lstm_4 = LSTM(neurons, activation='relu', return_sequences=True, kernel_initializer=Orthogonal())
        h4 = lstm_4(h3)
        lstm_5 = LSTM(neurons, activation='relu', return_sequences=True, kernel_initializer=Orthogonal())
        h5 = lstm_5(h4)
        lstm_6 = LSTM(neurons, activation='relu', kernel_initializer=Orthogonal())
        h6 = lstm_6(h5)
        distribution_input = h6
    else:
        lstm_1 = LSTM(neurons, activation='relu', kernel_initializer=Orthogonal())
        h1 = lstm_1(inputs)
        distribution_input = h1

    alphas = Dense(components, activation="softmax", name="alphas")(distribution_input)
    mus = Dense(components*no_dimensions, name="mus")(distribution_input)
    sigmas = Dense(components*no_dimensions, activation="nnelu", name="sigmas")(distribution_input)
    pvector = Concatenate(name="output")([alphas, mus, sigmas])

    model = Model(inputs=inputs, outputs=pvector, name="model")
    print(model.summary())

    model.compile(loss=gnll_loss, optimizer=Adam())
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[callback_model])

    return model

def calulate_input_vector(ee_pos, point_cloud, goal):
    prev_time = time.time()
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

def convert_to_pose(point_list, goal):
    print(goal)
    pose_list = []
    for points in point_list:
        poses = []
        for point in points:
            v1 = np.array([0, 0, np.linalg.norm(np.subtract(goal, point))])
            v2 = np.array(np.subtract(goal,point))
            # print(v1)
            # print(v2)
            rotation_matrix = rotation_matrix_from_vectors(v1, v2)
            pose = []

def rotation_matrix_from_vectors(vec1, vec2):


    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def test_trajectory(folder_name, model, input_shape, num_tests = 3, plot=True, save_validation = False):
    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    mean_errors = []
    min_errors = []
    median_errors = []
    for file_name in onlyfiles:
        # get only trajectory file name
        if 'goal' in file_name:
            continue
        if 'point_cloud' in file_name:
            continue
        if 'RRT' in file_name:
            continue
        file_name = file_name[0:-4]
        # import trajectory and environment information
        traj_df = pd.read_pickle(folder_name + file_name + '.pkl')
        goal_df = pd.read_pickle(folder_name + file_name + '_goal.pkl')
        point_cloud_df = pd.read_pickle(folder_name + file_name + '_point_cloud.pkl')

        # convert to usable form
        point_cloud = []
        for index, row in point_cloud_df.iterrows():
            if index%10:
                continue
            point_cloud.append([row[0], row[1], row[2]])
        goal = [goal_df.iloc[0, 0], goal_df.iloc[1, 0], goal_df.iloc[2, 0]]

        point_list = []
        planning_times = []
        for j in range(num_tests):
            start = time.time()
            current_pos = [-traj_df.iloc[0]['traj'][3], -traj_df.iloc[0]['traj'][7], traj_df.iloc[0]['traj'][11]]
            input_data = np.zeros((1, 20, input_shape[2]))
            distances = []
            min_dist = 100
            points = [current_pos]
            for i in range(100):
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
                if dist<min_dist:
                    min_dist = dist

                # # restart criteria for a bad run
                # if i > 30 and dist>1.5:
                #     break

                # do not allow excessively large changes in ee pos
                if np.linalg.norm(delta_pos) > max_movement:
                    continue

                current_pos = new_pos
                points.append(new_pos)

                # termination criteria: distance to goal
                if dist < stop_distance:
                    break

                # termination criteria: distance is much greater than the min dist
                if dist - min_dist > 0.2:
                    break

            end = time.time()
            planning_times.append(end - start)
            min_index = distances.index(min_dist)
            point_list.append(points[0:min_index+2])

        if save_validation:
            # print final distances to see if any didn't make it to within 0.35
            i = 0
            for point in point_list:
                print(np.linalg.norm(np.subtract(point[-1], goal)))
                gen_traj_df = pd.DataFrame(point)
                gen_traj_df.to_csv('validation results/' + file_name + '_traj_' + str(i) + '.csv', index=False)
                i+=1

            planning_times_df = pd.DataFrame(data=planning_times)
            print(planning_times_df)
            planning_times_df.to_csv('validation results/' + file_name +  '_planning_times.csv', index=False)
            goal_df.to_csv('validation results/' + file_name +  '_goal_pos.csv', index=False)


        # plot the demonstrated trajectories, the goal and the tree
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot([p[0] for p in point_cloud], [p[1] for p in point_cloud],[p[2] for p in point_cloud], 'bo', markersize=2, alpha=0.2)
            ax.plot([-p[3] for p in traj_df['traj']], [-p[7] for p in traj_df['traj']],
                    [p[11] for p in traj_df['traj']], 'y', linewidth=4, label='demonstration')
            ax.plot([goal[0]], [goal[1]], [goal[2]], 'go')

        # for each generated trajectory, plot and calculate the swept error area
        errors = []
        for points in point_list:
            error = swept_error_area(traj_df['traj'].values, points)
            errors.append(error)
            if plot:
                ax.plot([p[0] for p in points], [p[1] for p in points], [p[2] for p in points], label=str(error))

        # calculate mean and min errors
        mean_errors.append(np.mean(errors))
        min_errors.append(min(errors))
        median_errors.append(np.median(errors))

        # show plot
        if plot:
            plt.legend()
            plt.show()


    return mean_errors, min_errors, median_errors

def train_test_model(data_name, test_data):
    ############ train model ###########
    train_x, train_y = read_data(data_name)
    print(train_x.shape)
    print(train_y.shape)

    # optimal parameters
    # global components
    # components = 8
    # model = build_model(train_x, train_y, epochs = 150, batch_size=32, neurons=100, layers=3, test_name='grid_search')

    # optimal model from grid search
    global components
    components = 4
    filepath = "model tests/Grid Search/grid_search-batch_size16-neurons100-layers6-components4-100.hdf5"
    # filepath = "model tests/current test/grid_search-batch_size32-neurons100-layers3-components8-150.hdf5"
    model = models.load_model(filepath, custom_objects={'Activation': Activation(nnelu), 'nnelu': Activation(nnelu),
                                                        'gnll_loss': gnll_loss})

    ############ TEST MODEL ##############
    test_trajectory(test_data, model, train_x.shape, num_tests=10, save_validation=True)
    # test_trajectory('test trajectories/', model, train_x.shape, num_tests=10, save_validation=True)

def generate_random_params():
    batch_range = [16, 32, 128, 256, 512]
    layer_range = [1,3,6]
    component_range = [2,4, 8,16]
    neuron_range = [25,50,100,200,400]

    batch_size = batch_range[randint(0,len(batch_range)-1)]
    layers = layer_range[randint(0,len(layer_range)-1)]
    components = component_range[randint(0,len(component_range)-1)]
    neurons = neuron_range[randint(0,len(neuron_range)-1)]

    return batch_size, neurons, layers, components

def create_error_files():
    with open('model tests/current test/min_error.csv', mode='w') as file:
        employee_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(
            ['Batch Size', 'Neurons', 'Components', 'Layers', 'Epoch', 'Traj 1','Traj 2', 'Traj 3', 'Traj 4', 'Traj 5', 'Traj 6'])

    with open('model tests/current test/mean_error.csv', mode='w') as file:
        employee_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(
            ['Batch Size', 'Neurons', 'Components', 'Layers', 'Epoch', 'Traj 1','Traj 2', 'Traj 3', 'Traj 4', 'Traj 5', 'Traj 6'])

    with open('model tests/current test/median_error.csv', mode='w') as file:
        employee_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(
            ['Batch Size', 'Neurons', 'Components', 'Layers', 'Epoch', 'Traj 1','Traj 2', 'Traj 3', 'Traj 4', 'Traj 5', 'Traj 6'])

def run_grid_search(data_name, test_name):
    train_x, train_y = read_data(data_name)
    print(train_x.shape)
    print(train_y.shape)

    # create_error_files()

    # generate random models, train and assess error
    while True:
        # generate random model params
        batch_size, neurons, layers, components_local = generate_random_params()
        print(batch_size, neurons, layers, components_local)
        global components
        components = components_local

        # train model
        model = build_model(train_x, train_y, epochs=100, batch_size=batch_size, neurons=neurons, layers=layers, test_name=test_name)

        # iterate through epochs of interest
        for epoch in [10, 25, 50, 75, 100]:
            # import model and test on test trajectories
            filepath = "model tests/current test/" + test_name + "-batch_size" + str(batch_size) + '-neurons' + str(neurons) + '-layers' + str(layers) + '-components' + str(components) + "-" + str(epoch) + ".hdf5"
            model = models.load_model(filepath, custom_objects={'Activation': Activation(nnelu), 'nnelu': Activation(nnelu), 'gnll_loss': gnll_loss})
            mean_errors, min_errors, median_errors = test_trajectory('test trajectories/', model, train_x.shape, num_tests=5, plot=False)

            # generate the row to write to file
            min_error_row = [batch_size, neurons, components, layers, epoch]
            for err in min_errors:
                min_error_row.append(err)
            mean_error_row = [batch_size, neurons, components, layers, epoch]
            for err in mean_errors:
                mean_error_row.append(err)
            median_error_row = [batch_size, neurons, components, layers, epoch]
            for err in median_errors:
                median_error_row.append(err)

            # write each row to file
            with open('model tests/current test/min_error.csv', mode='a') as file:
                employee_writer = csv.writer(file, delimiter=',')
                employee_writer.writerow(min_error_row)
            with open('model tests/current test/mean_error.csv', mode='a') as file:
                employee_writer = csv.writer(file, delimiter=',')
                employee_writer.writerow(mean_error_row)
            with open('model tests/current test/median_error.csv', mode='a') as file:
                employee_writer = csv.writer(file, delimiter=',')
                employee_writer.writerow(median_error_row)


