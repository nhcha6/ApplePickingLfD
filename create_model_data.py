import pandas as pd
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join

def calculate_weighted_centre(current_pos, point_cloud, goal_pos, goal_flag=False):
    # extract the repulsive vector from the point cloud data
    inverse_distance_sum = 0
    weighted_average_point = [0, 0, 0]
    max_repulsion_dist = 0.5
    closest_point = None
    closest_distance = 100
    # iterate through each point
    for point in point_cloud:
        # either consider points close to goal or ee
        if goal_flag:
            goal_to_point = np.subtract(goal_pos, point)
            dist = np.linalg.norm(goal_to_point)
        else:
            ee_to_point = np.subtract(current_pos, point)
            dist = np.linalg.norm(ee_to_point)
        # if point is close to goal/ee, point contributes to weighted centre
        if (dist < max_repulsion_dist):
            inverse_distance_sum += (1 / dist)
            for i in range(len(point)):
                weighted_average_point[i] += point[i] * (1 / dist)
            if dist < closest_distance:
                closest_point = point
                closest_distance = dist
    try:
        weighted_average_point = [x / inverse_distance_sum for x in weighted_average_point]
    except ZeroDivisionError:
        weighted_average_point = [0, 0, 0]
        closest_point = [0, 0, 0]

    return weighted_average_point, closest_point

def norm_vector(vector):
    mag = np.linalg.norm(vector)
    mag_norm = 1 / (1 + mag)
    norm_vector = (vector * mag_norm) / mag
    return norm_vector

def generate_test_data(file_name, take_every=3):
    # read in trajectory df and intialise x and y
    traj_df = pd.read_pickle('trajectory data/' + file_name + '.pkl')
    goal_df = pd.read_pickle('trajectory data/' + file_name + '_goal.pkl')
    point_cloud_df = pd.read_pickle('trajectory data/' + file_name + '_point_cloud.pkl')

    point_cloud = []
    i = 10
    for index, row in point_cloud_df.iterrows():
        point_cloud.append([row[0], row[1], row[2]])

    goal = [goal_df.iloc[0, 0], goal_df.iloc[1, 0], goal_df.iloc[2, 0]]

    x = []
    y = []

    # can generate many different trajectories by using lower sample rate
    for start in range(take_every):
        x_series = []
        y_series = []

        # generate the series for this particular subset of the trajectory
        for i in range(start, traj_df.shape[0]-take_every, take_every):
            # extract the motion generated at this timestep
            ee_pos = [-traj_df.iloc[i]['traj'][3], -traj_df.iloc[i]['traj'][7], traj_df.iloc[i]['traj'][11]]
            ee_pos_next = [-traj_df.iloc[i + take_every]['traj'][3], -traj_df.iloc[i + take_every]['traj'][7], traj_df.iloc[i + take_every]['traj'][11]]
            motion = np.subtract(ee_pos_next, ee_pos)

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

            print('\n')
            print("attractive")
            print(attractive)
            print(attractive_norm)
            print('repulsive_ee')
            print(repulsive_ee)
            print(repulsive_ee_norm)
            print('repulsive_goal')
            print(repulsive_goal)
            print(repulsive_goal_norm)
            print('repulsive_closest')
            print(repulsive_closest)
            print(repulsive_closest_norm)
            print('motion')
            print(motion)

            return

            # alternative, can use the points instead of the normalised attractive and repulsive vectors
            args = (attractive_norm, repulsive_ee_norm, repulsive_goal_norm, repulsive_closest_norm)
            x_series.append(np.concatenate(args))
            y_series.append(ee_pos)

        # convert the x_series to the just the first sample
        x_series = np.array(x_series)
        y_series = np.array(y_series)
        start_sample = np.zeros((20, x_series.shape[1]))
        start_sample[-1] = x_series[0]

        # print('\n')
        # print(start_sample)
        # print(y_series)

        x.append(start_sample)
        y.append(y_series)

    return x, y


def generate_train_data(folder, file, take_every=3, points = False):
    print(file)
    # read in trajectory df and intialise x and y
    traj_df = pd.read_pickle(folder + file + '.pkl')
    goal_df = pd.read_pickle(folder + file + '_goal.pkl')
    point_cloud_df = pd.read_pickle(folder + file + '_point_cloud.pkl')
    # update point cloud data
    point_cloud = []
    for index, row in point_cloud_df.iterrows():
        point_cloud.append([row[0], row[1], row[2]])
    # update goal
    goal = [goal_df.iloc[0, 0], goal_df.iloc[1, 0], goal_df.iloc[2, 0]]

    x = []
    y = []
    # can generate many different trajectories by using lower sample rate
    for start in range(take_every):
        x_series = []
        y_series = []

        # generate the series for this particular subset of the trajectory
        for i in range(start, traj_df.shape[0] - take_every, take_every):
            # extract the motion generated at this timestep
            ee_pos = [-traj_df.iloc[i]['traj'][3], -traj_df.iloc[i]['traj'][7], traj_df.iloc[i]['traj'][11]]
            ee_pos_next = [-traj_df.iloc[i + take_every]['traj'][3], -traj_df.iloc[i + take_every]['traj'][7],
                           traj_df.iloc[i + take_every]['traj'][11]]
            motion = np.subtract(ee_pos_next, ee_pos)

            [repulsive_ee, repulsive_closest] = calculate_weighted_centre(ee_pos, point_cloud, goal, goal_flag=False)
            [repulsive_goal, closest_point] = calculate_weighted_centre(ee_pos, point_cloud, goal, goal_flag=True)

            # calculate vectors between goal/repulsive points
            attractive = np.subtract(goal, ee_pos)
            repulsive_ee = np.subtract(ee_pos, repulsive_ee)
            repulsive_goal = np.subtract(ee_pos, repulsive_goal)
            repulsive_closest = np.subtract(ee_pos, repulsive_closest)

            if not points:
                # normalise each one
                attractive_norm = norm_vector(attractive)
                repulsive_ee_norm = norm_vector(repulsive_ee)
                repulsive_goal_norm = norm_vector(repulsive_goal)
                repulsive_closest_norm = norm_vector(repulsive_closest)
            else:
                print('points')
                attractive_norm = attractive
                repulsive_ee_norm = repulsive_ee
                repulsive_goal_norm = repulsive_goal
                repulsive_closest_norm = repulsive_closest

            # print('\n')
            # print("attractive")
            # print(attractive)
            # print(attractive_norm)
            # print('repulsive_ee')
            # print(repulsive_ee)
            # print(repulsive_ee_norm)
            # print('repulsive_goal')
            # print(repulsive_goal)
            # print(repulsive_goal_norm)
            # print('repulsive_closest')
            # print(repulsive_closest)
            # print(repulsive_closest_norm)
            # print('motion')
            # print(motion)

            # alternative, can use the points instead of the normalised attractive and repulsive vectors
            args = (attractive_norm, repulsive_ee_norm, repulsive_goal_norm, repulsive_closest_norm)
            x_series.append(np.concatenate(args))
            y_series.append(motion)

        # convert the x_series to an array for conversion to input data
        x_series = np.array(x_series)
        print(x_series)
        print(y_series)

        # convert the series into input and output data
        # variable starting point: start from any of the first 5 samples
        # pad the input to 20 samples
        for k in range(5):
            x_subset = []
            y_subset = []
            # x_series = np.array([[x] for x in range(k, 30)])
            x_sub_series = x_series[k:]
            for i in range(len(x_sub_series)):
                x_sample = np.zeros((20, x_sub_series.shape[1]))
                for j in range(0, min(i+1,20)):
                    x_sample[19-j] = x_sub_series[i-j]
                # print(x_sample)
                x_subset.append(x_sample)
                y_subset.append(y_series[i])

            # once input is processed for a particular starting point, add to the x and y data
            x.extend(x_subset)
            y.extend(y_subset)

    return x, y

def save_train_data(recorded_data, save_name, points = False):
    x = []
    y = []
    onlyfiles = [f for f in listdir(recorded_data) if isfile(join(recorded_data, f))]

    i=0
    for file_name in onlyfiles:
        # get only trajectory file name
        if 'goal' in file_name:
            continue
        if 'point_cloud' in file_name:
            continue

        print(i)
        i+=1
        x_traj, y_traj = generate_train_data(recorded_data, file_name[0:-4], points=points)
        x.extend(x_traj)
        y.extend(y_traj)

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)

    with open('model data/x_' + save_name + '.pkl', 'wb') as f:
        pickle.dump(x, f)
    with open('model data/y_' + save_name + '.pkl', 'wb') as f:
        pickle.dump(y, f)

def save_test_data(recorded_data, file_name):
    x = []
    y = []
    for file_name in recorded_data:
        path = 'trajectory data/' + file_name + '.pkl'
        x_traj, y_traj = generate_test_data(file_name)
        x.extend(x_traj)
        y.extend(y_traj)

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)

    with open('model data/x_' + file_name + '.pkl', 'wb') as f:
        pickle.dump(x, f)
    with open('model data/y_' + file_name + '.pkl', 'wb') as f:
        pickle.dump(y, f)




