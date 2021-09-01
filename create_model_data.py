import pandas as pd
import numpy as np


def norm_vector(vector):
    mag = np.linalg.norm(vector)
    mag_norm = 1 / (1 + mag)
    norm_vector = (vector * mag_norm) / mag
    return norm_vector

def generate_data_series(file, take_every=3):
    traj_df = pd.read_pickle(file)
    x = []
    y = []

    # can generate many different trajectories by using lower sample rate
    for start in range(take_every):
        x_series = []
        y_series = []

        # generate the series
        for i in range(start, traj_df.shape[0] - 3, take_every):
            # extract the motion generated at this timestep
            ee_pos = [-traj_df.iloc[i]['traj'][3], -traj_df.iloc[i]['traj'][7], traj_df.iloc[i]['traj'][11]]
            ee_pos_next = [-traj_df.iloc[i + take_every]['traj'][3], -traj_df.iloc[i + take_every]['traj'][7],
                           traj_df.iloc[i + take_every]['traj'][11]]
            motion = np.subtract(ee_pos_next, ee_pos)

            # calculate vectors between goal/repulsive points
            attractive = np.subtract(traj_df.iloc[i]['attractive'], ee_pos)
            repulsive_ee = np.subtract(ee_pos, traj_df.iloc[i]['repulsive_ee'])
            repulsive_goal = np.subtract(ee_pos, traj_df.iloc[i]['repulsive_goal'])
            repulsive_closest = np.subtract(ee_pos, traj_df.iloc[i]['repulsive_closest'])

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

            args = (attractive_norm, repulsive_ee_norm, repulsive_goal_norm, repulsive_closest_norm)
            x_series.append(np.concatenate(args))
            y_series.append(motion)

        print(x_series)
        print(y_series)

        # convert the series into input and output data
        # variable starting point: start from any of the first 5 samples
        # pad the input to 20 samples
        for j in range(5):
            for i in range(j+1, len(y_series)+1):






    return x_series, y_series


recorded_data = ['trajectory data/test_1.pkl']

for path in recorded_data:
    x,y = generate_data_series(path)

