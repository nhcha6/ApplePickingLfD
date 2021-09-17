from train_LSTM_MDN import *

def analyse_RRT_trajectories(folder_name):
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
        RRT_traj = pd.read_pickle(folder_name + file_name + '_RRT_traj.pkl')
        RRT_times = pd.read_pickle(folder_name + file_name + '_RRT_times.pkl')

        print('execution time')
        print(RRT_times.iloc[0])
        print('plan time')
        print(RRT_times.iloc[1])

        # convert to usable form
        point_cloud = []
        for index, row in point_cloud_df.iterrows():
            if index %10:
                continue
            point_cloud.append([row[0], row[1], row[2]])
        goal = [goal_df.iloc[0, 0], goal_df.iloc[1, 0], goal_df.iloc[2, 0]]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot([p[0] for p in point_cloud], [p[1] for p in point_cloud], [p[2] for p in point_cloud], 'bo', markersize=2,alpha=0.2)
        for i in range(10):
            traj = [x for x in RRT_traj.iloc[i] if x]
            ax.plot([-p[3] for p in traj], [-p[7] for p in traj], [p[11] for p in traj])
        plt.show()