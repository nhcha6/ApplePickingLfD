from train_LSTM_MDN import *

def analyse_RRT_trajectories(folder_name, plot=False):
    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    mean_execution_times = {}
    mean_planning_times = {}
    traj_distance = {}
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

        # original planning and execution times
        # mean_execution_times[file_name] = np.mean(RRT_times.iloc[0].values)
        mean_planning_times[file_name] = np.mean(RRT_times.iloc[1].values)

        # new execution times
        execution_time = pd.read_pickle(folder_name + 'results/' + file_name + '_times.pkl')
        mean_execution_times[file_name] = np.mean(execution_time[1].values)

        distances = []
        for i in range(10):
            traj_df = pd.read_pickle(
                folder_name + 'results/' + file_name + '_js_difference_' + str(i) + '.pkl')
            total_dist = 0
            for j in range(0, traj_df.shape[0]):
                delta_js = traj_df.iloc[j].values
                # dist = sum([abs(x) for x in delta_js])
                dist = np.linalg.norm(delta_js)
                total_dist += dist
            distances.append(total_dist)
        traj_distance[file_name] = np.mean(distances)

        if plot:
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

    mean_planning_total = np.mean([x for x in mean_planning_times.values()])
    mean_execution_total = np.mean([x for x in mean_execution_times.values()])
    print(traj_distance)
    print(np.mean([x for x in traj_distance.values()]))

    plot_cycle_time(mean_execution_times, mean_planning_times, 'RRTConnect')

    return mean_planning_total, mean_execution_total, traj_distance

def analyse_lfd_times():
    folder_name = 'validation results/'
    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    mean_execution_times = {}
    mean_planning_times = {}
    traj_distance = {}
    for file_name in onlyfiles:
        # get only trajectory file name
        if 'goal' in file_name:
            continue
        if 'traj' in file_name:
            continue
        file_name = file_name[0:8]

        planning_time = pd.read_csv(folder_name + file_name + '_planning_times.csv')

        mean_plan_time = np.mean(planning_time['0'].values)

        execution_time = pd.read_pickle(folder_name + 'final validation/' + file_name + '_times.pkl')
        mean_js_conversion = np.mean(execution_time[0].values)
        mean_execution_time = np.mean(execution_time[1].values)

        mean_planning_times[file_name] = mean_plan_time + mean_js_conversion
        mean_execution_times[file_name] = mean_execution_time

        distances = []
        for i in range(10):
            traj_df = pd.read_pickle(folder_name + 'final validation/' + file_name + '_js_difference_' + str(i) + '.pkl')
            total_dist = 0
            for j in range(0, traj_df.shape[0]):
                delta_js = traj_df.iloc[j].values
                # dist = sum([abs(x) for x in delta_js])
                dist = np.linalg.norm(delta_js)
                total_dist += dist
            distances.append(total_dist)
        traj_distance[file_name] = np.mean(distances)

    mean_planning_total = np.mean([x for x in mean_planning_times.values()])
    mean_execution_total = np.mean([x for x in mean_execution_times.values()])
    print(traj_distance)
    print(np.mean([x for x in traj_distance.values()]))

    plot_cycle_time(mean_execution_times, mean_planning_times, 'LfD Model')

    return mean_planning_total, mean_execution_total, traj_distance


def plot_cycle_time(execution_time, planning_time, header):
    N = len(execution_time)
    test_names = [x for x in execution_time.keys()]
    execution = [execution_time[name] for name in test_names]
    planning = [planning_time[name] for name in test_names]
    ind = np.arange(N)  # the x locations for the groups
    width = 0.6
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.bar(ind, planning, width, color='r')
    ax.bar(ind, execution, width, bottom=planning, color='b')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Picking Scenario')
    ax.set_title('Execution and Planning Time for ' + header)
    ax.set_xticks(ind)
    ax.set_yticks(np.arange(0, 17, 2))
    ax.legend(labels=['Planning', 'Execution'])

def compare_lfd_rrt():
    planning_rrt, exec_rrt, js_dist_rrt = analyse_RRT_trajectories('validation data/RRT comparison/')
    planning_lfd, exec_lfd, js_dist_rrt = analyse_lfd_times()

    # plot comparison of planning and execution time
    N = 2
    execution = [exec_lfd, exec_rrt]
    planning = [planning_lfd, planning_rrt]
    ind = np.arange(N)  # the x locations for the groups
    width = 0.8
    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax.bar(('LfD', 'RRTConnect'), planning, width, color='r')
    ax.bar(('LfD', 'RRTConnect'), execution, width, bottom=planning, color='b')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Model')
    ax.set_title('Mean Execution and Planning Time for Test Scenarios')
    ax.set_xticks(ind)
    ax.set_yticks(np.arange(0, 9, 1))
    ax.legend(labels=['Planning', 'Execution'])
    plt.show()

def visualise_task_comparison(approach_vector_dict, obstacle_distance_dict):
    approach_above = []
    distance_above = []
    approach_below = []
    distance_below = []
    for name in approach_vector_dict.keys():
        approach_below.append(approach_vector_dict[name][0])
        distance_below.append(obstacle_distance_dict[name][0])
        approach_above.append(approach_vector_dict[name][1])
        distance_above.append(obstacle_distance_dict[name][1])

    print('Mean Approach Angle: Aggressive Above')
    print(np.mean(approach_above))
    print('Mean Distance from Closest Collision: Aggressive Above')
    print(np.mean(distance_above))
    print('Mean Approach Angle: Conservative Below')
    print(np.mean(approach_below))
    print('Mean Distance from Closest Collision: Conservative Below')
    print(np.mean(distance_below))

