from create_model_data import *
from train_LSTM_MDN import *
from validation_analysis import *

# save training data for grid search
# save_train_data('trajectory data/', 'grid_search_train', points=False)

# save training data for model to learn to approach from above, and not avoid obstacles too much.
# save_train_data('trajectory data/aggressive above/', 'aggressive_above_train', points=False)

# save training data for model to learn to approach from below, and carefully avoid obstacles
# save_train_data('trajectory data/conservative below/', 'conservative_below_train', points=False)

# train and test aggressive above model
# train_test_model('aggressive_above_train', 'test trajectories/', validation=False)
# train and test conservative below model
# train_test_model('conservative_below_train', 'test trajectories/', validation=False)
# run validation of optimal grid search model
# train_test_model('grid_search_train', 'validation data/RRT comparison/', validation=True)
train_test_model('grid_search_train', 'test trajectories/', validation=False)

# run the grid search
# run_grid_search('grid_search_train', 'grid_search')

# analyse RRT trajectories
# compare_lfd_rrt()

# compare the trajectories for the different task test
# compare_task_trajectories('conservative_below_train', 'test trajectories/')