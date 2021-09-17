from create_model_data import *
from train_LSTM_MDN import *
from validation_analysis import *


# save_train_data('trajectory data/', 'grid_search_train', points=False)

# train and test model
train_test_model('grid_search_train', 'validation data/RRT comparison/')

# run_grid_search('grid_search_train', 'grid_search')

# analyse RRT trajectories
# analyse_RRT_trajectories('validation data/RRT comparison/')