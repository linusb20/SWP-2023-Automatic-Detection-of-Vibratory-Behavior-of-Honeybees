import os
import datetime
import torch

# path_pickle = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
# path_images = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "wdd_ground_truth")

path_pickle = os.path.join(os.sep, "srv", "data", "joeh97", "data", "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
path_images = os.path.join(os.sep, "srv", "data", "joeh97", "data", "wdd_ground_truth")

path_checkpoints = os.path.join(os.getcwd(), "checkpoints")
path_stats = os.path.join(os.getcwd(), "resources", "stats_" + datetime.datetime.now().strftime("%Y%m%dT%H%M"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1234

img_w = 64
img_h = 64
img_mean = 0.5
img_std = 0.5

batch_size = 16
num_workers = 4
num_epochs = 140

log_interval = 5
save_interval = 10
