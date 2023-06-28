import os
import datetime
import torch

# PATH_PICKLE = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
# PATH_IMAGES = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "wdd_ground_truth")

PATH_PICKLE = os.path.join(os.sep, "srv", "data", "joeh97", "data", "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
PATH_IMAGES = os.path.join(os.sep, "srv", "data", "joeh97", "data", "wdd_ground_truth")

PATH_CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")
PATH_CHECKPOINT_RESTORE = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_W = 224
IMG_H = 224
IMG_MEAN = (0.485 + 0.456 + 0.406) / 3
IMG_STD = (0.229 + 0.224 + 0.225) / 3

BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 128

LOG_INTERVAL = 5
SAVE_INTERVAL = 5

STATS_PATH = os.path.join(os.getcwd(), "stats_" + datetime.datetime.now().strftime("%Y%m%dT%H%M"))
SAVE_PATH_ACCURACY = os.path.join(STATS_PATH, "accuracy.pdf")
SAVE_PATH_LOSS = os.path.join(STATS_PATH, "loss.pdf")
SAVE_PATH_CONFUSION = os.path.join(STATS_PATH, "confusion.pdf")
SAVE_PATH_JSON = os.path.join(STATS_PATH, "stats.json")
SAVE_PATH_MODEL_SUMMARY = os.path.join(STATS_PATH, "model.txt")
