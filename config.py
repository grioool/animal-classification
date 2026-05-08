import os

SEED = 42

DATA_TRAIN = os.path.join("data", "train")
DATA_TEST = os.path.join("data", "test")
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

IMG_SIZE = 128
VAL_SPLIT = 0.2
BATCH_SIZE = 32  # small batch works better with only 200 training images
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS_MAIN = 15
EPOCHS_GRID = 3
PATIENCE = 4
DROPOUT_HEAD = 0.5
USE_TRANSFER = False  # set True to swap in a frozen MobileNetV2 backbone

# ImageNet mean/std - used even for scratch training to keep pixel scale consistent
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
