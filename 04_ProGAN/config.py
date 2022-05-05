import torch


START_TRAIN_AT_IMG_SIZE = 16
DATASET = '/home/tetae-gpu-server/disk_4000/data/celeba'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = False
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
# in my project batch size will be determined by MULTIPLIED_IMGSIZE_BATCHSIZE_CHANNELS
# BATCH_SIZES = [32, 32, 32, 16, 16, 16, 12, 4, 2]
CHANNELS_IMG = 3
Z_DIM = 256*2  # should be 512 in original paper
IN_CHANNELS = 256*2  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 8
MULTIPLIED_IMGSIZE_BATCHSIZE_CHANNELS = 256*128*16 # channels * img size * batch size