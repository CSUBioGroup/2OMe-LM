import torch

# Set global variables and modify them here
class config:
    def __init__(self):
        self.d = 128
        # Number of the hidden neurons
        self.hidden_dim = 256
        # Seed for model initialization
        self.seed = 10
        # Number of sample categories
        self.n_classes = 2
        # Training parameters
        self.batchSize = 32
        self.numEpochs = 1000
        self.lr = 0.0005
        self.earlyStop = 100
        self.kFold = 5
        # The weight factor of the supervised contrastive learning loss function
        self.alpha = 0.1
        # Models saved here
        self.savePath = f"checkpoints/"
        self.device = torch.device("cuda:0")
