from cv_train import *
from data.mydataset import *
from utils.config import *
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = SpliceBERT(fasta_path="data/H. sapines_train.fasta")
cv_models = cv_train()
cv_models.train(dataset)