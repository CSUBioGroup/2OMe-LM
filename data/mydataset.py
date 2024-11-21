from collections import Counter
from Bio import SeqIO
from torch.utils import data
import torch as t
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import os
import pickle
from utils.config import *
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, \
    AutoModelForSequenceClassification

from tqdm import tqdm

params = config()

SPLICEBERT_PATH = "models_folder/SpliceBERT.1024nt"

class SpliceBERT(data.Dataset):
    def __init__(self, fasta_path, device = params.device):
        self.device = device
        print('Loading the raw data...')

        records = list(SeqIO.parse(fasta_path, 'fasta'))

        fasta_seqs = [str(record.seq) for record in records]
        fasta_seq_names = [record.id for record in records]
        self.tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)
        model = AutoModel.from_pretrained(SPLICEBERT_PATH)
        model.to(device) 
        model.eval()  
        print(len(fasta_seqs), len(fasta_seq_names))

        self.id2lab = ['Positive', 'Negative']

        self.labels = [1 if name.startswith('P') else 0 if name.startswith('N') else None for name in fasta_seq_names]

        self.seqs = [(seq_name, seq) for seq_name, seq in zip(fasta_seq_names, fasta_seqs)]

        self.k_RNA = [[i[j:j + 4] for j in range(len(i) - 4 + 1)] for i in fasta_seqs]

        num_class = len(set(self.labels))
        self.all_hidden_states = []

        for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Processing sequences"):
            input_ids = self.process_sequence(str(record.seq))
            with torch.no_grad():
                output = model(input_ids)
            last_hidden_state = output.last_hidden_state
            self.all_hidden_states.append(last_hidden_state.cpu())

        self.all_hidden_states = torch.cat(self.all_hidden_states, dim=0)

        print(self.all_hidden_states.shape)

        self.vector_size = 128
        self.model_path = 'models_folder/w2v_model.model'
        if os.path.exists(self.model_path):
            model = Word2Vec.load(self.model_path)
        else:
            model = Word2Vec(sentences=self.k_RNA, vector_size=self.vector_size, window=5, min_count=1, workers=4)
            model.save(self.model_path)

        with torch.no_grad():
            self.features = t.tensor(np.array([self.sequence_to_matrix(seq, model) for seq in self.k_RNA]))
        print('Dataset initialization is completed!')

    def __getitem__(self, index):
        return (self.features[index], self.all_hidden_states[index]), self.labels[index]

    def __len__(self):
        return len(self.labels)

    def process_sequence(self,seq):
        seq = ' '.join(list(seq.upper().replace("U", "T")))
        input_ids = self.tokenizer.encode(seq, add_special_tokens=True)  
        return torch.tensor(input_ids).unsqueeze(0).to(self.device) 

    def get_all_features_labels(self):
        features = [(self.features[i], self.all_hidden_states[i]) for i in range(len(self.seqs))]
        labels = self.labels
        return features, labels

    def sequence_to_matrix(self, sequence, model):
        return np.array([model.wv[kmer] for kmer in sequence if kmer in model.wv])
