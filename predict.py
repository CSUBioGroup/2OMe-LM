import argparse
from sklearn import metrics as skmetrics
from models.LM_2OME import *
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import os
import pickle
from utils.config import *
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, \
    AutoModelForSequenceClassification

import logging
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

params = config()

def sequence_to_matrix(sequence, model):
    return torch.tensor(np.array([model.wv[kmer] for kmer in sequence if kmer in model.wv]))

SPLICEBERT_PATH = "models_folder/SpliceBERT.1024nt"


def data_process(seq):
    k_RNA = [seq[j:j + 4] for j in range(len(seq) - 4 + 1)]
    sp_fea = []
    tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)
    sp_model = AutoModel.from_pretrained(SPLICEBERT_PATH)
    sp_model.eval() 
    seq = ' '.join(list(seq.upper().replace("U", "T")))
    input_ids = tokenizer.encode(seq, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():  
        output = sp_model(input_ids)
    last_hidden_state = output.last_hidden_state
    sp_fea.append(last_hidden_state.cpu())
    sp_fea = torch.cat(sp_fea, dim=0)
    w2v_model = Word2Vec.load('models_folder/w2v_model.model')
    with torch.no_grad():
        w2v_fea = sequence_to_matrix(k_RNA, w2v_model).unsqueeze(0)
    return (w2v_fea, sp_fea)

def predict(seq, model_list):
    input = data_process(seq)
    res = 0.0
    with torch.no_grad():
        for _, model in enumerate(model_list):
            y, _, _ = model(input)
            res += y.detach()
    res /= 5
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using trained model on a FASTA file.")
    parser.add_argument('fasta_path', type=str, help="Path to the FASTA file")
    parser.add_argument('model1_path', type=str, help="Path to the cv1 model .pkl file")
    parser.add_argument('model2_path', type=str, help="Path to the cv2 model .pkl file")
    parser.add_argument('model3_path', type=str, help="Path to the cv3 model .pkl file")
    parser.add_argument('model4_path', type=str, help="Path to the cv4 model .pkl file")
    parser.add_argument('model5_path', type=str, help="Path to the cv5 model .pkl file")

    args = parser.parse_args()

    fasta_path = args.fasta_path

    model1 = LM_2OME()
    model1.load_state_dict(torch.load(args.model1_path))
    model1.eval()

    model2 = LM_2OME()
    model2.load_state_dict(torch.load(args.model2_path))
    model2.eval()

    model3 = LM_2OME()
    model3.load_state_dict(torch.load(args.model3_path))
    model3.eval()

    model4 = LM_2OME()
    model4.load_state_dict(torch.load(args.model4_path))
    model4.eval()

    model5 = LM_2OME()
    model5.load_state_dict(torch.load(args.model5_path))
    model5.eval()

    model_list = [model1, model2, model3, model4, model5]

    records = list(SeqIO.parse(fasta_path, 'fasta'))

    fasta_seqs = [str(record.seq) for record in records]
    fasta_seq_names = [record.id for record in records]
    
    for i, seq in enumerate(fasta_seqs):
        res = predict(seq, model_list)
        print(i+1, ":", res.item())
