# 2OMe-LM

Predicting 2’-O-methylation Sites in Human RNA Using A Pre-trained RNA Language Model.

The web server for prediction and visualization available at [https://csuligroup.com:9200/2OMe-LM](https://csuligroup.com:9200/2OMe-LM).



## Setup

### Requirements

biopython==1.83

gensim==4.2.0

numpy==1.23.1

optuna==3.1.0

pandas==2.2.2

scikit-learn==1.1.1

scipy==1.10.1

tokenizers==0.13.3

torch==1.10.0

transformers==4.29.2

### Create an environment

We highly recommend using a virtual environment for the installation of 2OMe-LM and its dependencies. A virtual environment can be created and (de)activated as follows by using [conda](https://conda.io/docs/):

```sh
# create
conda create -n 2OMe-LM_Env python=3.9
# activate
conda activate 2OMe-LM_Env
```

### Install 2OMe-LM

After creating and activating the environment, download and install 2OMe-LM (**latest version**) from github:

```sh
git clone https://github.com/CSUBioGroup/2OMe-LM.git
cd 2OMe-LM
pip install -r requirements.txt
```



## Get Started
### Pre-trained model download
The pre-trained model used in 2OMe-LM is [**SpliceBERT.1024nt**](https://zenodo.org/record/7995778/files/models.tar.gz?download=1), it should be downloaded to `models_folder`.

### Training
In the `utils/config.py` file, you can adjust the hyperparameters. 

>In the `utils/config.py`, the meaning of the variables is explained as follows:

>> ***seed*** is the seed for model initialization.
>> ***batchSize*** is the batchsize of training.
>> ***numEpochs*** is the largest number of training epochs.
>> ***earlyStop*** is the parameter corresponding to the early stop method.
>> ***lr*** is the learning rate of training.
>> ***savePath*** is the folder where the model is saved.
>> ***device*** is the device you used to build and train the model. It can be "cpu" for cpu or "cuda" for gpu, and "cuda:0" for gpu 0.

If you don’t modify the file, the default parameters will be used for training. To start the training, just run the following command:

```bash
python train.py
```
After training, you will find the trained model saved in the `checkpoints` folder. You can use this model for making predictions.

### Prediction
For prediction,if your file to be tested is in FASTA format. You can place the file in the working directory and run the following command to get the prediction results:
```bash
python predict.py yourfile.fasta model1_path model2_path model3_path model4_path model5_path
```

In addition, you can test your sequence as follows:

```python
#First, import the package.
from predict import *

model1 = LM_2OME()
model1.load_state_dict(torch.load(path/to/model1_path))
model1.eval()

model2 = LM_2OME()
model2.load_state_dict(torch.load(path/to/model2_path))
model2.eval()

model3 = LM_2OME()
model3.load_state_dict(torch.load(path/to/model3_path))
model3.eval()

model4 = LM_2OME()
model4.load_state_dict(torch.load(path/to/model4_path))
model4.eval()

model5 = LM_2OME()
model5.load_state_dict(torch.load(path/to/model5_path))
model5.eval()

model_list = [model1, model2, model3, model4, model5]

#Then assume an RNA sequence.
seq = "CCAUCUACUAUGAGACCAGGAAGAUCGUGGGUGCGGAGAUC"

result = predict(seq,model_list)
print(result)
```



## Citation

Qianpei Liu #, Min Zeng #, Yiming Li, Chengqian Lu, Shichao Kan, Fei Guo, and Min Li*, "2OMe-LM: predicting 2’-O-methylation sites in human RNA using a pre-trained RNA language model".
