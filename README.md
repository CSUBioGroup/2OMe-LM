# 2OMe-LM

Predicting 2’-O-methylation Sites in Human RNA Using A Pretrained RNA Language Model.

The web server for prediction and visualization available at [http://csuligroup.com:8000/2OMe-LM](http://csuligroup.com:8000/2OMe-LM).

## Requirements
biopython==1.83

gensim==4.2.0

numpy==1.23.1

optuna==3.1.0

pandas==2.2.2

scikit-learn==1.1.1

scipy==1.10.1

tokenizers==0.13.3

torch==1.10.0+cu113

transformers==4.29.2

## Get Started
### Pre-trained model download
The pre-trained model used in 2OMe-LM is [**SpliceBERT.1024nt**](https://zenodo.org/record/7995778/files/models.tar.gz?download=1), it should be downloaded to `models_folder`.

### Training
In the `utils/config.py` file, you can adjust the hyperparameters. If you don’t modify the file, the default parameters will be used for training. To start the training, just run the following command:
```bash
python train.py
```
After training, you will find the trained model saved in the `checkpoints` folder. You can use this model for making predictions.

### Prediction
For prediction, all inputs must be in FASTA format. You can place the file in the working directory and run the following command to get the prediction results:
```bash
python predict.py yourfile.fasta model1_path model2_path model3_path model4_path model5_path
```

## Citation
Qianpei Liu #, Min Zeng #, Yiming Li, Chengqian Lu, Shichao Kan, Fei Guo, and Min Li*, "2OMe-LM: predicting 2’-O-methylation sites in human RNA using a pre-trained RNA language model".
