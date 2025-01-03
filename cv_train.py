from data.mydataset import *
from models.LM_2OME import *
from utils.WarmupLR import *
from utils.config import *
from utils.binary_metrics import *
import torch.nn.functional as F
import torch
import numpy as np
import random
import os
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import SubsetRandomSampler, DataLoader

params = config()

arr = [9255, 6695, 5305, 8897, 8419]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class cv_train:
    def __init__(self):
        pass

    def train(self, dataset, batchSize=params.batchSize, num_epochs=params.numEpochs, lr=params.lr,
              kFold=params.kFold, savePath=params.savePath, earlyStop=params.earlyStop, device=params.device,
              seed=params.seed):
        self.device = device
        splits = StratifiedKFold(n_splits=kFold, shuffle=True, random_state=10)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        valid_res = []
        features, labels = dataset.get_all_features_labels()
        for fold, (train_idx, valid_idx) in enumerate(splits.split(features, labels)):
            setup_seed(arr[fold])
            self.model = LM_2OME()
            savePath2 = savePath + "cv" + str(fold + 1)
            best_auc = 0.0
            print(f'Fold{fold + 1}:')
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            self.train_loader = DataLoader(dataset, batch_size=batchSize, sampler=train_sampler, num_workers=4)
            self.valid_loader = DataLoader(dataset, batch_size=batchSize, sampler=valid_sampler, num_workers=4)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            lr_scheduler = WarmupLR(self.optimizer, warmup_steps=150)
            self.lossfn1 = nn.BCELoss()
            self.metrictor = Metrictor()

            self.report = ["ACC", "F1", "Pre", "Rec", "AUC", "AUPR", "MCC"]

            nobetter = 0
            for epoch in range(num_epochs):
                print(f'========== Epoch:{epoch + 1:5d} ==========')
                self.train_epoch(epoch=epoch)
                lr_scheduler.step()

                print("[Train]", end='')
                self.test(train_mode=True)
                print("[Valid]", end='')
                res = self.test(train_mode=False)

                if best_auc < res["AUC"]:
                    nobetter = 0
                    best_auc = res["AUC"]
                    print(f'>Bingo!!! Get a better Model with valid AUC: {best_auc:.3f}!!!')
                    torch.save(self.model.state_dict(), f"{savePath2}.pkl")
                else:
                    nobetter += 1
                    if nobetter >= earlyStop:
                        print(
                            f'Valid AUC has not improved for more than {earlyStop} steps in epoch {epoch + 1}, stop training.')
                        break
                print('=================================')

            self.model.load_state_dict(torch.load(f"{savePath2}.pkl"))

            os.rename("%s.pkl" % savePath2, "%s_%s.pkl" % (savePath2, ("%.3f" % best_auc)))

            print(f'============ Result ============')
            print("[Train]", end='')
            self.test(train_mode=True)
            print("[Valid]", end='')
            res = self.test(train_mode=False)

            valid_res.append(res)
            self.metrictor.each_class_indictor_show(dataset.id2lab)
            print(f'================================')

        Metrictor.table_show(valid_res, self.report)

    def train_epoch(self, epoch):
        self.model.train()

        for i, (text, labels) in enumerate(self.train_loader):
            labels = labels.to(self.device).to(torch.float)
            self.optimizer.zero_grad()
            y, _,_ = self.model(text)
            loss1 = self.lossfn1(y, labels)
            loss_sum = loss1
            loss_sum.backward()
            self.optimizer.step()
        return

    def test(self, train_mode=False):
        self.model.eval()
        loader = self.train_loader if train_mode else self.valid_loader
        with torch.no_grad():
            y_list, label_list = [], []
            for (text, labels) in loader:
                labels = labels.to(self.device).to(torch.float)
                y, _,_ = self.model(text)
                y_list.append(y)
                label_list.append(labels)

            x1_tensor = torch.cat(y_list, 0)
            label_tensor = torch.cat(label_list, 0)

            loss1 = self.lossfn1(x1_tensor, label_tensor)
            loss_sum = loss1

            print(f"Loss= {loss_sum:.3f};", end='')

            label_tensor = label_tensor.to(torch.int)
            x1_numpy, label_numpy = x1_tensor.cpu().numpy(), label_tensor.cpu().numpy()
            self.metrictor.set_data(x1_numpy, label_numpy)
            res = self.metrictor(self.report)
            return res
