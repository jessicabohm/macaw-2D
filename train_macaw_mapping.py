from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random

#from macaw_mapping import *
import SimpleITK as sitk
import os
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.distributions as td
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from macaw.flows import Flow, NormalizingFlowModel

import sys
import os
sys.path.append('../UNIT')
from networks_update import Decoder
from macaw_mapping import *
import csv


# need return format to be: flow_batch, features_batch, species_batch = batch[0], batch[1], batch[2]

class MACAW_data_loader(Dataset):
    def __init__(self, mouse_encodings_folder, human_encodings_folder):
        self.mouse_encodings_folder = mouse_encodings_folder
        self.human_encodings_folder = human_encodings_folder

        self.mouse_encodings_files = os.listdir(mouse_encodings_folder)
        self.human_encodings_files = os.listdir(human_encodings_folder)

        self.paths = ([mouse_encodings_folder + file for file in self.mouse_encodings_files]) + ([human_encodings_folder + file for file in self.human_encodings_files])
        random.seed(0)
        random.shuffle(self.paths) # shuffle to mix up mice and human paths

        # compute wm/gm/csf ratios
        self.features = []
        self.species = []
        for path in self.paths:
            split_path = path.split("/")
            is_mouse = False
            is_test = False

            if split_path[3].split("_")[0] == "mouse":
                self.species.append(0)
                is_mouse = True
            else:
                self.species.append(1)

            is_test = (split_path[4] == "test_latent_encodings")

            img_path = "../UNIT/datasets/2d/" + ("mouse_" if is_mouse else "human_") + ("test/" if is_test else "train/") + split_path[-1][8:-3]

            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

            gm_count = np.sum(img_arr == 1)
            wm_count = np.sum(img_arr == 2)
            csf_count = np.sum(img_arr == 3)
            total_count = gm_count + wm_count + csf_count

            ratios = []
            ratios.append(gm_count/total_count)
            ratios.append(wm_count/total_count)
            ratios.append(csf_count/total_count)

            self.features.append(ratios)
        
        self.features = torch.tensor(self.features)
        self.species = torch.tensor(self.species)

    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        latent_encoding = torch.load(self.paths[idx]).detach().flatten()
        features = self.features[idx]
        species = self.species[idx]

        return [latent_encoding, features, species]
    
    def get_s_a_f(self):
        return self.species, self.features
    


# load data into random ordered vectors
mouse_encodings_folder = "../UNIT/save_2d/mouse_train_3/train_latent_encodings/"
human_encodings_folder = "../UNIT/save_2d/human_train_3/train_latent_encodings/"

mouse_encodings_folder_test = "../UNIT/save_2d/mouse_train_3/test_latent_encodings/"
human_encodings_folder_test = "../UNIT/save_2d/human_train_3/test_latent_encodings/"

batch_size = 4

dataset_train = MACAW_data_loader(mouse_encodings_folder, human_encodings_folder)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)

dataset_test = MACAW_data_loader(mouse_encodings_folder_test, human_encodings_folder_test)
val_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)


encoded_dim = 3600 # fits 3000!

model = macaw_mapping(encoded_dim, lr=0.0001)
#model.load_state('../UNIT/save_2d/mapping_models/macaw.pth')
losses = {'nll_train_loss':[],'nll_val_loss':[]}

from torch.utils.data._utils.collate import default_collate

mouse_test = dataset_test.__getitem__(1) 
human_test = dataset_test.__getitem__(3) 
mouse_train = dataset_train.__getitem__(1) 
human_train = dataset_train.__getitem__(2) 


test_imgs_batch = default_collate([mouse_test, human_test, mouse_train, human_train])


save_img_freq = 20
save_checkpoint_freq = 50
imgs_save_path = "./save_models/train_1/test_imgs/"
checkpoints_save_path = "./save_models/train_1/checkpoints/"
csv_path = "./save_models/train_1/loss_log.csv"

# Create CSV and write header if it doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "nll_train_loss", "nll_val_loss"])

for epoch in (pbar := tqdm(range(500))):
    nll_train_loss = model.train_macaw(train_loader)
    nll_val_loss = model.test_likelihood(val_loader)
    
    losses['nll_train_loss'].append(nll_train_loss)
    losses['nll_val_loss'].append(nll_val_loss)
        
    pbar.set_description(f"nll_train: {nll_train_loss:.3f}, nll_val: {losses['nll_val_loss'][-1]:.3f}")

    if epoch % save_img_freq == 0:
        print("saving imgs...")
        model.cf_test(model, test_imgs_batch, save_path=(imgs_save_path + "epoch_" + str(epoch)))
        print("done")

    
    if epoch % save_checkpoint_freq == 0:
        torch.save({
            'model_state_dict': model.get_macaw().state_dict(),
            'optimizer_state_dict': model.get_optimizer().state_dict()
        }, checkpoints_save_path + "epoch_" + str(epoch) + ".pth")


    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, nll_train_loss, nll_val_loss])
