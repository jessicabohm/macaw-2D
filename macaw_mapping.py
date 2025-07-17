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


class macaw_mapping:

    def __init__(self, encoded_dim, lr=0.005, weight_decay=1e-6):
        self.encoded_dim = encoded_dim

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        self.macaw = self._init_macaw()

        self.feature_mean = None
        self.feature_std = None

        self.optimizer = torch.optim.Adam(self.macaw.parameters(), lr=lr, weight_decay=weight_decay)


    def _init_macaw(self):
    
        n_layers = 1
        hidden = [4, 6, 4]

        self.num_nodes = 5
        self.num_species = 2

        self.gm_node = 2
        self.wm_node = 3
        self.csf_node = 4

        # define connections between graph nodes
        #   1: species, 2: gm, 3: wm, 4: csf, 5-3605: latents
        species_to_latents = [(s, i) for s in range(self.num_species) for i in range(self.num_nodes, self.encoded_dim + self.num_nodes)]
        labels_to_gm = [(s, self.gm_node) for s in range(self.num_species)]
        labels_to_wm = [(s, self.wm_node) for s in range(self.num_species)]
        labels_to_csf = [(s, self.csf_node) for s in range(self.num_species)]
        gm_to_wm = [(self.gm_node, self.wm_node)]
        gm_to_latents = [(self.gm_node, i) for i in range(self.num_nodes, self.encoded_dim + self.num_nodes)]
        wm_to_latents = [(self.wm_node, i) for i in range(self.num_nodes, self.encoded_dim + self.num_nodes)]
        csf_to_latents = [(self.csf_node, i) for i in range(self.num_nodes, self.encoded_dim + self.num_nodes)]
        autoregressive_latents = [(i, j) for i in range(self.num_nodes, self.encoded_dim + self.num_nodes) for j in
                                  range(i + 1, self.encoded_dim + self.num_nodes)]

        edges = species_to_latents + labels_to_gm + labels_to_wm + labels_to_csf + gm_to_wm + gm_to_latents + wm_to_latents + csf_to_latents + autoregressive_latents

        priors = [(slice(0, self.num_species), td.OneHotCategorical(0.1 * torch.ones(self.num_species).to(self.device))), # NOTE: what is the 0.1?
                  (slice(self.num_species, self.encoded_dim + self.num_nodes),
                   td.Normal(torch.zeros(self.encoded_dim + self.num_nodes - self.num_species).to(self.device),
                             torch.ones(self.encoded_dim + self.num_nodes - self.num_species).to(self.device))),
                  ]

        flow_list = [Flow(self.encoded_dim + self.num_nodes, edges, self.device, hm=hidden) for _ in range(n_layers)]
        return NormalizingFlowModel(priors, flow_list).to(self.device)

    def _batch_to_x(self, batch):
        flow_batch, features_batch, species_batch = batch[0], batch[1], batch[2]
        species_batch = one_hot(species_batch).to(self.device) # one hot the species
        features_batch = (features_batch.to(self.device) - self.feature_mean) / self.feature_std
        x = torch.hstack([species_batch, features_batch, flow_batch]).type(torch.float32)
        return x, flow_batch

    def _x_to_batch(self, x, latents):
        latents[:, :] = x[:, self.num_nodes:]
        cfs = latents.detach().cpu().numpy() # also just don't decode here
        labels = inverse_one_hot(x[:, :self.num_species]).detach().cpu().numpy()
        features = (x[:, self.num_species:self.num_nodes] * self.feature_std + self.feature_mean).detach().cpu().numpy()
        return [cfs, features, labels]

    def _compute_feature_mean(self, train_loader):
        features = []
        for b in train_loader:
            features.append(b[1].numpy())
        features = np.vstack(features)
        scaler_f = StandardScaler()
        _ = scaler_f.fit_transform(features)
        self.feature_mean = torch.tensor(scaler_f.mean_).to(self.device)
        self.feature_std = torch.tensor(np.sqrt(scaler_f.var_)).to(self.device)

    def train_macaw(self, train_loader):

        self.macaw.train()

        if self.feature_mean is None:
            self._compute_feature_mean(train_loader)

        loss_val = []
        for batch in train_loader:
            batch = [batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)]
            with torch.no_grad():
                x, _ = self._batch_to_x(batch)

            _, prior_logprob, log_det = self.macaw(x)
            loss = - torch.sum(prior_logprob + log_det)
            loss /= train_loader.batch_size
            loss_val.append(loss.detach().cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return np.mean(loss_val)

    def test_likelihood(self, dataloader):

        self.macaw.eval()

        loss_val = []
        for batch in dataloader:
            with torch.no_grad():
                batch = [batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)]
                x, _ = self._batch_to_x(batch)

                _, prior_logprob, log_det = self.macaw(x)
                loss = - torch.sum(prior_logprob + log_det)
                loss /= dataloader.batch_size
                loss_val.append(loss.detach().cpu().numpy())

        return np.mean(loss_val)

    def _cf(self, x_obs, cf_vals):
        self.macaw.eval()

        with torch.no_grad():
            z_obs = self.macaw(x_obs)[0][-1]
            x_cf = x_obs.detach().clone()
            for key in cf_vals:
                x_cf[:, key] = cf_vals[key]

            z_cf_val = self.macaw(x_cf)[0][-1]
            for key in cf_vals:
                z_obs[:, key] = z_cf_val[:, key]

        return self.macaw.backward(z_obs)[0][-1]

    def counterfactual(self, batch, cf_vals):
        self.macaw.eval()

        cf_vals = self._cfvals_transform(cf_vals)

        with torch.no_grad():
            batch = [batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)]
            x_obs, latents = self._batch_to_x(batch)
            x_cf = self._cf(x_obs, cf_vals)
            return self._x_to_batch(x_cf, latents)

    def _cfvals_transform(self, cf_vals):
        cfv = {}

        if cf_vals is not None:
            for key in cf_vals:
                #print(key, cf_vals[key])
                if key == 'species':
                    cfv.update({i: int(x) for i, x in enumerate(one_hot(np.array([cf_vals[key]]))[0])})

                if key == 'gm':
                    tf = (torch.tensor([cf_vals[key], 0, 0]).to(self.device) - self.feature_mean) / self.feature_std
                    cfv.update({self.gm_node: tf[0]})
                if key == 'wm':
                    tf = (torch.tensor([0, cf_vals[key], 0]).to(self.device) - self.feature_mean) / self.feature_std
                    cfv.update({self.wm_node: tf[1]})
                if key == 'csf':
                    tf = (torch.tensor([0, 0, cf_vals[key]]).to(self.device) - self.feature_mean) / self.feature_std
                    cfv.update({self.csf_node: tf[2]})

            return cfv
    
    def decode(self, latent):
        decoder_save_folder = "../UNIT/save_2d/human_train_3/"
        decoder_epoch = 1000
        latent_shape = [1, 16, 15,15]

        decoder_trans = Decoder(n_upsample=3, n_res=2, dim=16, output_dim=4)
        decoder_trans.load_state_dict(torch.load(decoder_save_folder + "checkpoint_epoch_" + str(decoder_epoch))['decoder_state_dict'])
        trans_img = decoder_trans(torch.tensor(latent).reshape(latent_shape))
        trans_img = torch.argmax(trans_img, dim=1)
        trans_img = np.array(trans_img.squeeze())

        return trans_img

    def cf_test(self, model, batch, save_path="", nsamples=4):

        batch = [batch[0].to("cuda"), batch[1].to("cuda"), batch[2].to("cuda")]
        real_latents = batch[0].detach().cpu().numpy()

        cf_vals = {}
        cfs_none = model.counterfactual(batch, cf_vals)

        cf_vals = {'species': 1}
        cfs_human = model.counterfactual(batch, cf_vals)

        cf_vals = {'species': 1, 'gm': 0.59}
        cfs_human_large_gm = model.counterfactual(batch, cf_vals)

        # plot grid
        plt.rcParams['figure.figsize'] = (9, 7)
        fig = plt.figure()

        subfigs = fig.subfigures(nrows=4, ncols=1)

        # row 1 - the image
        nsamples = 4
        axs0 = subfigs[0].subplots(nrows=1, ncols=nsamples)
        for i, img in enumerate(real_latents):
            D_img = model.decode(img)
            axs0[i].imshow(D_img, cmap='gray')
            axs0[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)
            axs0[i].set_title(["Human test", "Mouse test", "Human train", "Mouse train"][i])
            axs0[i].set_xlabel("gm: " + str(np.round(np.sum(D_img == 1) / (np.sum((D_img == 1) | (D_img == 2) | (D_img == 3)) + 1e-10), 3)))

        # row 2 - 'counterfactual - but no values changed
        axs1 = subfigs[1].subplots(nrows=1, ncols=nsamples)
        for i, img in enumerate(cfs_none[0]):
            D_img = model.decode(img)
            axs1[i].imshow(D_img, cmap='gray')
            axs1[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)
            axs1[i].set_xlabel("gm: " + str(np.round(np.sum(D_img == 1) / (np.sum((D_img == 1) | (D_img == 2) | (D_img == 3)) + 1e-10), 3)))


        axs2 = subfigs[2].subplots(nrows=1, ncols=nsamples)
        for i, img in enumerate(cfs_human[0]):
            D_img = model.decode(img)
            axs2[i].imshow(D_img, cmap='gray')
            axs2[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)
            axs2[i].set_xlabel("gm: " + str(np.round(np.sum(D_img == 1) / (np.sum((D_img == 1) | (D_img == 2) | (D_img == 3)) + 1e-10), 3)))

        # row 4 - counterfactual - set gm to 0.59 (high end of human distribution)
        axs3 = subfigs[3].subplots(nrows=1, ncols=nsamples)
        for i, img in enumerate(cfs_human_large_gm[0]):
            D_img = model.decode(img)
            axs3[i].imshow(D_img, cmap='gray')
            axs3[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)
            axs3[i].set_xlabel("gm: " + str(np.round(np.sum(D_img == 1) / (np.sum((D_img == 1) | (D_img == 2) | (D_img == 3)) + 1e-10), 3)))

        axs0[0].set_ylabel("Decoded_H img")
        axs1[0].set_ylabel("CF no changes")
        axs2[0].set_ylabel("CF -> Human")
        axs3[0].set_ylabel("CF -> H + gm=0.59")
        
        if save_path != "":
            print("save to:", save_path)
            plt.savefig(save_path)
        else:
            plt.show()
            

    def get_optimizer(self):
        return self.optimizer
    
    def get_macaw(self):
        return self.macaw
    
    def load_state(self, state_path):
        self.macaw.load_state_dict(torch.load(state_path)['model_state_dict'])
        self.optimizer.load_state_dict(torch.load(state_path)['optimizer_state_dict'])
        


# default is 2 for num_species
def one_hot(a, veclen=2):
    b = torch.zeros((a.shape[0], veclen))
    b[torch.arange(a.shape[0]), a] = 1
    return b


def inverse_one_hot(b):
    return torch.argmax(b, axis=1)
