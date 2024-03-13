import numpy as np
import torch
import torch.distributions as td
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from ae.decoder import DecoderUKBB
from ae.encoder import EncoderUKBB
from macaw.flows import Flow, NormalizingFlowModel


class automacaw_ukbb:

    def __init__(self, encoded_dim):
        self.encoded_dim = encoded_dim

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        self.encoder = EncoderUKBB(encoded_dim=encoded_dim).to(self.device)
        self.decoder = DecoderUKBB(encoded_dim=encoded_dim).to(self.device)
        self.macaw = self._init_macaw()

        self.feature_mean = None
        self.feature_std = None

    def _init_macaw(self):
        n_layers = 4
        hidden = [4, 6, 4]
        self.ncauses = 3
        self.min_age = 46

        P_sex = 0.46
        P_age = np.array([0.00079143, 0.00364058, 0.00707012, 0.01192423, 0.01698939,
                          0.02416504, 0.03033821, 0.0311824, 0.0311824, 0.03429536,
                          0.03345117, 0.03324012, 0.03503403, 0.03856909, 0.03936052,
                          0.04147101, 0.04294835, 0.0427373, 0.04147101, 0.04395083,
                          0.04505883, 0.0462196, 0.0434232, 0.04421464, 0.04347597,
                          0.03941329, 0.03498127, 0.02949401, 0.02279323, 0.02015512,
                          0.01551206, 0.00997204, 0.00896956, 0.00733393, 0.00416821,
                          0.00100248])

        sex_to_latents = [(0, i) for i in range(self.ncauses, self.encoded_dim + self.ncauses)]
        sex_to_bmi = [(0, 2)]

        age_to_latents = [(1, i) for i in range(self.ncauses, self.encoded_dim + self.ncauses)]
        age_to_bmi = [(1, 2)]

        bmi_to_latents = [(2, i) for i in range(self.ncauses, self.encoded_dim + self.ncauses)]
        autoregressive_latents = [(i, j) for i in range(self.ncauses, self.encoded_dim + self.ncauses) for j in
                                  range(i + 1, self.encoded_dim + self.ncauses)]

        edges = sex_to_latents + sex_to_bmi + age_to_latents + age_to_bmi + bmi_to_latents + autoregressive_latents

        priors = [(slice(0, 1), td.Bernoulli(torch.tensor([P_sex]).to(self.device))),  # sex
                  (slice(1, 2), td.Categorical(torch.tensor([P_age]).to(self.device))),  # age
                  (slice(2, 3), td.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))),  # BMI
                  (slice(self.ncauses, self.encoded_dim + self.ncauses),
                   td.Normal(torch.zeros(self.encoded_dim).to(self.device), torch.ones(
                       self.encoded_dim).to(self.device)))
                  ]

        flow_list = [Flow(self.encoded_dim + self.ncauses, edges, self.device, hm=hidden) for _ in range(n_layers)]
        return NormalizingFlowModel(priors, flow_list).to(self.device)

    def train_ae(self, train_loader, lr=0.0005, weight_decay=1e-06):
        self.encoder.train()
        self.decoder.train()

        params = [{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        loss_func = torch.nn.MSELoss()

        train_loss = []
        for batch in train_loader:
            image_batch = batch[3]
            image_batch = image_batch.to(self.device)
            encoded_data = self.encoder(image_batch)
            decoded_data = self.decoder(encoded_data)

            loss = loss_func(decoded_data, image_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def train(self, train_loader, lr=0.0005, alpha=0.1, beta=1000):
        self.encoder.train()
        self.macaw.train()
        self.decoder.train()

        if self.feature_mean is None:
            self._compute_feature_mean(train_loader)

        params = [{'params': self.encoder.parameters()}, {'params': self.macaw.parameters()},
                  {'params': self.decoder.parameters()}]

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-06)
        mse_loss_func = torch.nn.MSELoss()

        mse_loss_vals = []
        nll_loss_vals = []
        for batch in train_loader:
            with torch.no_grad():
                x = self._batch_to_x(batch)
                image_batch = batch[3]

            image_batch = image_batch.to(self.device)
            encoded_data = self.encoder(image_batch)
            decoded_data = self.decoder(encoded_data)

            _, prior_logprob, log_det = self.macaw(x)
            nll_loss = - torch.sum(prior_logprob + log_det)
            nll_loss /= train_loader.batch_size
            nll_loss_vals.append(nll_loss.detach().cpu().numpy())

            mse_loss = mse_loss_func(decoded_data, image_batch)
            mse_loss_vals.append(mse_loss.detach().cpu().numpy())

            loss = alpha * nll_loss + beta * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.mean(nll_loss_vals), np.mean(mse_loss_vals)

    def test(self, data_loader):
        self.encoder.eval()
        self.macaw.eval()
        self.decoder.eval()

        mse_loss_func = torch.nn.MSELoss()

        mse_loss_vals = []
        nll_loss_vals = []
        for batch in data_loader:
            with torch.no_grad():
                x = self._batch_to_x(batch)
                image_batch = batch[3].to(self.device)
                encoded_data = self.encoder(image_batch)
                decoded_data = self.decoder(encoded_data)

                _, prior_logprob, log_det = self.macaw(x)
                nll_loss = - torch.sum(prior_logprob + log_det)
                nll_loss /= data_loader.batch_size
                nll_loss_vals.append(nll_loss.detach().cpu().numpy())

                mse_loss = mse_loss_func(decoded_data, image_batch)
                mse_loss_vals.append(mse_loss.detach().cpu().numpy())

        return np.mean(nll_loss_vals), np.mean(mse_loss_vals)

    def test_MSE(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()

        loss_func = torch.nn.MSELoss()

        with torch.no_grad():
            conc_out = []
            conc_label = []
            for batch in dataloader:
                image_batch = batch[3]
                image_batch = image_batch.to(self.device)
                encoded_data = self.encoder(image_batch)
                decoded_data = self.decoder(encoded_data)

                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())

            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            val_loss = loss_func(conc_out, conc_label)
        return val_loss.data.numpy()

    def train_macaw(self, train_loader, lr=0.001):

        optimizer = torch.optim.Adam(self.macaw.parameters(), lr=lr, weight_decay=1e-6)

        self.macaw.train()
        self.encoder.eval()

        if self.feature_mean is None:
            self._compute_feature_mean(train_loader)

        loss_val = []
        for batch in train_loader:
            with torch.no_grad():
                x = self._batch_to_x(batch)

            _, prior_logprob, log_det = self.macaw(x)
            loss = - torch.sum(prior_logprob + log_det)
            loss /= train_loader.batch_size
            loss_val.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.mean(loss_val)

    def _batch_to_x(self, batch):
        sex_batch, age_batch, bmi_batch, image_batch = batch[0], batch[1], batch[2], batch[3]
        sex_batch = sex_batch.reshape(-1, 1).to(self.device)
        age_batch = (age_batch - self.min_age).reshape(-1, 1).to(self.device)
        encoded_batch = self.encoder(image_batch.to(self.device))
        bmi_batch = ((bmi_batch.to(self.device) - self.feature_mean) / self.feature_std).reshape(-1, 1)
        x = torch.hstack([sex_batch, age_batch, bmi_batch, encoded_batch]).type(torch.float32)
        return x

    def _x_to_batch(self, x):
        images = np.squeeze(self.decoder(x[:, self.ncauses:]).detach().cpu().numpy())
        age = x[:, 0].detach().cpu().numpy()
        sex = x[:, 1].detach().cpu().numpy()
        bmi = x[:, 2].detach().cpu().numpy()
        return sex, age, bmi, images

    def test_likelihood(self, dataloader):
        self.macaw.eval()
        self.encoder.eval()

        loss_val = []
        for batch in dataloader:
            with torch.no_grad():
                x = self._batch_to_x(batch)
                _, prior_logprob, log_det = self.macaw(x)
                loss = - torch.sum(prior_logprob + log_det)
                loss /= dataloader.batch_size

                loss_val.append(loss.detach().cpu().numpy())

        return np.mean(loss_val)

    def _compute_feature_mean(self, train_loader):
        features = []
        for b in train_loader:
            features.append(b[2].numpy())
        features = np.hstack(features).reshape(-1, 1)
        scaler_f = StandardScaler()
        _ = scaler_f.fit_transform(features)
        self.feature_mean = torch.tensor(scaler_f.mean_).to(self.device)
        self.feature_std = torch.tensor(np.sqrt(scaler_f.var_)).to(self.device)

    def samples(self, n_samples=5):
        self.macaw.eval()
        self.decoder.eval()

        with torch.no_grad():
            z = np.zeros((n_samples, 45))
            for sl, dist in self.macaw.priors:
                z[:, sl] = dist.sample((n_samples,)).cpu().detach().numpy()

            samples = self.macaw.backward(torch.tensor(z.astype(np.float32)).to(self.device))[0][
                -1]

            images = np.squeeze(self.decoder(samples[:, 13:]).detach().cpu().numpy())
            labels = inverse_one_hot(samples[:, :10]).detach().cpu().numpy()
            features = (samples[:, 10:13] * self.feature_std + self.feature_mean).detach().cpu().numpy()

            return images, features, labels

    def counterfactuals(self, dataloader, cf_vals):
        self.encoder.eval()
        self.macaw.eval()
        self.decoder.eval()

        with torch.no_grad():

            batch = next(iter(dataloader))
            x_obs = self._batch_to_x(batch)
            image_batch = batch[3]

            obs = image_batch.detach().cpu().numpy()

            z_obs = self.macaw(x_obs)[0][-1]
            x_cf = x_obs.detach().clone()
            for key in cf_vals:
                x_cf[:, key] = cf_vals[key]

            z_cf_val = self.macaw(x_cf)[0][-1]
            for key in cf_vals:
                z_obs[:, key] = z_cf_val[:, key]

            x_cf = self.macaw.backward(z_obs)[0][-1]

            sex, age, bmi, images = self._x_to_batch(x_cf)

        return obs, images, sex, age, bmi

    def cf_test(self, val_loader, nsamples=12):
        plt.rcParams['figure.figsize'] = (20, 10)
        fig = plt.figure()

        subfigs = fig.subfigures(nrows=7, ncols=1)

        cf_vals = {}
        obs, cfs_o, sex, age, bmi = self.counterfactuals(val_loader, cf_vals)

        axs0 = subfigs[0].subplots(nrows=1, ncols=nsamples)
        axs1 = subfigs[1].subplots(nrows=1, ncols=nsamples)
        axs2 = subfigs[2].subplots(nrows=1, ncols=nsamples)
        axs3 = subfigs[3].subplots(nrows=1, ncols=nsamples)
        axs4 = subfigs[4].subplots(nrows=1, ncols=nsamples)
        axs5 = subfigs[5].subplots(nrows=1, ncols=nsamples)
        axs6 = subfigs[6].subplots(nrows=1, ncols=nsamples)

        for i in range(nsamples):
            o = np.squeeze(obs[i])
            c = np.squeeze(cfs_o[i])
            axs0[i].imshow(o, cmap='gray')
            axs0[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)
            axs1[i].imshow(c, cmap='gray')
            axs1[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)
            axs2[i].imshow(o - c, clim=(-1, 1), cmap='seismic')
            axs2[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)

        age = 70
        cf_vals = {1: age - self.min_age}
        obs, cfs, sex, age, bmi = self.counterfactuals(val_loader, cf_vals)

        for i in range(nsamples):
            oo = np.squeeze(obs[i])
            o = np.squeeze(cfs_o[i])
            c = np.squeeze(cfs[i])
            r = oo - o

            axs3[i].imshow(c + r, cmap='gray')
            axs3[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)
            axs4[i].imshow(o - c, clim=(-1, 1), cmap='seismic')
            axs4[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)

        sex = 0
        cf_vals = {0: sex}
        obs, cfs, sex, age, bmi = self.counterfactuals(val_loader, cf_vals)

        for i in range(nsamples):
            oo = np.squeeze(obs[i])
            o = np.squeeze(cfs_o[i])
            c = np.squeeze(cfs[i])

            r = oo - o

            axs5[i].imshow(c + r, cmap='gray')
            axs5[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)
            axs6[i].imshow(o - c, clim=(-1, 1), cmap='seismic')
            axs6[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)

        plt.subplots_adjust(top=1, bottom=0, left=0, right=0.5, wspace=0, hspace=0)
        plt.close()
        return fig

    def plot_ae_outputs(self, dataloader, n=10):
        self.encoder.eval()
        self.decoder.eval()

        fig = plt.figure(figsize=(20, 4))
        targets = next(iter(dataloader))[3]
        with torch.no_grad():
            for i in range(n):
                ax = fig.add_subplot(2, n, i + 1)
                img = targets[i].unsqueeze(0).to(self.device)

                rec_img = self.decoder(self.encoder(img))

                ax.imshow(img.cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                if i == n // 2:
                    ax.set_title('Original images')

                ax = fig.add_subplot(2, n, i + 1 + n)
                ax.imshow(rec_img.cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                if i == n // 2:
                    ax.set_title('Reconstructed images')

        plt.close()
        return fig


def one_hot(a, veclen=10):
    b = torch.zeros((a.shape[0], veclen))
    b[torch.arange(a.shape[0]), a] = 1
    return b


def inverse_one_hot(b):
    return torch.argmax(b, axis=1)
