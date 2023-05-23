import torch
import torch.nn as nn
import numpy as np
#import texar.torch as tx
from torchtext.data.metrics import bleu_score

from vae import utils


class CLUB(nn.Module):
    '''
    Credit to https://github.com/Linear95/CLUB

    Pengyu Cheng, Weituo Hao, Shuyang Dai, Jiachang Liu,
    Zhe Gan, Lawrence Carin Proceedings of the 37th International
    Conference on Machine Learning, PMLR 119:1779-1788, 2020.

        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() : provides the estimation with input samples
            loglikeli() : provides the log-likelihood of the approximation
                            q(Y|X) with input samples
        Arguments:
            x_dim, y_dim : the dimensions of samples from X, Y respectively
            hidden_size : the dimension of the hidden layer of the
                          approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape
                                   [sample_size, x_dim/y_dim]
    '''
    def __init__(self, x_dim, y_dim, hidden_size, device):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim)).to(device)
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh()).to(device)
        # DO NOT CHANGE LEARNING RATE! IT WORKS NOW BUT WONT IF YOU CHANGE IT!
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

    def optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp()  # noqa
        mi_est = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return mi_est

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)  # noqa

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample(nn.Module):
    """
    Credit to https://github.com/Linear95/CLUB

    Pengyu Cheng, Weituo Hao, Shuyang Dai, Jiachang Liu,
    Zhe Gan, Lawrence Carin Proceedings of the 37th International
    Conference on Machine Learning, PMLR 119:1779-1788, 2020.
    """
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)

    def optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)  # noqa

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


def compute_bleu(Xbatch, pred_batch, idx2word, eos_token_idx):
    Xtext = [[utils.tensor2text(X, idx2word, eos_token_idx)[1:-1]]  # RM SOS and EOS   # noqa
             for X in Xbatch.cpu().detach()]
    pred_text = [utils.tensor2text(pred, idx2word, eos_token_idx)[1:-1]
                 for pred in pred_batch.cpu().detach()]
    bleu = bleu_score(pred_text, Xtext)
    return bleu


'''def reconstruction_loss(targets, logits, target_lengths):
    recon_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=targets, logits=logits, sequence_length=target_lengths)
    return {"reconstruction_loss": recon_loss}'''


def get_cyclic_kl_weight(step, total_steps, cycles=4, rate=0.5):
    denom = total_steps / cycles
    numer = step % np.ceil(denom)
    tau = numer / denom
    if tau <= rate:
        return tau / rate
    else:
        return 1


def kl_divergence(mu, logvar):
    kl = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar)
    kl = kl.mean(0).sum()
    return kl


def compute_kl_divergence_losses(model, latent_params, kl_weights_dict):
    # KL for each latent space
    idv_kls = dict()
    # total kl over all latent spaces
    total_kl = 0.0  # scalar for logging
    # tensor scalar for backward pass
    total_weighted_kl = torch.tensor(0.0).to(model.device)
    #total_weighted_kl = torch.tensor(0.0)
    for (latent_name, latent_params) in latent_params.items():
        kl = kl_divergence(latent_params.mu, latent_params.logvar)
        idv_kls[latent_name] = kl.item()
        total_kl += kl.item()
        try:
            weight = kl_weights_dict[latent_name]
        except KeyError:
            weight = kl_weights_dict["default"]
        total_weighted_kl += weight * kl
    return {"total_weighted_kl": total_weighted_kl,
            "total_kl": total_kl,
            "idv_kls": idv_kls}


def compute_discriminator_losses(model, discriminator_logits, Ybatch):
    # Loss and accuracy for each discriminator
    idv_dsc_losses = dict()
    idv_dsc_accs = dict()
    # total loss over all discriminators
    total_dsc_loss = torch.tensor(0.0).to(model.device)
    for (dsc_name, dsc_logits) in discriminator_logits.items():
        dsc = model.discriminators[dsc_name]
        targets = Ybatch[dsc_name].to(model.device)
        dsc_loss = dsc.compute_loss(dsc_logits, targets)
        dsc_acc = dsc.compute_accuracy(dsc_logits, targets)
        idv_dsc_losses[dsc_name] = dsc_loss.item()
        idv_dsc_accs[dsc_name] = dsc_acc.item()
        total_dsc_loss += dsc_loss
    return {"total_dsc_loss": total_dsc_loss,
            "idv_dsc_losses": idv_dsc_losses,
            "idv_dsc_accs": idv_dsc_accs}


def compute_adversarial_losses(model, adversary_logits, Ybatch):
    # Adversarial loss for each individual adversary
    idv_adv_losses = dict()
    # Discriminator loss for each individual adversary
    idv_dsc_losses = dict()
    # Accuracies of the discriminators
    idv_dsc_accs = dict()
    # total loss over all adversarial discriminators
    total_adv_loss = torch.tensor(0.0).to(model.device)
    for (adv_name, adv_logits) in adversary_logits.items():
        adv = model.adversaries[adv_name]
        latent_name, label_name = adv_name.split('-')
        targets = Ybatch[label_name].to(model.device)
        adv_loss = adv.compute_adversarial_loss(adv_logits)
        idv_adv_losses[adv_name] = adv_loss.item()
        total_adv_loss += adv_loss
        # This will be used to update the adversaries
        dsc_loss = adv.compute_discriminator_loss(adv_logits, targets)
        idv_dsc_losses[adv_name] = dsc_loss
        dsc_acc = adv.compute_accuracy(adv_logits, targets)
        idv_dsc_accs[adv_name] = dsc_acc.item()
    return {"total_adv_loss": total_adv_loss,
            "idv_adv_losses": idv_adv_losses,
            "idv_adv_dsc_losses": idv_dsc_losses,
            "idv_adv_dsc_accs": idv_dsc_accs}


def compute_mi_losses(model, latent_params, beta=1.0):
    idv_mi_estimates = dict()
    #total_mi = torch.tensor(0.0)
    total_mi = torch.tensor(0.0).to(model.device)
    for (latent_name_1, params1) in latent_params.items():
        for (latent_name_2, params2) in latent_params.items():
            if latent_name_1 == "content" or latent_name_2 == "content":
                continue
            if latent_name_1 == latent_name_2:
                continue
            try:
                name = f"{latent_name_1}-{latent_name_2}"
                mi_estimator = model.mi_estimators[name]
            except KeyError:
                continue
            mi_estimate = mi_estimator(params1.z, params2.z) * beta
            idv_mi_estimates[name] = mi_estimate.item()
            total_mi += mi_estimate
    return {"total_mi": total_mi,
            "idv_mi_estimates": idv_mi_estimates}
