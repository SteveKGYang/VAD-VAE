import torch
from torch import nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForMaskedLM, AutoModel, AutoTokenizer, AutoConfig, BartTokenizer, BartConfig
from collections import namedtuple

from pytorch_transformers.my_modeling_roberta import RobertaModelwithAdapter
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding, BartEncoderLayer, BartDecoderLayer, _expand_mask, \
    _make_causal_mask, shift_tokens_right
from modeling_bart import BartModel, BartDecoder
from loss import EMDLoss, SupConLoss
from vae import losses


class RobertaClassifier(nn.Module):
    """Fine-tune RoBERTa to directly predict categorical emotions."""
    def __init__(self, check_point, num_class):
        super(RobertaClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(check_point)
        self.tokenizer = AutoTokenizer.from_pretrained(check_point)
        self.config = AutoConfig.from_pretrained(check_point)
        hidden_size = self.config.hidden_size
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(hidden_size, num_class)

    def forward(self, x, mask):
        """
        :param x: The input of PLM. Dim: [B, seq_len, D]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.dense2(x)


class BARTDecoderClassifier(nn.Module):
    """The target utterance reconstruction model with fixed representations."""
    def __init__(self, check_point, bart_check_point, num_class):
        super(BARTDecoderClassifier, self).__init__()
        self.encoder = RobertaModel.from_pretrained(check_point)
        self.decoder = BartDecoder.from_pretrained(bart_check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = AutoConfig.from_pretrained(check_point)
        hidden_size = self.config.hidden_size
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(hidden_size, num_class)
        self.decoder_start_token_id = 2
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size)
        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='none')

    def get_lm_loss(self, logits, labels, masks):
        loss = self.lm_loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        masked_loss = loss * masks.view(-1)
        return torch.mean(masked_loss)

    def forward(self, inputs, mask, decoder_inputs, decoder_masks, decoder_labels):
        """
        :param inputs: The input of PLM. Dim: [B, seq_len]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        '''decoder_input_ids = shift_tokens_right(
            x, self.config.pad_token_id, self.decoder_start_token_id
        )'''
        x = self.encoder(inputs, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_masks,
            encoder_hidden_states=x.unsqueeze(1))
        #print(decoder_outputs.last_hidden_state.shape)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        lm_loss = self.get_lm_loss(lm_logits, decoder_labels, decoder_masks)
        return self.dense2(x), lm_loss


class BARTVAEClassifier(nn.Module):
    """The VAE target utterance reconstruction model."""
    def __init__(self, check_point, bart_check_point, num_class, emo_dim, device):
        super(BARTVAEClassifier, self).__init__()
        self.device = device
        self.encoder = RobertaModel.from_pretrained(check_point)
        self.decoder = BartDecoder.from_pretrained(bart_check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = AutoConfig.from_pretrained(check_point)
        hidden_size = self.config.hidden_size
        self.decoder_start_token_id = 2
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size)
        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='none')

        self.content_dim = hidden_size - emo_dim
        self.context2params = nn.ModuleDict()
        params_layer = nn.Linear(
            # 2 for mu, logvar
            hidden_size, 2 * emo_dim)
        self.context2params["emo"] = params_layer
        leftover_layer = nn.Linear(
            hidden_size, 2 * self.content_dim)
        self.context2params["content"] = leftover_layer

        self.mi_estimators = self._get_mi_estimators()
        self.z2hidden = nn.Linear(hidden_size, hidden_size)

        self.dense1 = nn.Linear(emo_dim, emo_dim)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(emo_dim, num_class)

    def _get_mi_estimators(self):
        mi_estimators = dict()
        seen_combos = set()
        for (latent_name_i, layer_i) in self.context2params.items():
            latent_size_i = int(layer_i.out_features / 2)
            for (latent_name_j, layer_j) in self.context2params.items():
                if latent_name_i == latent_name_j:
                    continue
                if (latent_name_j, latent_name_i) in seen_combos:
                    continue
                seen_combos.add((latent_name_i, latent_name_j))
                latent_size_j = int(layer_j.out_features / 2)
                mi_hidden_size = max([latent_size_i, latent_size_j, 5])
                # mi_estimator = losses.CLUBSample(
                mi_estimator = losses.CLUB(
                    latent_size_i, latent_size_j, mi_hidden_size, self.device)
                name = f"{latent_name_i}-{latent_name_j}"
                mi_estimators[name] = mi_estimator
        return mi_estimators

    def compute_latent_params(self, context, mode):
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        for (name, layer) in self.context2params.items():
            params = layer(context)
            mu, logvar = params.chunk(2, dim=1)
            logvar = torch.tanh(logvar)
            '''if mode == 'train':
                z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            else:
                z = mu'''
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            latent_params[name] = Params(z, mu, logvar)

        return latent_params

    def get_lm_loss(self, logits, labels, masks):
        loss = self.lm_loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        masked_loss = loss * masks.view(-1)
        return torch.mean(masked_loss)

    def forward(self, inputs, mask, decoder_inputs, decoder_masks, decoder_labels, mode, labels):
        """
        :param inputs: The input of PLM. Dim: [B, seq_len]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        '''decoder_input_ids = shift_tokens_right(
            x, self.config.pad_token_id, self.decoder_start_token_id
        )'''
        x = self.encoder(inputs, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        latent_params = self.compute_latent_params(x, mode)

        #zs = [param.z for param in latent_params.values()]
        zs = [latent_params["emo"].z, latent_params["content"].z]
        zs = torch.cat(zs, dim=1)
        decoder_hidden = self.z2hidden(zs)

        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_masks,
            encoder_hidden_states=decoder_hidden.unsqueeze(1))
        #print(decoder_outputs.last_hidden_state.shape)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        #con_rep = latent_params["emo"].mu.unsqueeze(1).clone().detach()
        #con_loss = SupConLoss(features=torch.cat([latent_params["emo"].mu.unsqueeze(1), con_rep], dim=1), labels=labels)
        x = self.dropout(latent_params["emo"].z)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        lm_loss = self.get_lm_loss(lm_logits, decoder_labels, decoder_masks)
        return self.dense2(x), lm_loss, latent_params


class BARTGVAEClassifier(nn.Module):
    '''The VAE model with an unified disentangled representation for emotion.'''
    def __init__(self, check_point, bart_check_point, num_class, emo_dim, device):
        super(BARTGVAEClassifier, self).__init__()
        self.device = device
        self.encoder = RobertaModel.from_pretrained(check_point)
        self.decoder = BartDecoder.from_pretrained(bart_check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = AutoConfig.from_pretrained(check_point)
        hidden_size = self.config.hidden_size
        self.decoder_start_token_id = 2
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size)
        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='none')

        self.content_dim = hidden_size - emo_dim
        self.emo_layer = nn.Linear(
            hidden_size, emo_dim)
        self.context2params = nn.ModuleDict()
        '''params_layer = nn.Linear(
            # 2 for mu, logvar
            hidden_size, 2 * emo_dim)
        self.context2params["emo"] = params_layer'''
        leftover_layer = nn.Linear(
            hidden_size, 2 * self.content_dim)
        self.context2params["content"] = leftover_layer

        self.mi_estimators = self._get_mi_estimators()
        self.z2hidden = nn.Linear(hidden_size, hidden_size)

        self.dense1 = nn.Linear(emo_dim, emo_dim)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(emo_dim, num_class)

    def _get_mi_estimators(self):
        mi_estimators = dict()
        seen_combos = set()
        for (latent_name_i, layer_i) in self.context2params.items():
            latent_size_i = int(layer_i.out_features / 2)
            for (latent_name_j, layer_j) in self.context2params.items():
                if latent_name_i == latent_name_j:
                    continue
                if (latent_name_j, latent_name_i) in seen_combos:
                    continue
                seen_combos.add((latent_name_i, latent_name_j))
                latent_size_j = int(layer_j.out_features / 2)
                mi_hidden_size = max([latent_size_i, latent_size_j, 5])
                # mi_estimator = losses.CLUBSample(
                mi_estimator = losses.CLUB(
                    latent_size_i, latent_size_j, mi_hidden_size, self.device)
                name = f"{latent_name_i}-{latent_name_j}"
                mi_estimators[name] = mi_estimator
        return mi_estimators

    def compute_latent_params(self, context, mode):
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        for (name, layer) in self.context2params.items():
            params = layer(context)
            mu, logvar = params.chunk(2, dim=1)
            logvar = torch.tanh(logvar)
            '''if mode == 'train':
                z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            else:
                z = mu'''
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            latent_params[name] = Params(z, mu, logvar)

        return latent_params

    def get_lm_loss(self, logits, labels, masks):
        loss = self.lm_loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        masked_loss = loss * masks.view(-1)
        return torch.mean(masked_loss)

    def forward(self, inputs, mask, decoder_inputs, decoder_masks, decoder_labels, mode, labels):
        """
        :param inputs: The input of PLM. Dim: [B, seq_len]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        '''decoder_input_ids = shift_tokens_right(
            x, self.config.pad_token_id, self.decoder_start_token_id
        )'''
        x = self.encoder(inputs, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        emo_space = self.emo_layer(x)
        latent_params = self.compute_latent_params(x, mode)

        #zs = [param.z for param in latent_params.values()]
        #zs = [latent_params["emo"].z, latent_params["content"].z]
        zs = [emo_space, latent_params["content"].z]
        zs = torch.cat(zs, dim=1)
        decoder_hidden = self.z2hidden(zs)

        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_masks,
            encoder_hidden_states=decoder_hidden.unsqueeze(1))
        #print(decoder_outputs.last_hidden_state.shape)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        x = self.dropout(emo_space)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        con_rep = x.unsqueeze(1).clone().detach()
        con_loss = SupConLoss(features=torch.cat([x.unsqueeze(1), con_rep], dim=1), labels=labels)
        lm_loss = self.get_lm_loss(lm_logits, decoder_labels, decoder_masks)
        return self.dense2(x), lm_loss, latent_params, con_loss


class BARTDisentangleDecoderClassifier(nn.Module):
    """Fine-tune PLMs to directly predict categorical emotions."""
    def __init__(self, check_point, bart_check_point, num_class, emo_dim, device):
        super(BARTDisentangleDecoderClassifier, self).__init__()
        self.device = device
        self.encoder = RobertaModel.from_pretrained(check_point)
        self.decoder = BartDecoder.from_pretrained(bart_check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = AutoConfig.from_pretrained(check_point)
        hidden_size = self.config.hidden_size
        self.decoder_start_token_id = 2
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size)
        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='none')

        self.content_dim = hidden_size - emo_dim
        self.emo_layer = nn.Linear(
            hidden_size, emo_dim)
        self.content_layer = nn.Linear(hidden_size, self.content_dim)

        #self.mi_estimators = self._get_mi_estimators()
        self.z2hidden = nn.Linear(hidden_size, hidden_size)

        self.dense1 = nn.Linear(emo_dim, emo_dim)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(emo_dim, num_class)

    def _get_mi_estimators(self):
        mi_estimators = dict()
        seen_combos = set()
        for (latent_name_i, layer_i) in self.context2params.items():
            latent_size_i = int(layer_i.out_features / 2)
            for (latent_name_j, layer_j) in self.context2params.items():
                if latent_name_i == latent_name_j:
                    continue
                if (latent_name_j, latent_name_i) in seen_combos:
                    continue
                seen_combos.add((latent_name_i, latent_name_j))
                latent_size_j = int(layer_j.out_features / 2)
                mi_hidden_size = max([latent_size_i, latent_size_j, 5])
                # mi_estimator = losses.CLUBSample(
                mi_estimator = losses.CLUB(
                    latent_size_i, latent_size_j, mi_hidden_size, self.device)
                name = f"{latent_name_i}-{latent_name_j}"
                mi_estimators[name] = mi_estimator
        return mi_estimators

    def compute_latent_params(self, context, mode):
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        for (name, layer) in self.context2params.items():
            params = layer(context)
            mu, logvar = params.chunk(2, dim=1)
            logvar = torch.tanh(logvar)
            '''if mode == 'train':
                z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            else:
                z = mu'''
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            latent_params[name] = Params(z, mu, logvar)

        return latent_params

    def get_lm_loss(self, logits, labels, masks):
        loss = self.lm_loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        masked_loss = loss * masks.view(-1)
        return torch.mean(masked_loss)

    def forward(self, inputs, mask, decoder_inputs, decoder_masks, decoder_labels, mode, labels):
        """
        :param inputs: The input of PLM. Dim: [B, seq_len]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        '''decoder_input_ids = shift_tokens_right(
            x, self.config.pad_token_id, self.decoder_start_token_id
        )'''
        x = self.encoder(inputs, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        emo_space = self.emo_layer(x)
        content_space = self.content_layer(x)
        #latent_params = self.compute_latent_params(x, mode)
        latent_params = None

        #zs = [param.z for param in latent_params.values()]
        #zs = [latent_params["emo"].z, latent_params["content"].z]
        zs = [emo_space, content_space]
        zs = torch.cat(zs, dim=1)
        decoder_hidden = self.z2hidden(zs)
        #decoder_hidden = torch.tanh(decoder_hidden)

        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_masks,
            encoder_hidden_states=decoder_hidden.unsqueeze(1))
        #print(decoder_outputs.last_hidden_state.shape)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        x = self.dropout(emo_space)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        #con_rep = x.unsqueeze(1).clone().detach()
        #con_loss = SupConLoss(features=torch.cat([x.unsqueeze(1), con_rep], dim=1), labels=labels)
        lm_loss = self.get_lm_loss(lm_logits, decoder_labels, decoder_masks)
        return self.dense2(x), lm_loss, latent_params


class BARTVADVAEClassifier(nn.Module):
    """The VAD-VAE model."""
    def __init__(self, check_point, bart_check_point, num_class, emo_dim, device, batch_size, latent_variables, decoder_type):
        super(BARTVADVAEClassifier, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.decoder_type = decoder_type

        #Prepare the encoder and decoder for VAE.
        self.encoder = RobertaModel.from_pretrained(check_point)
        self.tokenizer = RobertaTokenizer.from_pretrained(check_point)
        self.config = AutoConfig.from_pretrained(check_point)

        hidden_size = self.config.hidden_size
        if decoder_type == 'BART':
            self.decoder = BartDecoder.from_pretrained(bart_check_point)
        elif decoder_type == 'LSTM':
            self.decoder = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)

        self.decoder_start_token_id = 2
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size)
        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='sum')

        #Prepare the disentanglement modules.
        self.latent_variables = latent_variables
        self.content_dim = hidden_size - emo_dim * len(self.latent_variables)
        self.context2params = nn.ModuleDict()
        for variable in self.latent_variables:
            params_layer = nn.Linear(
                # 2 for mu, logvar
                hidden_size, 2 * emo_dim)
            self.context2params[variable] = params_layer
        self.latent2regression = nn.ModuleDict()
        for variable in self.latent_variables:
            params_layer = nn.Linear(emo_dim, 1)
            self.latent2regression[variable] = params_layer
        leftover_layer = nn.Linear(
            hidden_size, 2 * self.content_dim)
        self.context2params["content"] = leftover_layer

        #The vCLUB MI estimator.
        self.mi_estimators = self._get_mi_estimators()

        self.z2hidden = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(hidden_size, num_class)

    def _get_mi_estimators(self):
        '''Get the vCLUB MI estimator.'''
        mi_estimators = dict()
        seen_combos = set()
        for (latent_name_i, layer_i) in self.context2params.items():
            latent_size_i = int(layer_i.out_features / 2)
            for (latent_name_j, layer_j) in self.context2params.items():
                if latent_name_i == "content" or latent_name_j == "content":
                    continue
                if latent_name_i == latent_name_j:
                    continue
                if (latent_name_j, latent_name_i) in seen_combos:
                    continue
                seen_combos.add((latent_name_i, latent_name_j))
                latent_size_j = int(layer_j.out_features / 2)
                mi_hidden_size = max([latent_size_i, latent_size_j, 5])
                # mi_estimator = losses.CLUBSample(
                mi_estimator = losses.CLUB(
                    latent_size_i, latent_size_j, mi_hidden_size, self.device)
                name = f"{latent_name_i}-{latent_name_j}"
                mi_estimators[name] = mi_estimator
        return mi_estimators

    def compute_latent_params(self, context, mode):
        '''Estimate the latent parameters.'''
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        for (name, layer) in self.context2params.items():
            params = layer(context)
            mu, logvar = params.chunk(2, dim=1)
            logvar = torch.tanh(logvar)
            '''if mode == 'train':
                z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            else:
                z = mu'''
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            latent_params[name] = Params(z, mu, logvar)

        return latent_params

    def get_lm_loss(self, logits, labels, masks):
        '''Get the utterance reconstruction loss.'''
        loss = self.lm_loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        masked_loss = loss * masks.view(-1)
        return torch.mean(masked_loss)

    def get_vad_loss(self, latent_params, vad_labels):
        '''Get the loss for predicting the NRC-VAD supervision signals.'''
        predicts = []
        for variable in self.latent_variables:
            logits = self.latent2regression[variable](latent_params[variable].z)
            logits = torch.sigmoid(logits)
            predicts.append(logits)
        predicts = torch.cat(predicts, dim=-1)
        return self.mse_loss(predicts, vad_labels)/self.batch_size, predicts

    def forward(self, inputs, mask, decoder_inputs, decoder_masks, decoder_labels, mode, vad_labels, labels):
        """
        :param inputs: The input of PLM. Dim: [B, seq_len]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        '''decoder_input_ids = shift_tokens_right(
            x, self.config.pad_token_id, self.decoder_start_token_id
        )'''
        x = self.encoder(inputs, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        latent_params = self.compute_latent_params(x, mode)

        zs = [latent_params[param].z for param in self.latent_variables]
        zs.append(latent_params["content"].z)
        zs = torch.cat(zs, dim=1)
        decoder_hidden = self.z2hidden(zs)

        if self.decoder_type == 'BART':
            decoder_outputs = self.decoder(
                input_ids=decoder_inputs,
                attention_mask=decoder_masks,
                encoder_hidden_states=decoder_hidden.unsqueeze(1))
            # print(decoder_outputs.last_hidden_state.shape)
            lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        elif self.decoder_type == 'LSTM':
            input_embeddings = self.encoder.embeddings(decoder_inputs)
            h = decoder_hidden.unsqueeze(0)
            decoder_outputs, (_, _) = self.decoder(input_embeddings, (h, torch.zeros(h.shape).to(self.device)))
            lm_logits = self.lm_head(decoder_outputs)
        x = self.dropout(decoder_hidden)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        lm_loss = self.get_lm_loss(lm_logits, decoder_labels, decoder_masks)
        vad_loss, vad_predicts = self.get_vad_loss(latent_params, vad_labels)
        return self.dense2(x), lm_loss, latent_params, vad_loss, vad_predicts


class ConRobertaClassifier(nn.Module):
    """The RoBERTa model enhanced with supervised contrastive learning."""
    def __init__(self, args, num_class):
        super(ConRobertaClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained(args['model_checkpoint'])
        self.tokenizer = RobertaTokenizer.from_pretrained(args['model_checkpoint'])
        self.config = RobertaConfig.from_pretrained(args['model_checkpoint'])
        hidden_size = self.config.hidden_size
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense2 = nn.Linear(hidden_size, num_class)
        self.dense21 = nn.Linear(hidden_size, 3)

    def forward(self, x, mask, labels):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].unsqueeze(1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        vad = torch.sigmoid(self.dense21(x))
        con_rep = vad.clone().detach()
        con_loss = SupConLoss(features=torch.cat([vad, con_rep], dim=1), labels=labels)
        return self.dense2(x.squeeze(1)), con_loss