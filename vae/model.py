# Built-in packages
import random
from collections import namedtuple

# External packages
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae import losses


class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, emb_matrix=None, dropout_rate=0.5):
        super(BOWEncoder, self).__init__()
        self._device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        if emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(emb_matrix))
            self.embedding.weight.requires_grad = False
            self.vocab_size, self.emb_dim = emb_matrix.shape
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # compatibility AHH!
        self.hidden_size = emb_dim
        self.num_layers = 1
        self.num_directions = 1

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    # *args for compatibility with VariationalEncoder
    def forward(self, inputs, *args):
        # inputs: [batch_size, max(lengths)]
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, max(lengths), emb_dim]
        # Max over positions in the sentence
        max_vals, max_idxs = torch.max(embedded, 1)
        return max_vals


class VariationalEncoder(nn.Module):
    """
    embedding -> dropout -> LSTM
    """
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers,
                 dropout_rate=0.5, emb_matrix=None, bidirectional=False):
        super(VariationalEncoder, self).__init__()
        self._device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional is True else 1
        self.dropout_rate = dropout_rate
        if emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(emb_matrix))
            self.embedding.weight.requires_grad = False
            self.vocab_size, self.emb_dim = emb_matrix.shape
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.recurrent = nn.LSTM(self.emb_dim, self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout_rate, batch_first=True,
                                 bidirectional=bidirectional)

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    def forward(self, inputs, lengths, hidden):
        # inputs: [batch_size, max(lengths)]
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, max(lengths), self.emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
        # packed: [sum(lengths), self.emb_dim]
        encoded, hidden = self.recurrent(packed, hidden)
        # encoded: [sum(lengths), hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_size]
        unpacked, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(
                encoded, batch_first=True)
        # unpacked: [batch_size, max(lengths), hidden_size * num_directions]
        return unpacked, hidden

    def init_hidden(self, batch_size):
        # Initialize the LSTM state.
        # One for hidden and one for the cell
        return (torch.zeros(self.num_layers * self.num_directions, batch_size,
                            self.hidden_size, device=self.device),
                torch.zeros(self.num_layers * self.num_directions, batch_size,
                            self.hidden_size, device=self.device))


class VariationalDecoder(nn.Module):
    """
    LSTM -> linear -> token_predictions
    """
    def __init__(self, vocab_size, emb_dim, hidden_size,
                 num_layers, dropout_rate=0.5, emb_matrix=None):
        super(VariationalDecoder, self).__init__()
        self._device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        if num_layers == 1:
            num_layers = 2
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        if emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(emb_matrix))
            self.embedding.weight.requires_grad = False
            self.vocab_size, self.emb_dim = emb_matrix.shape
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.recurrent = nn.LSTM(self.emb_dim,
                                 self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout_rate, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    def forward(self, inputs, lengths, hidden):
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, len(inputs), emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
        # packed: [sum(lengths), self.emb_dim]
        decoded, hidden = self.recurrent(packed, hidden)
        # decoded: [batch_size, max(lengths), self.hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        unpacked, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(
                decoded, batch_first=True)
        # logits: [batch_size, len(inputs), vocab_size]
        logits = self.linear(unpacked)
        return logits, hidden


class Discriminator(nn.Module):
    def __init__(self, name, latent_dim, output_dim):
        super(Discriminator, self).__init__()
        self._device = torch.device("cpu")
        self.name = name
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(latent_dim, output_dim)
        assert self.output_dim > 0
        if self.output_dim == 1:
            self.loss_fn = F.binary_cross_entropy_with_logits
            self.activation = torch.sigmoid
            self.activation_args = []
        else:
            self.loss_fn = F.cross_entropy
            self.activation = torch.softmax
            self.activation_args = [-1]

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    def forward(self, inputs):
        return self.linear(inputs)

    # TODO: add parameter to pass label weights for balancing
    def compute_loss(self, logits, targets):
        if self.output_dim > 1:
            targets = targets.squeeze()
        return self.loss_fn(logits, targets)

    def predict(self, logits):
        probs = self.activation(logits, *self.activation_args)
        if probs.size(1) == 1:
            preds = (probs > 0.5).long().squeeze()
        else:
            preds = probs.argmax(-1).squeeze()
        return preds

    def compute_accuracy(self, logits, targets):
        preds = self.predict(logits)
        targets = targets.squeeze()
        acc = torch.mean((preds == targets).float())
        return acc


class AdversarialDiscriminator(Discriminator):
    def __init__(self, latent_name, label_name, latent_dim, output_dim):
        name = f"{latent_name}-{label_name}"
        super(AdversarialDiscriminator, self).__init__(
            name, latent_dim, output_dim)
        self.latent_name = latent_name
        self.label_name = label_name
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.detached_inputs = None

    def forward(self, inputs):
        self.detached_inputs = inputs.clone().detach()
        return self.linear(inputs)

    def compute_discriminator_loss(self, logits, targets):
        detached_logits = self.forward(self.detached_inputs)
        if self.output_dim > 1:
            targets = targets.squeeze()
        return self.loss_fn(detached_logits, targets)

    def optimizer_step(self, dsc_loss):
        """
        dsc_loss: output of compute_discriminator_loss
        """
        dsc_loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def compute_adversarial_loss(self, logits):
        """
        We want to maximise the entropy of the logits.
        """
        if len(logits.size()) == 1:
            logits = logits.unsqueeze(1)
        elif len(logits.size()) > 2:
            raise ValueError(f"Got unexpected logits shape {logits.size()}")
        probs = self.activation(logits, *self.activation_args)
        probs = torch.clamp(probs, min=1e-8, max=1 - 1e-8)
        H = -torch.sum(probs * torch.log(probs), dim=1).mean()
        return -H  # Negate just to be clear we want to maximise it.


class VariationalSeq2Seq(nn.Module):
    """
    total_latent_dim = 5
    polarity_dsc = Discriminator("polarity", 1, 1)
    modality_dsc = Discriminator("modality", 2, 5)
    # vae will have a 2-dimensional leftover "content" latent space
    vae = VariationalSeq2Seq(encoder, decoder, total_latent_dim,
                             [polarity_dsc, modality_dsc], sos, eos)
    """
    def __init__(self, encoder, decoder, discriminators, latent_dim,
                 sos_token_idx, eos_token_idx, adversarial_loss=False,
                 mi_loss=False):
        super(VariationalSeq2Seq, self).__init__()
        self._device = torch.device("cpu")
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.discriminators = nn.ModuleDict()  # Name: Discriminator
        self.context2params = nn.ModuleDict()  # Name: (mu,logvar) linear layer

        # Classifier heads
        # Total latent dimensions of the discriminators
        self.dsc_latent_dim = 0
        linear_insize = encoder.hidden_size * \
            encoder.num_layers * \
            encoder.num_directions
        for dsc in discriminators:
            self.dsc_latent_dim += dsc.latent_dim
            self.discriminators[dsc.name] = dsc
            params_layer = nn.Linear(
                    # 2 for mu, logvar
                    linear_insize, 2 * dsc.latent_dim)
            self.context2params[dsc.name] = params_layer
        assert self.dsc_latent_dim <= self.latent_dim

        # Left over latent dims are treated as a generic "content" space
        if self.dsc_latent_dim < self.latent_dim:
            leftover_latent_dim = self.latent_dim - self.dsc_latent_dim
            leftover_layer = nn.Linear(
                linear_insize, 2 * leftover_latent_dim)
            self.context2params["content"] = leftover_layer
            assert self.dsc_latent_dim + leftover_latent_dim == self.latent_dim

        # Adversarial heads
        self.adversarial_loss = adversarial_loss
        if self.adversarial_loss is True:
            self.adversaries = self._get_adversaries()
        else:
            self.adversaries = dict()

        # MI Estimators
        self.mi_loss = mi_loss
        if self.mi_loss is True:
            self.mi_estimators = self._get_mi_estimators()
        else:
            self.mi_estimators = dict()

        self.z2hidden = nn.Linear(
                self.latent_dim, 2 * decoder.hidden_size * decoder.num_layers)
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx

    def _get_adversaries(self):
        adversaries = nn.ModuleDict()
        for (latent_name, layer) in self.context2params.items():
            latent_size = int(layer.out_features / 2)
            for (label_name, dsc) in self.discriminators.items():
                if latent_name == label_name:
                    continue
                # Try to predict label_name from latent_name
                adversary = AdversarialDiscriminator(
                    latent_name, label_name, latent_size, dsc.output_dim)
                name = f"{latent_name}-{label_name}"
                adversaries[name] = adversary
        return adversaries

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
                    latent_size_i, latent_size_j, mi_hidden_size)
                name = f"{latent_name_i}-{latent_name_j}"
                mi_estimators[name] = mi_estimator
        return mi_estimators

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)
        for mi_estimator in self.mi_estimators.values():
            mi_estimator.to(value)

    def trainable_parameters(self):
        return [param for (name, param) in self.named_parameters()
                if param.requires_grad is True
                and not name.startswith("adversaries")]

    def encode(self, inputs, lengths):
        batch_size = inputs.size(0)
        # state, cell: [num_layers, batch_size, hidden_size]
        state, cell = self.encoder.init_hidden(batch_size)
        # encoded: [batch_size, max(lengths), hidden_size]
        encoded, (hidden_state, hidden_cell) = self.encoder(
                inputs, lengths, (state, cell))
        # context: [batch_size, num_layers * hidden_size]
        state_context = torch.cat([layer for layer in hidden_state], dim=1)
        return encoded, state_context, (hidden_state, hidden_cell)

    def compute_latent_params(self, context):
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        for (name, layer) in self.context2params.items():
            params = layer(context)
            mu, logvar = params.chunk(2, dim=1)
            logvar = torch.tanh(logvar)
            if self.training is True:
                z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            else:
                z = mu
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            latent_params[name] = Params(z, mu, logvar)

        return latent_params

    def compute_hidden(self, z, batch_size):
        # hidden: [batch_size, 2 * hidden_size * decoder.num_layers]
        # TODO: why am I using tanh? Try with no activation.
        hidden = torch.tanh(self.z2hidden(z))
        # state, cell: [batch_size, hidden_size * decoder.num_layers]
        state, cell = hidden.chunk(2, dim=1)
        # state, cell = [num_layers, batch_size, hidden_size]
        state = state.chunk(self.decoder.num_layers, dim=-1)
        state = torch.stack(state, dim=0)
        cell = cell.chunk(self.decoder.num_layers, dim=-1)
        cell = torch.stack(cell, dim=0)
        return (state, cell)

    def forward(self, inputs, lengths, teacher_forcing_prob=0.5):
        # inputs: [batch_size, max(lengths)]
        batch_size = inputs.size(0)

        if isinstance(self.encoder, BOWEncoder):
            context = self.encoder(inputs)
        else:
            encoded, context, encoder_hidden = self.encode(inputs, lengths)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        latent_params = self.compute_latent_params(context)

        # Forward pass for each discriminator
        dsc_logits = {}
        for (name, dsc) in self.discriminators.items():
            dlogits = dsc(latent_params[name].z)
            dsc_logits[name] = dlogits

        adv_logits = {}
        for (name, adv) in self.adversaries.items():
            latent_name = name.split('-')[0]
            alogits = adv(latent_params[latent_name].z)
            adv_logits[name] = alogits

        # TODO: it only is consistent because I set the random seed.
        #       sort latent_params by key, but always putting content last.
        zs = [param.z for param in latent_params.values()]
        z = torch.cat(zs, dim=1)
        decoder_hidden = self.compute_hidden(z, batch_size)
        # decoder_hidden = encoder_hidden

        decoder_input = torch.LongTensor(
                [[self.sos_token_idx]]).to(self.device)
        decoder_input = decoder_input.repeat(batch_size, 1)
        input_lengths = [1] * batch_size
        vocab_size = self.decoder.vocab_size
        target_length = inputs.size(-1)
        # Placeholder for predictions
        out_logits = torch.zeros(
                batch_size, target_length, vocab_size).to(self.device)
        out_logits[:, 0, self.sos_token_idx] = 1.0  # Always output <SOS> first
        out_predictions = torch.zeros(batch_size, target_length, dtype=int)
        out_predictions[:, 0] = self.sos_token_idx
        for i in range(1, target_length):
            # logits: [batch_size, 1, vocab_size]
            logits, decoder_hidden = self.decoder(
                    decoder_input, input_lengths, decoder_hidden)
            logits = logits.squeeze()
            out_logits[:, i, :] = logits
            use_teacher_forcing = random.random() < teacher_forcing_prob
            if use_teacher_forcing is True:
                target = inputs[:, i]
                decoder_input = torch.unsqueeze(target, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                decoder_input = torch.multinomial(probs, 1)
                # logprobs = torch.log_softmax(logits, dim=-1)
                # decoder_input = logprobs.argmax(-1).unsqueeze(1).detach()
            out_predictions[:, i] = decoder_input.squeeze()

        # decoder_logits: (batch_size, target_length, vocab_size)
        # latent_params: dict({latent_name: Params})
        # dsc_logits: dict({dsc_name: dsc_logits})
        output = {"decoder_logits": out_logits,
                  "latent_params": latent_params,  # Params(z, mu, logvar)
                  "dsc_logits": dsc_logits,
                  "adv_logits": adv_logits,
                  "token_predictions": out_predictions}
        return output

    def sample(self, z, max_length=30):
        batch_size = z.size(0)
        decoder_hidden = self.compute_hidden(z, batch_size)
        decoder_input = torch.LongTensor(
            [[self.sos_token_idx]]).to(self.device)
        decoder_input = decoder_input.repeat(batch_size, 1)
        input_lengths = [1] * batch_size
        # Placeholder for predictions
        vocab_size = self.decoder.vocab_size
        out_logits = torch.zeros(
                batch_size, max_length, vocab_size).to(self.device)
        out_logits[:, 0, self.sos_token_idx] = 1.0
        predictions = torch.zeros(batch_size, max_length, dtype=int)
        predictions[:, 0] = self.sos_token_idx
        for i in range(1, max_length):
            # logits: [batch_size, 1, vocab_size]
            logits, decoder_hidden = self.decoder(
                    decoder_input, input_lengths, decoder_hidden)
            logits = logits.squeeze()
            out_logits[:, i, :] = logits
            probs = torch.softmax(logits, dim=-1)
            decoder_input = torch.multinomial(probs, 1)
            if len(decoder_input.size()) == 1:
                decoder_input = decoder_input.unsqueeze(0)
            predictions[:, i] = decoder_input.squeeze()

        output = {"decoder_logits": out_logits,
                  "token_predictions": predictions}
        return output


def build_vae(params, vocab_size, emb_matrix, label_dims, device,
              sos_token_idx, eos_token_idx):
    """
    :param dict params: Dict of parameters stored in config.json
    :param int vocab_size: Number of tokens in the vocabulary
    :param numpy.ndarray emb_matrix: Matrix of embeddings for
                                     each word in vocab. If None,
                                     the model uses random initialization
    :param dict label_dims: Dict of label_names and their dimensionality
    :param torch.device device: Device on which to put the model
    :param int {sos,eos}_token_idx: Index in vocab of <SOS>/<EOS> tokens
    """
    if params["bow_encoder"] is True:
        encoder = BOWEncoder(vocab_size, params["embedding_dim"],
                             emb_matrix=emb_matrix,
                             dropout_rate=params["encoder_dropout"])
    else:
        encoder = VariationalEncoder(
            vocab_size, params["embedding_dim"], params["hidden_dim"],
            params["num_rnn_layers"], dropout_rate=params["encoder_dropout"],
            emb_matrix=emb_matrix, bidirectional=params["bidirectional_encoder"])  # noqa
    encoder.set_device(device)

    decoder = VariationalDecoder(
            vocab_size, params["embedding_dim"], params["hidden_dim"],
            params["num_rnn_layers"], dropout_rate=params["decoder_dropout"],
            emb_matrix=emb_matrix)
    decoder.set_device(device)

    discriminators = []
    for (name, outdim) in label_dims.items():
        if name not in params["latent_dims"]:
            continue
        latent_dim = params["latent_dims"][name]
        dsc = Discriminator(name, latent_dim, outdim)
        dsc.set_device(device)
        discriminators.append(dsc)

    vae = VariationalSeq2Seq(encoder, decoder, discriminators,
                             params["latent_dims"]["total"],
                             sos_token_idx, eos_token_idx,
                             adversarial_loss=params["adversarial_loss"],
                             mi_loss=params["mi_loss"])
    vae.set_device(device)
    return vae
