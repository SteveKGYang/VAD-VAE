import os
import pdb
import pickle
import random
import logging
import traceback
from collections import defaultdict

import torch
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AutogradDebugger(torch.autograd.detect_anomaly):

    def __init__(self):
        super(AutogradDebugger, self).__init__()

    def __enter__(self):
        super(AutogradDebugger, self).__enter__()
        return self

    def __exit__(self, type, value, trace):
        super(AutogradDebugger, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            self.halt(str(value))

    @staticmethod
    def halt(msg):
        print()
        print("==========================================")
        print("     Failure! Left mouse to continue.")
        print("==========================================")
        print()
        print(msg)
        pdb.set_trace()


def validate_params(params):
    valid_params = {
            "name": str,  # experiment name
            "random_seed": int,
            "data_dir": str,  # directory with {train,dev,test}.jsonl
            "combined_dataset": bool,  # whether data_dir contains a "data_source" key  # noqa
            "dataset_minibatch_ratios": dict,  # {data_source_value: [0,1]}
            "checkpoint_dir": str,
            "glove_path": str,
            "num_train_examples": int,  # -1 for all examples
            "lowercase": bool,  # whether to lowercase input
            "reverse_input": bool,
            "embedding_dim": int,  # unused if glove_path != ""
            "hidden_dim": int,  # RNN hidden dim. unused if num_rnn_layers == 1.  # noqa
            "num_rnn_layers": int,
            "bidirectional_encoder": bool,
            "bow_encoder": bool,  # overrides bidirectional_encoder
            "latent_dims": dict,
            "epochs": int,
            "batch_size": int,
            "learn_rate": float,
            "encoder_dropout": float,
            "decoder_dropout": float,
            "teacher_forcing_prob": float,
            "lambdas": dict,  # KL div weights for each latent space.
            "adversarial_loss": bool,  # Use adversarial objective.
            "mi_loss": bool,   # Use mutual info minimization objective.
            "train": bool,
            "validate": bool,
            "test": bool}
    for (key, val) in valid_params.items():
        if key not in params.keys():
            raise ValueError(f"parameter file missing '{key}'")
        if not isinstance(params[key], val):
            param_type = type(params[key])
            raise ValueError(f"Parameter '{key}' of incorrect type!")
            raise ValueError(f"  Expected '{val}' but got '{param_type}'.")

    for key in params.keys():
        if key not in valid_params.keys():
            print(f"WARNING: Ignoring unused parameter '{key}' in parameter file.")  # noqa


def load_glove(path):
    """
    Load the GLoVe embeddings from the provided path.
    Return the embedding matrix and the embedding dimension.
    Pickles the loaded embedding matrix for fast loading
    in the future.

    :param str path: Path to the embeddings. E.g.
                     `glove.6B/glove.6B.100d.txt`
    :returns: embeddings, embedding_dim
    :rtype: Tuple(numpy.ndarray, int)
    """
    bn = os.path.splitext(os.path.basename(path))[0]
    pickle_file = bn + ".pickle"
    if os.path.exists(pickle_file):
        logging.warning(f"Loading embeddings from pickle file {pickle_file} in current directory.")  # noqa
        glove = pickle.load(open(pickle_file, "rb"))
        emb_dim = list(glove.values())[0].shape[0]
        return glove, emb_dim

    vectors = []
    words = []
    idx = 0
    word2idx = {}

    with open(path, "rb") as inF:
        for line in inF:
            line = line.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    emb_dim = vect.shape[0]
    glove = {word: np.array(vectors[word2idx[word]]) for word in words}
    if not os.path.exists(pickle_file):
        pickle.dump(glove, open(pickle_file, "wb"))
    return glove, emb_dim


def get_embedding_matrix(vocab, glove):
    emb_dim = len(list(glove.values())[0])
    matrix = np.zeros((len(vocab), emb_dim), dtype=np.float32)
    found = 0
    for (i, word) in enumerate(vocab):
        try:
            matrix[i] = glove[word]
            found += 1
        except KeyError:
            matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    logging.info(f"Found {found}/{len(vocab)} vocab words in embedding.")
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    return matrix, word2idx


def load_latest_checkpoint(model, optimizer, checkpoint_dir,
                           map_location=None):
    """
    Find the most recent (in epochs) checkpoint in checkpoint dir and load
    it into the model and optimizer. Return the model and optimizer
    along with the epoch the checkpoint was trained to.
    If not checkpoint is found, return the unchanged model and optimizer,
    and 0 for the epoch.
    """
    ls = os.listdir(checkpoint_dir)
    ckpts = [fname for fname in ls if fname.endswith(".pt")]
    if len(ckpts) == 0:
        return model, optimizer, 0, None
    else:
        latest_ckpt_idx = 0
        latest_epoch = 0
        for (i, ckpt) in enumerate(ckpts):
            epoch = ckpt.replace("model_", '').replace(".pt", '')
            epoch = int(epoch)
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt_idx = i

    ckpt = torch.load(os.path.join(checkpoint_dir, ckpts[latest_ckpt_idx]),
                      map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    next_epoch = ckpt["epoch"] + 1
    return model, optimizer, next_epoch, ckpts[latest_ckpt_idx]


def pad_sequence(batch):
    """
    Pad the sequence batch with 0 vectors.
    Compute the original lengths of each sequence.
    Collect labels into a dict of tensors.
    Meant to be used as the value of the `collate_fn`
    argument to `torch.utils.data.DataLoader`.
    """
    seqs = [torch.squeeze(x) for (x, _) in batch]
    seqs_padded = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)  # 0 = <PAD>
    lengths = torch.LongTensor([len(s) for s in seqs])
    labels = defaultdict(list)
    for (_, y) in batch:
        for label_name in y.keys():
            labels[label_name].append(y[label_name])
    for label_name in labels.keys():
        labels[label_name] = torch.stack(labels[label_name])
    return seqs_padded, labels, lengths


def pad_sequence_denoising(batch):
    """
    Pad the sequence batch with 0 vectors.
    Compute the original lengths of each sequence.
    Collect labels into a dict of tensors.
    Meant to be used as the value of the `collate_fn`
    argument to `torch.utils.data.DataLoader`.
    """
    noisy_seqs = [torch.squeeze(x) for (x, _, _, _) in batch]
    noisy_seqs_padded = torch.nn.utils.rnn.pad_sequence(
            noisy_seqs, batch_first=True, padding_value=0)  # 0 = <PAD>
    seqs = [torch.squeeze(x) for (_, x, _, _) in batch]
    seqs_padded = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)  # 0 = <PAD>
    lengths = torch.LongTensor([len(s) for s in seqs])
    labels = defaultdict(list)
    for (_, _, y, _) in batch:
        for label_name in y.keys():
            labels[label_name].append(y[label_name])
    for label_name in labels.keys():
        labels[label_name] = torch.stack(labels[label_name])
    ids = [i for (_, _, _, i) in batch]
    return noisy_seqs_padded, seqs_padded, labels, lengths, ids


# === RECONSTRUCT AND LOG INPUT ===
def tensor2text(tensor, idx2word, eos_token_idx):
    """
    Given a tensor of word indices, convert it to a list of strings.
    """
    try:
        eos = torch.where(tensor == eos_token_idx)[0][0]
    except IndexError:
        eos = tensor.size(0)
    return [idx2word[i.item()] for i in tensor[:eos+1]]


def get_reconstructions(model, dataset, idx2word, idxs):
    batch = [dataset[i] for i in idxs]
    noisy_Xs, target_Xs, _, lengths, ids = pad_sequence_denoising(batch)
    noisy_Xs = noisy_Xs.to(model.device)
    target_Xs = target_Xs.to(model.device)
    lengths = lengths.to(model.device)
    output = model(noisy_Xs, lengths, teacher_forcing_prob=0.0)

    X_text = [' '.join(tensor2text(X, idx2word, model.eos_token_idx))
              for X in target_Xs.cpu().detach()]
    recon_text = [' '.join(tensor2text(r, idx2word, model.eos_token_idx))
                  for r in output["token_predictions"]]
    joined = '\n'.join([f"'{x}' ==> '{r}'" for (x, r)
                        in zip(X_text, recon_text)])
    return joined


def log_reconstructions(model, dataset, idx2word, name, epoch, logdir, n=10):
    idxs = np.random.choice(len(dataset),
                            size=n,
                            replace=False)
    # Log inputs and their reconstructions before model training
    recon_file = os.path.join(logdir, f"reconstructions_{name}.log")
    recon_str = get_reconstructions(model, dataset, idx2word, idxs)
    with open(recon_file, 'a') as outF:
        outF.write(f"EPOCH {epoch}\n")
        outF.write(recon_str + '\n')
