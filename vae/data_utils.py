import re
import json
import random
import itertools
from collections import defaultdict

# External packages
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class RatioSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, split_key, ratios=None, batch_size=16):
        """
        Split self.dataset into subsets on the value of split_key.
        Each batch is randomly sampled from each subset according to ratios.
        If ratios=None, it is uniform.
        """
        self.dataset = dataset
        self.split_key = split_key
        self.batch_size = batch_size
        self.split_idxs = self._get_split_idxs()
        self.max_dataset_len = max(len(idxs) for idxs
                                   in self.split_idxs.values())
        if ratios is None:
            self.ratios = {k: 1/len(self.split_idxs)
                           for k in self.split_idxs.keys()}
        else:
            self.ratios = ratios

    def __iter__(self):
        """
        Randomly sample from datasets according to self.ratios
        up to self.batch_size.
        """
        self.groupers = self._get_groupers()
        while True:
            try:
                batch = []
                for key in self.split_idxs.keys():
                    sub_batch = next(self.groupers[key])
                    batch.extend(sub_batch)
                batch = [i for i in batch if i is not None]
                batch = torch.tensor(batch)
                yield batch
            except StopIteration:
                break

    def __len__(self):
        max_dataset_len = ("", 0)
        for (key, idxs) in self.split_idxs.items():
            if len(idxs) > max_dataset_len[1]:
                max_dataset_len = (key, len(idxs))
        ratio = self.ratios[max_dataset_len[0]]
        group_size = int(torch.round(torch.tensor(self.batch_size * ratio)))
        num_epochs = torch.ceil(torch.tensor(max_dataset_len[1] / group_size))
        return int(num_epochs)

    def _get_split_idxs(self):
        keyval2idxs = defaultdict(list)
        for (i, datum) in enumerate(self.dataset):
            val = datum[self.split_key]
            keyval2idxs[val].append(i)
        for (val, idxs) in keyval2idxs.items():
            keyval2idxs[val] = torch.tensor(idxs)
        return keyval2idxs

    def _get_groupers(self):
        groupers = {}
        for (k, ratio) in self.ratios.items():
            group_size = torch.round(torch.tensor(self.batch_size * ratio))
            group_size = int(group_size)
            order = torch.randperm(len(self.split_idxs[k]))
            idxs = self.split_idxs[k][order]
            if len(idxs) < self.max_dataset_len:
                idxs = idxs.repeat(self.max_dataset_len // len(idxs))
                idxs = torch.cat(
                    [idxs, idxs[:(self.max_dataset_len % len(idxs))]])
            groupers[k] = self.grouper(idxs, group_size)
        return groupers

    @staticmethod
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return itertools.zip_longest(fillvalue=fillvalue, *args)


class LabeledTextDataset(torch.utils.data.Dataset):

    def __init__(self, docs, labels, word2idx, label_encoders):
        super(LabeledTextDataset, self).__init__()
        self.docs = docs
        assert isinstance(labels[0], dict)
        self.labels = labels
        if "<UNK>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<UNK>' entry.")
        if "<PAD>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<PAD>' entry.")
        self.word2idx = word2idx
        self.idx2word = {idx: word for (word, idx) in self.word2idx.items()}
        self.label_encoders = label_encoders
        self.Xs = [self.doc2tensor(doc, self.word2idx) for doc in self.docs]
        self.Ys = [self.label2tensor(lab) for lab in self.labels]

    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]

    def __len__(self):
        return len(self.Xs)

    @property
    def y_dims(self):
        dims = dict()
        for (label_name, encoder) in self.label_encoders.items():
            num_classes = len(encoder.classes_)
            if num_classes == 2:
                num_classes = 1
            dims[label_name] = num_classes
        return dims

    def doc2tensor(self, doc, word2idx):
        idxs = []
        for tok in doc:
            try:
                idxs.append(word2idx[tok])
            except KeyError:
                idxs.append(word2idx["<UNK>"])
        return torch.LongTensor([[idxs]])

    def label2tensor(self, label_dict):
        tensorized = dict()
        for (label_name, label) in label_dict.items():
            encoder = self.label_encoders[label_name]
            # CrossEntropy requires LongTensors
            # BCELoss requires FloatTensors
            if len(encoder.classes_) > 2:
                tensor_fn = torch.LongTensor
            else:
                tensor_fn = torch.FloatTensor
            enc = encoder.transform([label])
            tensorized[label_name] = tensor_fn(enc)
        return tensorized


class DenoisingTextDataset(torch.utils.data.Dataset):
    """
    Like LabeledTextDataset but the input text is a corrupted
    version of the original and the goal is to denoise it in
    order to reconstruct the original, optionally classifying
    the labels as an auxilliary task.
    """

    def __init__(self, noisy_docs, orig_docs, labels, ids,
                 word2idx, label_encoders):
        super(DenoisingTextDataset, self).__init__()
        self._dims = None
        assert len(noisy_docs) == len(orig_docs)
        assert len(noisy_docs) == len(labels)
        assert len(noisy_docs) == len(ids)
        self.noisy_docs = noisy_docs
        self.orig_docs = orig_docs
        assert isinstance(labels[0], dict)
        self.labels = labels
        self.ids = ids
        if "<UNK>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<UNK>' entry.")
        if "<PAD>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<PAD>' entry.")
        self.word2idx = word2idx
        self.idx2word = {idx: word for (word, idx) in self.word2idx.items()}
        self.label_encoders = label_encoders
        #self.noisy_Xs = [self.doc2tensor(doc) for doc in self.noisy_docs]
        #self.orig_Xs = [self.doc2tensor(doc) for doc in self.orig_docs]
        #self.Ys = [self.label2tensor(lab) for lab in self.labels]

    def __getitem__(self, idx):
        noiseX = self.doc2tensor(self.noisy_docs[idx])
        origX = self.doc2tensor(self.orig_docs[idx])
        Y = self.label2tensor(self.labels[idx])
        #return (self.noisy_Xs[idx], self.orig_Xs[idx],
        return (noiseX, origX, Y, self.ids[idx])

    def get_by_id(self, uuid):
        idx = self.ids.index(uuid)
        return self[idx]

    def __len__(self):
        return len(self.orig_docs)

    @property
    def y_dims(self):
        if self._dims is not None:
            return self._dims
        dims = dict()
        for (label_name, encoder) in self.label_encoders.items():
            num_classes = len(encoder.classes_)
            if num_classes == 2:
                num_classes = 1
            dims[label_name] = num_classes
        self._dims = dims
        return dims

    def doc2tensor(self, doc):
        idxs = []
        for tok in doc:
            try:
                idxs.append(self.word2idx[tok])
            except KeyError:
                idxs.append(self.word2idx["<UNK>"])
        return torch.LongTensor([[idxs]])

    def label2tensor(self, label_dict):
        tensorized = dict()
        for (label_name, label) in label_dict.items():
            encoder = self.label_encoders[label_name]
            # CrossEntropy requires LongTensors
            # BCELoss requires FloatTensors
            if len(encoder.classes_) > 2:
                tensor_fn = torch.LongTensor
            else:
                tensor_fn = torch.FloatTensor
            enc = encoder.transform([label])
            tensorized[label_name] = tensor_fn(enc)
        return tensorized


def get_sentences_labels(path, label_keys=None, N=-1,
                         shuffle=True, prog_bar=False):
    sentence_ids = []
    sentences = []
    labels = []
    label_counts = defaultdict(lambda: defaultdict(int))
    with open(path, 'r') as inF:
        dataiterator = enumerate(inF)
        if prog_bar is True:
            dataiterator = tqdm(dataiterator)
        for (i, line) in dataiterator:
            data = json.loads(line)
            sentence_ids.append(data["id"])
            sentences.append(data["sentence"])
            if label_keys is None:
                label_keys = [key for key in data.keys()
                              if key not in ["sentence", "id"]]
            labs = {}
            for (key, value) in data.items():
                if key not in label_keys:
                    continue
                label_counts[key][value] += 1
                labs[key] = value
            labels.append(labs)
    if shuffle is True:
        tmp = list(zip(sentences, labels, sentence_ids))
        random.shuffle(tmp)
        sentences, labels, sentence_ids = zip(*tmp)
    if N == -1:
        N = len(sentences)
    return sentences[:N], labels[:N], sentence_ids[:N], label_counts


def preprocess_sentences(sentences, SOS=None, EOS=None,
                         lowercase=True, prog_bar=False):
    sents = []
    dataiterator = sentences
    if prog_bar is True:
        dataiterator = tqdm(dataiterator)
    for sent in dataiterator:
        sent = sent.strip()
        if lowercase is True:
            sent = sent.lower()
        sent = re.sub(r"(n't)", r" \1", sent)
        sent = re.sub(r"([.!?])", r" \1", sent)
        sent = re.sub(r"[^a-zA-Z.!?']+", r" ", sent)
        sent = sent.split()
        if SOS is not None and EOS is not None:
            sent = [SOS] + sent + [EOS]
        sents.append(sent)
    return sents


def reverse_sentences(sentences):
    return [sent[::-1] for sent in sentences]


def preprocess_labels(labels, label_encoders={}, prog_bar=False):
    raw_labels_by_name = defaultdict(list)
    for label_dict in labels:
        for (label_name, lab) in label_dict.items():
            raw_labels_by_name[label_name].append(lab)

    label_encoders = dict()
    enc_labels_by_name = dict()
    dataiterator = raw_labels_by_name.items()
    if prog_bar is True:
        dataiterator = tqdm(dataiterator)
    for (label_name, labs) in dataiterator:
        if label_name in label_encoders.keys():
            # We're passing in an already fit encoder
            le = label_encoders[label_name]
        else:
            le = LabelEncoder()
        y = le.fit_transform(labs)
        label_encoders[label_name] = le
        enc_labels_by_name[label_name] = y

    return labels, label_encoders
