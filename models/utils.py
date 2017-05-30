import torch
import ujson
from torch.utils.data.dataset import Dataset
import random
import numpy as np
from torch.autograd import Variable

##############
# CONSTANTS #
##############

LABELS = {
        "lie": 1,
        "truth": 0
      }

TOTAL_EXAMPLES = 4100

####################
# DATASET HELPERS #
####################

def pad_tensor(tensor, length):
  D, T = tensor.size()
  zeros = torch.FloatTensor(D, length - T).zero_()
  new = torch.cat((tensor, zeros), dim = 1)
  return new

def splitIndices(dataset, config, shuffle = True):
    #random.seed(config.seed)
    length = len(dataset)
    assert(length > config.nt)
    indices = list(range(0, length))  
    if shuffle:  
        random.shuffle(indices)
    if config.nv:
        num_val = config.nv
        val = indices[0:num_val]    
        train = indices[num_val:config.nt + num_val]
    else:
        num_val = length - config.nt
        val = indices[0:num_val]    
        train = indices[num_val:]
    return train, val

# Following function are only for Hybrid text-audio models (TBD)

def padProcess(model, x):
    x = [input_to_index(ex, model.vocab) for ex in x]
    x = torch.from_numpy(pad_batch(x))
    x_var = Variable(x.type(model.config.dtype).long())
    return x_var

def pad_batch(x, pad_idx = 0):
  max_length = max([len(i) for i in x])
  for i in range(len(x)):
    pad = max_length - len(x[i])
    x[i]= x[i] + [0]*pad
  return np.array(x)

def input_to_index(source, vocab):
    source = source.lower().split(' ')
    indices = []
    for token in source:
      if token in vocab:
        indices.append(vocab[token])
      else:
        indices.append(vocab['unk'])
    return indices


#############
# DATASETS #
#############

class AudioDataset(Dataset):
    """Dataset wrapping data and target tensors. Naive implementation does data preprocessing per 'get_item' call
    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    
    Arguments:
        data_path (str): path to image folder
    """
    def __init__(self, config):
      self.labels, self.features, self.examples = None, None, None
      with open(config.feats, 'r') as data:
        self.features = ujson.load(data)
      with open(config.labels, 'r') as labels:
        self.labels = ujson.load(labels)
      assert(self.features and self.labels)
      self.examples = list(self.labels.keys())
      for key in self.examples:
        feat_tensor = (torch.FloatTensor(self.features[key])[:, 0:config.max_length]).contiguous().type(config.dtype)
        if feat_tensor.size(1) < config.max_length:
          feat_tensor = pad_tensor(feat_tensor, config.max_length)
        del self.features[key]
        self.features[key] = feat_tensor.t()
      self.num_examples = len(self.examples)

    def __getitem__(self, idx):
      key = self.examples[idx]
      label = self.labels[key]
      feats = self.features[key]
      return feats, label

    def __len__(self):
      return self.num_examples

    def printDistributions(self, indices, msg = ""):
      lies, truths = 0, 0
      for i in indices:
        key = self.examples[i]
        label = self.labels[key]
        if label == LABELS["lie"]:
          lies += 1
        else:
          truths += 1
      print("--- Distribution of {} Data ---".format(msg))
      print("TRUTH: {} ({:.2f}%)".format(truths, truths / len(indices) *100))
      print("LIE: {} ({:.2f}%)\n".format(lies, lies / len(indices) *100))  




