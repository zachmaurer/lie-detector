
import torch
import ujson
from torch.utils.data.dataset import Dataset
import random
import numpy as np
from torch.autograd import Variable
import re
import json
from datetime import datetime
import os

##########
# Logger #
#########

class Logger():
  def __init__(self, title="csv_results"):
    self.timestamp = datetime.now().strftime("%m%d_%H.%M.%S")
    self._save_path = "./" + self.timestamp + "_" + title
    self.results = os.path.join(self._save_path, "results.csv")
    self.distributions = os.path.join(self._save_path, "distributions.csv")
    os.mkdir(self._save_path)

  def logResult(self, row):
    with open(self.results, 'a') as f:
        print(row, file = f)

  def logDistribution(self, row):
    with open(self.distributions, 'a') as f:
        print(row, file = f)

  def __str__(self):
    return self._save_path

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

def splitIndices(dataset, num_train, num_val=False, shuffle = True):
    #random.seed(config.seed)
    length = len(dataset)
    if num_val:
      assert(length > num_val)
    indices = list(range(0, length))  
    if shuffle:  
        random.shuffle(indices)
    if num_val:
        val = indices[0:num_val]    
        train = indices[num_val:num_train + num_val]
    else:
        #print(num_val, length, num_train)
        num_val = length - int(num_train)
        val = indices[0:num_val]    
        train = indices[num_val:]
    return train, val

# Following function are only for Hybrid text-audio models (TBD)

# def padProcess(model, x):
#     x = [input_to_index(ex, model.vocab) for ex in x]
#     x = torch.from_numpy(pad_batch(x))
#     x_var = Variable(x.type(model.config.dtype).long())
#     return x_var

# def pad_batch(x, pad_idx = 0):
#   max_length = max([len(i) for i in x])
#   for i in range(len(x)):
#     pad = max_length - len(x[i])
#     x[i]= x[i] + [0]*pad
#   return np.array(x)

# def input_to_index(source, vocab):
#     source = source.lower().split(' ')
#     indices = []
#     for token in source:
#       if token in vocab:
#         indices.append(vocab[token])
#       else:
#         indices.append(vocab['unk'])
#     return indices


#################
# AUDIO DATASET #
#################
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

"""
Super hacky way to extract int value of subject from the data we are reading, gets first number out of the filename
"""
def get_id(audiofile_name):
    return int(re.search(r'\d+', audiofile_name).group())
"""
returns two AudioDatasets: one that contains all subjects except those specified to be held out for testing, and one that contains only the held out subjects.
If no heldout subjects are specified, then simply returns all the data in one dataset and returns None instead of a test dataset.
"""
def getAudioDatasets(config,hold_out=False):
    if hold_out:
      #hold out everything from the test set that will be in the training set
      hold_out_test = {x for x in range(1,33) if x not in hold_out} 
      return AudioDataset(config, hold_out=hold_out), AudioDataset(config, hold_out=hold_out_test)
    else:
      return AudioDataset(config), None

def getHybridDatasets(config,vocab,hold_out = False):
  if hold_out:
    #hold out everything from the test set that will be in the training set
    hold_out_test = {x for x in range(1,33) if x not in hold_out} 
    return HybridDataset(config,vocab, hold_out=hold_out), HybridDataset(config,vocab, hold_out=hold_out_test)
  else:
    return HybridDataset(config,vocab), None

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
    def __init__(self, config,hold_out = False):
      self.labels, self.features, self.examples = None, None, None
      with open(config.feats, 'r') as data:
        self.features = json.load(data)
      with open(config.labels, 'r') as labels:
        self.labels = json.load(labels)
      assert(self.features and self.labels)
      self.examples = []
      """
      If we want to hold a speaker out for test time, then hold_out should be a set holding all the integers for the speakers to hold out
      """
      if hold_out:
        self.examples = [k for k in self.labels.keys() if get_id(k) not in hold_out]
      else:
        self.examples = [k for k in self.labels.keys()]




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
      return feats, label, key

    def __len__(self):
      return self.num_examples

    def printDistributions(self, indices, msg = "", logger = None, hold_out = -1):
      lies, truths = 0, 0
      for i in indices:
        key = self.examples[i]
        label = self.labels[key]
        if label == LABELS["lie"]:
          lies += 1
        else:
          truths += 1
      print("--- Distribution of {} Data ---".format(msg))
      print("TRUTH: {} ({:.2f}%)".format(truths, 1.0*truths / len(indices) *100))
      print("LIE: {} ({:.2f}%)\n".format(lies, 1.0*lies / len(indices) *100))
      if logger:
        row = '{},{},{},{}'.format(msg, hold_out, 1.0*truths / len(indices) *100, 1.0*lies / len(indices) *100)
        logger.logDistribution(row)  


##################
# HYBRID DATASET #
##################

class HybridDataset(AudioDataset):
    """Dataset wrapping data and target tensors. Naive implementation does data preprocessing per 'get_item' call
    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    
    Arguments:
        data_path (str): path to image folder
    """
    def __init__(self, config, vocab, hold_out=False):
      super().__init__(config, hold_out)
      # Load Transcripts
      self.transcripts = ujson.load(open(config.transcripts, 'r'))
      self.vocab = vocab
      self.config = config

    def __getitem__(self, idx):
      key = self.examples[idx]
      label = self.labels[key]
      feats = self.features[key]
      transcript = self._EncodeTranscript(self.transcripts[key], self.config.transcript_length)
      transcript = torch.LongTensor(transcript) if not self.config.use_gpu else torch.LongTensor(transcript).cuda()
      return feats, transcript, label, key

    def _EncodeTranscript(self, transcript, max_length):
      source = transcript.split(' ')
      indices = []
      for i, token in enumerate(source):
        if token in self.vocab:
          indices.append(self.vocab[token])
        else:
          indices.append(self.vocab['unk'])
      indices = indices + [self.vocab['<PAD>']]*(max_length - len(indices))
      indices = indices[0:max_length]
      return indices




