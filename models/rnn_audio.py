
# Python Imports
import argparse
import copy

# Torch Imports
import torch
#from torch import LongTensor
#from torchtext.vocab import load_word_vectors
from torch.autograd import Variable
#from torch.nn.utils.rnn import pack_padded_sequence#, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch import cuda, FloatTensor
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
# Our modules
from models import *
from utils import *


############
## CONFIG ##
############

class Config:
  def __init__(self, args):
    self.epochs = args.e
    self.batch_size = args.bs
    self.lr = args.lr
    self.nt = args.nt
    self.nv = args.nv
    self.print_every = args.pe
    self.hidden_size = args.hs
    self.feats = args.feats
    self.labels = args.labels
    self.max_length = args.length
    #self.eval_every = args.ee
    self.use_gpu = args.gpu
    self.dtype = cuda.FloatTensor if self.use_gpu else FloatTensor
    self.num_classes = 2
    self.finetuning_lr = args.ftlr
    self.finetuning_epochs = args.fte

  def __str__(self):
    properties = vars(self)
    properties = ["{} : {}".format(k, str(v)) for k, v in properties.items()]
    properties = '\n'.join(properties)
    properties = "--- Config --- \n" + properties + "\n"
    return properties

def parseConfig(description="Default Model Description"):
  parser = argparse.ArgumentParser(description=description)
  
  parser.add_argument('--feats', type=str, help='input features path', default = "../data/features/extracted_features_0.1_0.05.json")
  parser.add_argument('--labels', type=str, help='input labels', default = "../data/features/labels_0.1_0.05.json")
  parser.add_argument('--length', type=int, help='length of sequence', default = 300)
  parser.add_argument('--bs', type=int, help='batch size for training', default = 20)
  parser.add_argument('--e', type=int, help='number of epochs', default = 10)
  parser.add_argument('--nt', type=int, help='number of training examples', default = 3000)
  parser.add_argument('--nv', type=int, help='number of validation examples', default = None)
  parser.add_argument('--hs', type=int, help='hidden size', default = 100)
  parser.add_argument('--lr', type=float, help='learning rate', default = 1e-3)
  parser.add_argument('--gpu', action='store_true', help='use gpu', default = False)
  parser.add_argument('--pe', type=int, help='print frequency', default = None)
  parser.add_argument('--ee', type=int, help='eval frequency', default = None)
  parser.add_argument('--fte', type=int, help='number of finetuning epochs', default = 5)
  parser.add_argument('--ftlr', type=int, help='finetuning learning rate', default = 2e-3)
  args = parser.parse_args()
  return args

############
# TRAINING #
############

def train(model, loss_fn, optimizer, num_epochs = 1, logger = None, hold_out = -1):
  best_model = None
  best_val_acc = 0
  for epoch in range(num_epochs):
      print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
      model.train()
      loss_total = 0
      for t, (x, y, _) in enumerate(model.config.train_loader):
          x_var = Variable(x)
          y_var = Variable(y.type(model.config.dtype).long())
          scores = model(x_var) 
          loss = loss_fn(scores, y_var)
          loss_total += loss.data[0]
          optimizer.zero_grad()
          loss.backward()

          optimizer.step()

          if ((t+1) % 10) == 0:
            grad_magnitude = [(x.grad.data.sum(), torch.numel(x.grad.data)) for x in model.parameters() if x.grad.data.sum() != 0.0]
            grad_magnitude = sum([abs(x[0]) for x in grad_magnitude]) #/ sum([x[1] for x in grad_magnitude])
            print('t = %d, avg_loss = %.4f, grad_mag = %.2f' % (t + 1, loss_total / (t+1), grad_magnitude))
          
          
      print("--- Evaluating ---")
      check_accuracy(model, model.config.train_loader, type = "train")
      val_acc = check_accuracy(model, model.config.val_loader, type = "val")
      if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(model)
      print("\n")
  print("\n--- Final Evaluation ---")
  check_accuracy(model, model.config.train_loader, type = "train", logger = logger, hold_out = hold_out)
  check_accuracy(model, model.config.val_loader, type = "val", logger = logger, hold_out = hold_out)
  # Return model with best validation accuracy
  return best_model
  #check_accuracy(model, model.config.test_loader, type = "test")


def check_accuracy(model, loader, type="", logger = None, hold_out = -1):
  print("Checking accuracy on {} set".format(type))
  num_correct = 0
  num_samples = 0
  examples, all_labels, all_predicted = [], [], []
  model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
  for t, (x, y, keys) in enumerate(loader):
      x_var = Variable(x)
      #y_var = Variable(y.type(model.config.dtype).long())
      scores = model(x_var)
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
      examples.extend(keys)
      all_labels.extend(list(y))
      all_predicted.extend(list(np.ndarray.flatten(preds.numpy())))
      #print("Completed evaluating {} examples".format(t*model.config.batch_size))
  acc = float(num_correct) / num_samples
  print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
  if logger:
    for i in range(len(examples)):
      row = "{},{},{},{},{}".format(type, examples[i], all_labels[i], all_predicted[i], hold_out)
      logger.logResult(row)
  return acc

def eval_on_test_set(model,loss_fn,num_epochs=1, logger = None, hold_out = -1):
  #first check the accuracy of the model on all of the data
  print("Trained model on all test data:")
  check_accuracy(model,model.config.test_loader_all,type="test", logger = logger, hold_out = hold_out)
  """
  #Finetuning doesn't work
  print "Now trying finetuning"
  #Freeze the layers, replace the final layer with a fully connected layer
  num_ft = model.fc.in_features
  for param in model.parameters():
    param.requires_grad = False
  model.fc = nn.Linear(num_ft, 2)

  optimizer = optim.Adam(model.fc.parameters(), lr = model.config.finetuning_lr) 

  #try to train
  for epoch in range(num_epochs):
      print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
      model.train()
      loss_total = 0
      for t, (x, y) in enumerate(model.config.test_loader_finetuning):
          x_var = Variable(x)
          y_var = Variable(y.type(model.config.dtype).long())
          scores = model(x_var) 
          loss = loss_fn(scores, y_var)
          
          loss_total += loss.data[0]
          optimizer.zero_grad()
          loss.backward()

          optimizer.step()

          if ((t+1) % 10) == 0:
            grad_magnitude = [(x.grad.data.sum(), torch.numel(x.grad.data)) for x in model.parameters() if x.grad.data.sum() != 0.0]
            grad_magnitude = sum([abs(x[0]) for x in grad_magnitude]) #/ sum([x[1] for x in grad_magnitude])
            print('t = %d, avg_loss = %.4f, grad_mag = %.2f' % (t + 1, loss_total / (t+1), grad_magnitude))
      print("--- Evaluating ---")
      check_accuracy(model, model.config.train_loader, type = "Finetuning training")
      check_accuracy(model, model.config.val_loader, type = "Finetuning held out")
  print("\n--- Final Evaluation after finetuning ---")
  check_accuracy(model, model.config.test_loader_finetuning, type = "Finetuning training")
  check_accuracy(model, model.config.test_loader_holdout, type = "Finetuning held out")
  """

########
# MAIN #
########

def main():
  # Config
  args = parseConfig()
  config = Config(args) 
  print(config)

  logger = Logger()
  print("Logging destination: ", logger)
  # Load Embeddings
  #vocab, embeddings, embedding_dim = load_word_vectors('.', 'glove.6B', 100)

  # Model
  model = ComplexAudioRNN_2(config, audio_dim = 34)
  model.apply(initialize_weights)
  if config.use_gpu:
    model = model.cuda()


  # Load Data
  #new getAudioDatasets util lets you specify subjects to hold out from training and val datasets. 
  #train_dataset = AudioDataset(config)
  train_dataset, test_dataset = getAudioDatasets(config,hold_out={15})
  train_idx, val_idx = splitIndices(train_dataset, config.nt, config.nv, shuffle = True)
  test_finetuning_idx, test_holdout_idx = splitIndices(test_dataset,4*len(test_dataset)/5, shuffle = True)

  train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)
  test_finetuning_sampler, test_holdout_sampler = SubsetRandomSampler(test_finetuning_idx), SubsetRandomSampler(test_holdout_idx)

  train_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 3, sampler = train_sampler)
  val_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 1, sampler = val_sampler)
  test_loader_finetuning = DataLoader(test_dataset, batch_size = config.batch_size/2, num_workers = 1, sampler = test_finetuning_sampler)
  test_loader_holdout = DataLoader(test_dataset, batch_size = config.batch_size/2, num_workers = 1, sampler = test_holdout_sampler)
  test_loader_all = DataLoader(test_dataset, batch_size=config.batch_size)

  train_dataset.printDistributions(train_idx, msg = "Training", logger= logger)
  train_dataset.printDistributions(val_idx, msg = "Val",  logger= logger)
  test_dataset.printDistributions(range(len(test_dataset)), msg="Test",  logger= logger)

  config.train_loader = train_loader
  config.val_loader = val_loader
  config.test_loader_all = test_loader_all
  config.test_loader_finetuning = test_loader_finetuning
  config.test_loader_holdout = test_loader_holdout

  optimizer = optim.Adam(model.parameters(), lr = config.lr) 
  loss_fn = nn.CrossEntropyLoss().type(config.dtype)
  best_model = train(model, loss_fn, optimizer, config.epochs, logger = logger, hold_out = -1)

  #test on the held out speaker
  eval_on_test_set(best_model, loss_fn, config.finetuning_epochs, logger = logger, hold_out = -1)


  

if __name__ == '__main__':
  main()