
# Python Imports
import argparse

# Torch Imports
import torch
#from torch import LongTensor
from torchtext.vocab import load_word_vectors
from torch.autograd import Variable
#from torch.nn.utils.rnn import pack_padded_sequence#, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch import cuda, FloatTensor
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler

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
    self.transcripts = args.transcripts
    self.max_length = args.length
    self.transcript_length = args.trans_len
    #self.eval_every = args.ee
    self.use_gpu = args.gpu
    self.dtype = cuda.FloatTensor if self.use_gpu else FloatTensor
    self.num_classes = 2

  def __str__(self):
    properties = vars(self)
    properties = ["{} : {}".format(k, str(v)) for k, v in properties.items()]
    properties = '\n'.join(properties)
    properties = "--- Config --- \n" + properties + "\n"
    return properties

def parseConfig(description="Hybrid text and audio RNN"):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--feats', type=str, help='input features path', default = "../data/features/extracted_features_0.1_0.05.json")
  parser.add_argument('--labels', type=str, help='input labels', default = "../data/features/labels_0.1_0.05.json")
  parser.add_argument('--transcripts', type=str, help='input labels', default = "../data/features/tokenized_transcripts.json")
  parser.add_argument('--length', type=int, help='length of audio sequence', default = 300)
  parser.add_argument('--trans_len', type=int, help='length of audio sequence', default = 230)
  parser.add_argument('--bs', type=int, help='batch size for training', default = 20)
  parser.add_argument('--e', type=int, help='number of epochs', default = 10)
  parser.add_argument('--nt', type=int, help='number of training examples', default = 100)
  parser.add_argument('--nv', type=int, help='number of validation examples', default = None)
  parser.add_argument('--hs', type=int, help='hidden size', default = 100)
  parser.add_argument('--lr', type=float, help='learning rate', default = 1e-3)
  parser.add_argument('--gpu', action='store_true', help='use gpu', default = False)
  parser.add_argument('--pe', type=int, help='print frequency', default = None)
  parser.add_argument('--ee', type=int, help='eval frequency', default = None)
  args = parser.parse_args()
  return args

############
# TRAINING #
############

def hybrid_train(model, loss_fn, optimizer, num_epochs = 1, logger = None, hold_out = -1):
  for epoch in range(num_epochs):
      print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
      model.train()
      loss_total = 0
      for t, (x, transcript, y, _) in enumerate(model.config.train_loader):
          x_var = Variable(x)
          transcript_var = Variable(transcript)
          y_var = Variable(y.type(model.config.dtype).long())
          scores = model(x_var, transcript_var) 
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
      check_accuracy(model, model.config.val_loader, type = "val")
      print("\n")
  print("\n--- Final Evaluation ---")
  check_accuracy(model, model.config.train_loader, type = "train", logger = logger, hold_out = hold_out)
  check_accuracy(model, model.config.val_loader, type = "val", logger = logger, hold_out = hold_out)
  #check_accuracy(model, model.config.test_loader, type = "test")


def check_accuracy(model, loader, type="", logger = None, hold_out = -1):
  print("Checking accuracy on {} set".format(type))
  num_correct = 0
  num_samples = 0
  examples, all_labels, all_predicted = [], [], []
  model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
  for t, (x, transcript, y, keys) in enumerate(loader):
      x_var = Variable(x)
      transcript_var = Variable(transcript)
      #y_var = Variable(y.type(model.config.dtype).long())
      scores = model(x_var, transcript_var)
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

def eval_on_test_set(model,loss_fn,num_epochs=1, logger = None, hold_out = -1):
  #first check the accuracy of the model on all of the data
  print("Trained model on all test data:")
  check_accuracy(model,model.config.test_loader_all,type="test", logger = logger, hold_out = hold_out)
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
  vocab, embeddings, embedding_dim = load_word_vectors('../data/glove', 'glove.6B', 100)

  # Model
  model = RNNHybrid_1(config, embeddings, vocab)
  
  # Weights Init
  model.apply(initialize_weights)
  if config.use_gpu:
    model = model.cuda()


  # Load Data
  train_dataset = HybridDataset(config, vocab)

  # Train-Val Split
  train_idx, val_idx = splitIndices(train_dataset, config, shuffle = True)
  train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)
  train_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 3, sampler = train_sampler)
  val_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 1, sampler = val_sampler)
  config.train_loader = train_loader
  config.val_loader = val_loader

  # Print Distributions
  train_dataset.printDistributions(train_idx, msg = "Training", logger= logger)
  train_dataset.printDistributions(val_idx, msg = "Val", logger= logger)

  # Train
  optimizer = optim.Adam(model.parameters(), lr = config.lr) 
  loss_fn = nn.CrossEntropyLoss().type(config.dtype)
  hybrid_train(model, loss_fn, optimizer, config.epochs, logger= logger)
  

if __name__ == '__main__':
  main()