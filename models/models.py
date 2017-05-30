import torch
from torch import nn
from torch.nn import init

############
## LAYERS ##
############

def initialize_weights(m):
  if isinstance(m, nn.Linear): #or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
    init.xavier_uniform(m.weight.data)

class Flatten(nn.Module):
  def forward(self, x):
      N = x.size(0) # read in N, C, H, W
      return x.contiguous().view(N, -1)  # "flatten" the C * H * W values into a single vector per image

#######################
#  AUDIO ONLY MODELS  #
#######################

class SimpleAudioRNN(nn.Module):
  def __init__(self, config):
    super(SimpleAudioRNN, self).__init__()
    self.config = config
    self.rnn = nn.LSTM(34, config.hidden_size, batch_first = True)
    self.decoder =  nn.Linear(config.hidden_size, config.num_classes)

  def forward(self, input):
    seq_output, hidden = self.rnn(input)
    hidden_state, cell_state = hidden
    decoded = self.decoder(hidden_state.squeeze(0))
    #output = pad_packed_sequence(output, batch_first = True)
    return decoded 


#### ARCHIVED: ComplexAudioRNN_1

class ComplexAudioRNN_1(nn.Module):
  def __init__(self, config, feature_size = 34):
    super(ComplexAudioRNN_1, self).__init__()
    # Do not change model, copy and paste into new class.
    self.config = config
    self.rnn = nn.LSTM(feature_size, config.hidden_size, batch_first = True)
    self.flat_dim = config.max_length * config.hidden_size
    self.decoder =  nn.Sequential(
        Flatten(),
        nn.Linear(self.flat_dim, config.hidden_size*2),
        nn.BatchNorm1d(config.hidden_size*2),
        nn.ReLU(),
        nn.Linear(config.hidden_size*2, config.num_classes)
      )

  def forward(self, input):
    seq_output, hidden = self.rnn(input)
    #hidden_state, cell_state = hidden
    decoded = self.decoder(seq_output)
    #output = pad_packed_sequence(output, batch_first = True)
    return decoded 

#### UNDER_DEVELOPMENT: ComplexAudioRNN_2

class ComplexAudioRNN_2(nn.Module):
  def __init__(self, config, feature_size = 34):
    super(ComplexAudioRNN_2, self).__init__()
    self.config = config
    self.rnn = nn.LSTM(feature_size, config.hidden_size, batch_first = True)
    self.flat_dim = config.max_length * config.hidden_size
    self.decoder =  nn.Sequential(
        Flatten(),
        nn.Linear(self.flat_dim, self.flat_dim // 2),
        nn.BatchNorm1d(self.flat_dim // 2),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(self.flat_dim // 2, config.hidden_size*2),
        nn.BatchNorm1d(config.hidden_size*2),
        nn.ReLU(),
        nn.Linear(config.hidden_size*2, config.num_classes)
      )

  def forward(self, input):
    seq_output, hidden = self.rnn(input)
    #hidden_state, cell_state = hidden
    decoded = self.decoder(seq_output)
    #output = pad_packed_sequence(output, batch_first = True)
    return decoded 


###################
#  HYBRID MODELS  #
##################

class RNNContextEvaluator(nn.Module):
  def __init__(self, config, embeddings, vocab):
    super(RNNContextEvaluator, self).__init__()
    self.config = config
    self.vocab = vocab
    self.n_words, self.hidden_size = embeddings.size()

    #self.encoder = nn.Embedding(self.n_words, self.hidden_size, 0)
    #self.encoder.weight = nn.Parameter(embeddings)
    
    self.rnn_audio = nn.LSTM(self.hidden_size, self.hidden_size, batch_first = True)
    #self.rnn_target = nn.LSTM(self.hidden_size, self.hidden_size, batch_first = True)
    self.decoder =  nn.Linear(self.hidden_size, config.num_classes)

  def forward(self, input_src, input_target):
    embedded_src = self.encoder(input_src)
    embedded_target = self.encoder(input_target)
    #embedded = pack_padded_sequence(embedded, batch_first = True)
    _, hidden_src = self.rnn_audio(embedded_src)
    _, hidden_target = self.rnn_target(embedded_target)

    hidden_state_src, _ = hidden_src
    hidden_state_target, _ = hidden_target

    activations = torch.cat((hidden_state_src.squeeze(0), hidden_state_target.squeeze(0)), 1)
    decoded = self.decoder(activations)
    #output = pad_packed_sequence(output, batch_first = True)
    return decoded 

class RNNEvaluator(nn.Module):
  def __init__(self, config, embeddings, vocab):
    super(RNNEvaluator, self).__init__()
    self.config = config
    self.vocab = vocab
    self.n_words, self.hidden_size = embeddings.size()

    self.encoder = nn.Embedding(self.n_words, self.hidden_size, 0)
    self.encoder.weight = nn.Parameter(embeddings)
    
    self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first = True)
    self.decoder =  nn.Linear(self.hidden_size, config.num_classes)

  def forward(self, input):
    embedded = self.encoder(input)
    #embedded = pack_padded_sequence(embedded, batch_first = True)
    seq_output, hidden = self.rnn(embedded)
    hidden_state, cell_state = hidden
    decoded = self.decoder(hidden_state.squeeze(0))
    #output = pad_packed_sequence(output, batch_first = True)
    return decoded 




