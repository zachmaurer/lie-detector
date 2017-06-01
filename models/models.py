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
  def __init__(self, config, audio_dim = 34):
    super(ComplexAudioRNN_2, self).__init__()
    self.config = config
    self.rnn = nn.LSTM(audio_dim, config.hidden_size, batch_first = True)
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

class RNNHybrid_1(nn.Module):
  def __init__(self, config, embeddings, vocab, glove_dim = 100, audio_dim = 34):
    super(RNNHybrid_1, self).__init__()
    self.config = config
    self.vocab = vocab
    self.hidden_size = self.config.hidden_size

    # Embeddings
    self.n_words, self.hidden_size = embeddings.size()
    self.encoder = nn.Embedding(len(vocab), glove_dim, len(vocab)-1) # TODO FIX?
    self.encoder.weight = nn.Parameter(embeddings)
    
    # Encoders
    self.audio_rnn = nn.LSTM(audio_dim, self.hidden_size, batch_first = True)
    self.lex_rnn = nn.LSTM(glove_dim, self.hidden_size, batch_first = True)
    
    # Decoders
    self.flat_dim = config.max_length * self.hidden_size    
    self.audio_decoder = nn.Sequential(
                    Flatten(),
                    nn.Linear(self.flat_dim, 2*self.hidden_size),
                    nn.BatchNorm1d(2*self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(2*self.hidden_size, self.hidden_size)
            )

    self.lex_decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

    self.final_decoder = nn.Linear(self.hidden_size*2, config.num_classes)

  def forward(self, audio_input, lex_input):  
    # Lexical 
    embedded_lex = self.encoder(lex_input)
    #embedded = pack_padded_sequence(embedded, batch_first = True)
    lex_hidden_seq, lex_hidden = self.lex_rnn(embedded_lex)
    hidden_state_lex, _ = lex_hidden
    lex_activations = self.lex_decoder(hidden_state_lex.squeeze(0))

    # Audio
    audio_seq, audio_hidden = self.audio_rnn(audio_input)
    #hidden_state_audio, _ = audio_hidden
    audio_activation = self.audio_decoder(audio_seq)

    # Aggregation
    #print(lex_activations.size(), audio_activation.size())
    activations = torch.cat((lex_activations, audio_activation), 1)
    decoded = self.final_decoder(activations)
    #output = pad_packed_sequence(output, batch_first = True)
    return decoded 





