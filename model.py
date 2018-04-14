import numpy as np
import torch
import torch.nn.functional as F

from utils import get_sequences_lengths, variable, argmax


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, padding_idx, init_idx, max_len, teacher_forcing):
        """
        Sequence-to-sequence model
        :param vocab_size: the size of the vocabulary
        :param embedding_dim: Dimension of the embeddings
        :param hidden_size: The size of the encoder and the decoder
        :param padding_idx: Index of the special pad token
        :param init_idx: Index of the <s> token
        :param max_len: Maximum length of a sentence in tokens
        :param teacher_forcing: Probability of teacher forcing
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.init_idx = init_idx
        self.max_len = max_len
        self.teacher_forcing = teacher_forcing

        ##############################
        ### Insert your code below ###
        ##############################

        self.vocab_size = vocab_size

        # output embedding layer 
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, self.padding_idx)

        # decoder 
        self.encoder = torch.nn.LSTM(self.embedding_dim, self.hidden_size, 1)

        # encoder 
        self.decoder = torch.nn.LSTMCell(self.embedding_dim, self.hidden_size)

        # projection layer 
        self.project = torch.nn.Linear(self.embedding_dim, self.vocab_size)


        
        ###############################
        ### Insert your code above ####
        ###############################


    def zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tuple of two tensors (h and c) of zeros of the shape of (batch_size x hidden_size)
        """

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = 1
        state_shape = (nb_layers, batch_size, self.hidden_size)

        ##############################
        ### Insert your code below ###
        ##############################
        #weight = next(self.parameters()).data
        #h0 = Variable(weight.new(nb_layers, batch_size, self.hidden_size).zero_())
        h0 = variable(torch.zeros(state_shape))
        #c0 = Variable(weight.new(nb_layers, batch_size, self.hidden_size).zero_())
        c0 = variable(torch.zeros(state_shape))


        ###############################
        ### Insert your code above ####
        ###############################

        return h0, c0

    def encode_sentence(self, inputs):
        """
        Encode input sentences input a batch of hidden vectors z
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x hidden_size)
        """

        batch_size = inputs.size(0)

        ##############################
        ### Insert your code below ###
        ##############################

        # zero_state, length(input) check utils, embedding layer to encode to embeddings, 
        # sort to lengths & pass to packed_seq(sorts by length) 
        # packed_seq_obj returns unsorted data, sort it back into orioginal order and pass to decoder

        embeddings = self.embedding(inputs)
        input_lengths = get_sequences_lengths(embeddings)

        #sort the embeddings before packing?


        packed_seq = nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, batch_first=True)
        encoder_output, z = self.encoder(packed_seq)

        # reshape output
        z = nn.utils.rnn.pack_padded_sequence(encoder_output, get_sequences_lengths(encoder_output), batch_first=True)

        ###############################
        ### Insert your code above ####
        ###############################

        return z

    def decoder_state(self, z):
        """
        Create initial hidden state for the decoder based on the hidden vectors z
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tuple of two tensors (h and c) of size (batch_size x hidden_size)
        """

        batch_size = z.size(0)

        state_shape = (batch_size, self.hidden_size)
        ##############################
        ### Insert your code below ###
        ##############################

        _, c0 = self.zero_state(state_shape)

        ###############################
        ### Insert your code above ####
        ###############################

        return z, c0

    def decoder_initial_inputs(self, batch_size):
        """
        Create initial input the decoder on the first timestep
        :param inputs: The size of the batch
        :return: A vector of size (batch_size, ) filled with the index of self.init_idx
        """

        ##############################
        ### Insert your code below ###
        ##############################

        # return a vector filled with indices

        # not going to work: this should be indices

        inputs = variable(torch.rand(batch_size))

        #raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################
        return inputs

    def decode_sentence(self, z, targets=None):
        """
        Decode the tranlation of the input sentences based in the hidden vectors z and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        batch_size = z.size(0)

        ##############################
        ### Insert your code below ###
        ##############################
        init_state,init_cell  = self.decoder_state(z)
        init_inputs = self.decoder_initial_inputs(batch_size)

        logits = self.embedding(z)
        #logits = F.relu(logits)

        outputs, hidden = self.decoder(logits)

        outputs = self.project(outputs)

        ###############################
        ### Insert your code above ####
        ###############################

        return outputs

    def forward(self, inputs, targets=None):
        """
        Perform the forward pass of the network and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        if self.training and np.random.rand() < self.teacher_forcing:
            targets = inputs
        else:
            targets = None

        z = self.encode_sentence(inputs)
        outputs = self.decode_sentence(z, targets)
        return outputs
