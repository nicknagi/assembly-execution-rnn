import numpy as np
import random
from torch.utils.data import dataset
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''

        lstm_out, self.hidden = self.lstm(x_input)

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''

        return (torch.zeros(batch_size, self.num_layers, self.hidden_size).to(device),
                torch.zeros(batch_size, self.num_layers, batch_size, self.hidden_size).to(device))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 

        '''

        lstm_out, self.hidden = self.lstm(
            x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, hidden_size):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(
            input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=1, hidden_size=hidden_size)

    def train_model(self, train_dataset, batch_size, n_epochs, target_len, validation_dataset, training_prediction='recursive',
                    teacher_forcing_ratio=0.5, learning_rate=0.01, dynamic_tf=False):

        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        val_losses = []

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.L1Loss(reduction="sum")

        with trange(n_epochs) as tr:
            batches = 0
            for it in tr:

                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0

                for i, data in enumerate(train_loader):
                    # select data
                    code, target = data
                    code = code.float()

                    code = code.to(device)
                    target = target.to(device)

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, 1)

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(code)

                    # decoder with teacher forcing
                    decoder_input = torch.zeros((batch_size, 1)).to(device)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(
                                decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(
                                    decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = (target[:, t]).unsqueeze(-1)

                        # predict recursively
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(
                                    decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(
                                decoder_input, decoder_hidden)
                            outputs[t] = decoder_output

                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = (target[:, t]).unsqueeze(-1)

                            # predict recursively
                            else:
                                decoder_input = decoder_output
                    batches += 1

                    outputs = torch.transpose(
                        outputs.squeeze(), 0, 1).to(device)

                    # compute the loss
                    loss = criterion(outputs, target)

                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch
                batch_loss /= batches
                # losses[it] = batch_loss
                losses[it] = self.calculate_loss(train_dataset, 6)[
                    1]  # TEMP DELETE LATER

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

                # progress bar
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                val_losses.append(self.calculate_loss(
                    validation_dataset, 6)[1])

        return losses, val_losses

    @torch.no_grad()
    def calculate_loss(self, dataset, target_len):
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''

        train_loader = DataLoader(dataset, batch_size=32, drop_last=True)
        batches = 0
        loss = 0
        results = []
        for i, data in enumerate(train_loader):
            batches += 1
            input_tensor, target_tensor = data

            # encode input_tensor
            input_tensor = input_tensor.float()
            input_tensor = input_tensor.to(device)

            batch_size = input_tensor.size()[0]

            # initialize tensor for predictions
            outputs = torch.zeros(target_len, batch_size, 1)

            # initialize hidden state
            encoder_output, encoder_hidden = self.encoder(input_tensor)

            # decode input_tensor
            decoder_input = torch.zeros(batch_size, 1).to(device)
            decoder_hidden = encoder_hidden

            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output

            outputs = torch.transpose(outputs.squeeze(), 0, 1).to(device)
            outputs_np = outputs.cpu().numpy()
            target_np = target_tensor.numpy()
            loss += np.sum(np.absolute(target_np - outputs_np))
            results.append(outputs.detach())

        loss /= batches
        result = torch.stack(results)

        return result, loss
