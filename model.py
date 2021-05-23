import numpy as np
import random
from torch.utils.data import dataset
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
import os

from earlystopping import EarlyStopping


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class lstm_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.num_layers, self.hidden_size).to(device),
                torch.zeros(batch_size, self.num_layers, batch_size, self.hidden_size).to(device))


class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(
            x_input.unsqueeze(0), encoder_hidden_states)
        
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class lstm_seq2seq(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(
            input_size=input_size, hidden_size=hidden_size, num_layers=2)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size, num_layers=2)

    def train_model(self, train_dataset, batch_size, n_epochs, target_len, validation_dataset, training_prediction='recursive',
                    teacher_forcing_ratio=0.5, learning_rate=0.01, dynamic_tf=False):

        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        val_losses = []
        granular_loss = []

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()


        # Make checkpointing directory
        now = datetime.now()
        dir_name = now.strftime("%d %B %H:%M:%S")
        os.mkdir(f"models/{dir_name}")
        best_model_path = f"models/{dir_name}/best_model.pt"

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=20, verbose=False, path=best_model_path)

        with trange(n_epochs) as tr:
            for it in tr:
                batches = 0
                batch_loss = 0.

                for i, data in enumerate(train_loader):
                    # select data
                    code, target = data
                    code = code.float()

                    code = code.to(device)
                    target = target.to(device)

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, self.input_size)

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(code)

                    # decoder with teacher forcing
                    decoder_input = torch.zeros((batch_size, self.input_size)).to(device)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(
                                decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    elif training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(
                                    decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target[:, t, :].float()

                        # predict recursively
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(
                                    decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    elif training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(
                                decoder_input, decoder_hidden)
                            outputs[t] = decoder_output

                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target[:, t, :].float()

                            # predict recursively
                            else:
                                decoder_input = decoder_output
                    batches += 1

                    outputs = torch.transpose(
                        outputs.squeeze(), 0, 1).to(device)

                    # if i == 5:
                    #     print(outputs[0], target[0])

                    # compute the loss
                    loss = 0
                    for i in range(target_len):
                        output_argmax = outputs[:,i,:]
                        target_argmax = torch.argmax(target[:,i,:], dim=1).long()
                        loss += criterion(output_argmax, target_argmax)

                    batch_loss += loss.item()
                    granular_loss.append(loss.item())

                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # # loss for epoch
                batch_loss /= batches
                losses[it] = batch_loss
                # losses[it] = self.calculate_loss(train_dataset, self.input_size)[1]  # TEMP DELETE LATER

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

                validation_loss = self.calculate_loss(validation_dataset)[1]
                val_losses.append(validation_loss)

                # progress bar
                tr.set_description(f"loss: {batch_loss} val_loss: {validation_loss}")

                early_stopping(validation_loss, self)
                if early_stopping.early_stop:
                    print("Early Stopping")
                    break

        # load the last checkpoint with the best model
        self.load_state_dict(torch.load(best_model_path))

        return losses, val_losses

    @torch.no_grad()
    def calculate_loss(self, dataset):
        data_loader = DataLoader(dataset, batch_size=32, drop_last=True)
        criterion = nn.CrossEntropyLoss()

        self.eval() # Change model to eval mode

        batches = 0
        loss = 0
        results = []
        for i, data in enumerate(data_loader):
            batches += 1
            input_tensor, target_tensor = data

            target_len = target_tensor.size()[1]

            # encode input_tensor
            input_tensor = input_tensor.float()
            input_tensor = input_tensor.to(device)

            batch_size = input_tensor.size()[0]

            # initialize tensor for predictions
            outputs = torch.zeros(target_len, batch_size, self.input_size)

            # initialize hidden state
            encoder_output, encoder_hidden = self.encoder(input_tensor)

            # decode input_tensor
            decoder_input = torch.zeros(batch_size, self.input_size).to(device)
            decoder_hidden = encoder_hidden

            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output

            outputs = torch.transpose(outputs, 0, 1).to(device)

            batch_loss = 0
            for i in range(target_len):
                output_argmax = outputs[:,i,:]
                target_argmax = torch.argmax(target_tensor[:,i,:], dim=1).long().to(device)
                batch_loss += criterion(output_argmax, target_argmax)

            loss += batch_loss.item()
            results.append(outputs.detach())

        loss /= batches
        result = torch.stack(results)

        self.train() # Change model to train mode

        return result, loss
    
    # Make sure to change model to eval mode before calling this function
    @torch.no_grad()
    def predict(self, input_tensor, character_map, termination_char, temperature):
        from torch.distributions import Categorical
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.float().to(device)

        # initialize hidden state
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # decode input_tensor
        decoder_input = torch.zeros(1, self.input_size).to(device)
        decoder_hidden = encoder_hidden

        softmax = nn.Softmax(dim=-1)

        pred = ""

        for _ in range(20):
            decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
            decoder_input = decoder_output

            char_index = torch.multinomial(softmax(decoder_output/temperature), 1)[0].cpu().numpy()[0]
            char_pred = character_map[char_index]
            pred += char_pred

            if char_pred == termination_char:
                break

        return pred
            

