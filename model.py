import torch
import torch.nn as nn
import os
import torch.distributed as dist

# if torch.cuda.is_available():
#     device = "cuda:0"
# else:
#     device = "cpu"

device = "cpu"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Model adapted from https://github.com/lkulowski/LSTM_encoder_decoder
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

            char_index = torch.multinomial(softmax(decoder_output / temperature), 1)[0].cpu().numpy()[0]
            char_pred = character_map[char_index]
            pred += char_pred

            if char_pred == termination_char:
                break

        return pred
