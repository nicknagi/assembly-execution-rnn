from datetime import datetime
import torch
from torch.utils.data.dataloader import DataLoader
from torch import optim
import numpy as np
import torch.nn as nn
import os
import random
from torch.utils.data.distributed import DistributedSampler


from earlystopping import EarlyStopping
from tqdm import trange

# if torch.cuda.is_available():
#     device = "cuda:0"
# else:
#     device = "cpu"

device = "cpu"


def train_model(model, train_dataset, batch_size, n_epochs, target_len, validation_dataset, is_distributed,
                training_prediction='recursive',
                teacher_forcing_ratio=0.5, learning_rate=0.01, dynamic_tf=False):
    # initialize array of losses
    losses = np.full(n_epochs, np.nan)
    val_losses = []
    granular_loss = []

    sampler = DistributedSampler(train_dataset) if is_distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=(sampler is None),
        pin_memory=True, num_workers=0, sampler=sampler)

    optimizer = optim.SGD([{"params": model.encoder.parameters()}, {"params": model.decoder.parameters()}],
                          lr=learning_rate, momentum=0.9)
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
            if is_distributed:
                sampler.set_epoch(it)
            batches = 0
            batch_loss = 0.

            for i, data in enumerate(train_loader):
                # select data
                code, target = data
                code = code.float()

                code = code.to(device)
                target = target.to(device)

                # outputs tensor
                outputs = torch.zeros(target_len, batch_size, model.input_size)

                # # initialize hidden state - not needed for now
                # encoder_hidden = self.encoder.init_hidden(batch_size)

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output, encoder_hidden = model.encoder(code)

                # decoder with teacher forcing
                decoder_input = torch.zeros((batch_size, model.input_size)).to(device)
                decoder_hidden = encoder_hidden

                if training_prediction == 'recursive':
                    # predict recursively
                    for t in range(target_len):
                        decoder_output, decoder_hidden = model.decoder(
                            decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = decoder_output

                elif training_prediction == 'teacher_forcing':
                    # use teacher forcing
                    if random.random() < teacher_forcing_ratio:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = model.decoder(
                                decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = target[:, t, :].float()

                    # predict recursively
                    else:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = model.decoder(
                                decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                elif training_prediction == 'mixed_teacher_forcing':
                    # predict using mixed teacher forcing
                    for t in range(target_len):
                        decoder_output, decoder_hidden = model.decoder(
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
                    output_argmax = outputs[:, i, :]
                    target_argmax = torch.argmax(target[:, i, :], dim=1).long()
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

            validation_loss = calculate_loss(model, validation_dataset)[1]
            val_losses.append(validation_loss)

            # progress bar
            tr.set_description(f"loss: {batch_loss} val_loss: {validation_loss}")

            early_stopping(validation_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(best_model_path))

    return losses, val_losses


@torch.no_grad()
def calculate_loss(model, dataset):
    data_loader = DataLoader(dataset, batch_size=128, drop_last=True, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    model.eval()  # Change model to eval mode

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
        outputs = torch.zeros(target_len, batch_size, model.input_size)

        # initialize hidden state
        encoder_output, encoder_hidden = model.encoder(input_tensor)

        # decode input_tensor
        decoder_input = torch.zeros(batch_size, model.input_size).to(device)
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = model.decoder(
                decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output

        outputs = torch.transpose(outputs, 0, 1).to(device)

        batch_loss = 0
        for i in range(target_len):
            output_argmax = outputs[:, i, :]
            target_argmax = torch.argmax(target_tensor[:, i, :], dim=1).long().to(device)
            batch_loss += criterion(output_argmax, target_argmax)

        loss += batch_loss.item()
        results.append(outputs.detach())

    loss /= batches
    result = torch.stack(results)

    model.train()  # Change model to train mode

    return result, loss
