import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import trange
from model import lstm_seq2seq
import numpy as np

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

torch.set_printoptions(precision=10)
torch.set_printoptions(edgeitems=100)

print(f"Running on {device}")

# Simple assembly processor


def execute_assembly(instructions):
    registers = [0, 0, 0, 0, 0, 0]  # R0, R1, R2, R3, R4, R5
    for instruction in instructions:
        instr_type = instruction.split(" ")[0]
        source = instruction.split(" ")[1]

        # Read source register value if needed
        if source.startswith("R"):
            source = registers[int(source[-1])]

        dest_reg = int(instruction.split(" ")[2][-1])

        if instr_type == "ADD":
            registers[dest_reg] += int(source)
        elif instr_type == "SUB":
            registers[dest_reg] -= int(source)
        elif instr_type == "MOV":
            registers[dest_reg] = int(source)
    return registers

# Generate a random assembly instruction


def generate_assembly_instruction():
    instr = random.choice(["ADD", "SUB", "MOV"])
    value = random.randint(10, 99)
    destination_reg = "R" + str(random.randint(1, 5))
    source_reg = "R" + str(random.randint(0, 5))

    source_reg_or_val = random.choice([str(value), source_reg])
    return instr + " " + source_reg_or_val + " " + destination_reg


all_chars = ["A", "D", "S", "U", "B", "M", "O", "V", "1", "2",
             "3", "4", "5", "6", "7", "8", "9", "0", "R", " ", "~"]


def s_to_i(s):
    instr_num = []
    for char in s:
        instr_num.append(all_chars.index(char))
    return instr_num


def create_dataset(num_samples=10000):
    x, y = [], []
    for _ in trange(num_samples):
        instructions = [generate_assembly_instruction()
                        for _ in range(random.randint(1, 10))]

        result = execute_assembly(instructions)
        result = torch.tensor(result, dtype=torch.float)

        instructions = "~".join(instructions) + "~"
        instructions = s_to_i(instructions)
        instructions = torch.nn.functional.one_hot(torch.tensor(instructions))

        x.append(instructions)
        y.append(result)

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.stack(y)
    return x, y


train_x, train_y = create_dataset(num_samples=40000)
val_x, val_y = create_dataset(num_samples=1000)

train_dataset = TensorDataset(train_x, train_y)
validation_dataset = TensorDataset(val_x, val_y)

model = lstm_seq2seq(21, 128)
model = model.to(device)
training_loss, validation_loss = model.train_model(train_dataset=train_dataset, batch_size=64, n_epochs=5, target_len=6, validation_dataset=validation_dataset,
                                                   training_prediction="recursive")

plt.plot(training_loss)
plt.plot(validation_loss)
plt.show()
