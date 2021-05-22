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
             "3", "4", "5", "6", "7", "8", "9", "0", "R", " ", "-", "~"]


def s_to_i(s):
    instr_num = []
    for char in s:
        instr_num.append(all_chars.index(char))
    return instr_num

def convert_registers_to_one_hot(registers_list):
    result_string = ""
    for register in registers_list:
        result_string += str(register)
        result_string += " "
    result_string += "~"
    result_numerical = s_to_i(result_string)
    result = torch.nn.functional.one_hot(torch.tensor(result_numerical))
    return result

def create_dataset(num_samples=10000):
    x, y = [], []
    for _ in trange(num_samples):
        legal = False
        result = None
        while not legal:
            instructions = [generate_assembly_instruction()
                            for _ in range(random.randint(1, 10))]

            result = execute_assembly(instructions)
            legal = result[random.randint(1,5)] != 0 and result[random.randint(1,5)] != 0
        
        target = convert_registers_to_one_hot(result)

        instructions = "~".join(instructions) + "~"
        instructions = s_to_i(instructions)
        instructions = torch.nn.functional.one_hot(torch.tensor(instructions))

        x.append(instructions)
        y.append(target)

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return x, y


train_x, train_y = create_dataset(num_samples=100000)
val_x, val_y = create_dataset(num_samples=1000)

train_dataset = TensorDataset(train_x, train_y)
validation_dataset = TensorDataset(val_x, val_y)

model = lstm_seq2seq(len(all_chars), 512)
model = model.to(device)

init_validation_loss = model.calculate_loss(validation_dataset, val_y.size()[1])[1]

training_loss, validation_loss = model.train_model(train_dataset=train_dataset, batch_size=128, n_epochs=10, target_len=train_y.size()[1],
 validation_dataset=validation_dataset, training_prediction="teacher_forcing", learning_rate=0.005)

validation_loss.insert(0, init_validation_loss)
plt.plot(validation_loss)
plt.plot(training_loss)
plt.show()

# ------------------ MANUAL TESTING ---------------------

instrs = ["ADD 2 R2"]
expected = execute_assembly(instrs)

instructions = "~".join(instrs) + "~"
instructions = s_to_i(instructions)
instructions_tensor = torch.nn.functional.one_hot(torch.tensor(instructions))

pred = model.predict(instructions_tensor, all_chars, "~", temperature=0.5)
print(f"Expected: {expected}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=0.8)
print(f"Expected: {expected}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=1.01)
print(f"Expected: {expected}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=1.1)
print(f"Expected: {expected}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=1.5)
print(f"Expected: {expected}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=1.75)
print(f"Expected: {expected}, Prediction: {pred}")

'''
# TODO: Try multinomial sampling for making predictions
# TODO: Instead of assembly try text generation using some random dataset -- this seems the best!
'''
