import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

# Simple assembly processor
def execute_assembly(instructions):
    registers = [0,0,0,0,0,0] # R0, R1, R2, R3, R4, R5
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
    return sum(registers)

# Generate a random assembly instruction
def generate_assembly_instruction():
    instr = random.choice(["ADD", "SUB", "MOV"])
    value = random.randint(10, 99)
    destination_reg = "R" + str(random.randint(1,5))
    source_reg = "R" + str(random.randint(0,5))
    
    source_reg_or_val = random.choice([str(value), source_reg])
    return instr + " " + source_reg_or_val + " " + destination_reg

all_chars = ["A", "D", "S", "U", "B", "M", "O", "V", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "R", " ", "~"]

def s_to_i(s):
    instr_num = []
    for char in s:
        instr_num.append(all_chars.index(char))
    return instr_num

x,y = [], []
for _ in range(1000):
    instructions = [generate_assembly_instruction() for _ in range(20)]

    result = execute_assembly(instructions)
    result = torch.tensor(result, dtype=torch.float)

    instructions = "~".join(instructions)
    instructions = s_to_i(instructions)
    instructions = torch.nn.functional.one_hot(torch.tensor(instructions))

    x.append(instructions)
    y.append(result)

dataset = TensorDataset(torch.stack(x), torch.stack(y))
data_loader = DataLoader(dataset, batch_size=32, drop_last=True)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.name = "LSTM"
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_data, hidden):
        input_data = input_data.float()
        lstm_output, _ = self.lstm(input_data, hidden)
        out = self.fc(lstm_output[:, -1, :])
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_())
        return hidden

def train_model(model, training_loader, num_epochs=5, learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    for epoch in range(num_epochs):
        h = model.init_hidden(batch_size = 32)
        for instructions, expected_output in training_loader:
            h = tuple([e.data for e in h])
            optimizer.zero_grad()
            pred = model(instructions, h)
            expected_output = expected_output.unsqueeze(dim=1)
            loss = criterion(pred, expected_output)
            loss.backward()
            optimizer.step()
            train_loss.append(float(loss))

    plt.plot(train_loss)
    plt.show()

model = LSTM(21, 256)
train_model(model, data_loader)