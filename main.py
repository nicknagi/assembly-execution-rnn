import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import trange

if torch.cuda.is_available():  
  device = "cuda:0" 
else:  
  device = "cpu" 

print(f"Running on {device}")

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
for _ in range(10000):
    instructions = [generate_assembly_instruction() for _ in range(100)]

    result = execute_assembly(instructions)
    result = torch.tensor(result, dtype=torch.float)

    instructions = "~".join(instructions)
    instructions = s_to_i(instructions)
    instructions = torch.nn.functional.one_hot(torch.tensor(instructions))

    x.append(instructions)
    y.append(result)

dataset = TensorDataset(torch.stack(x), torch.stack(y))
data_loader = DataLoader(dataset, batch_size=64, drop_last=True)

class rnn(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.name = "rnn"
        super(rnn, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = 1
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_data):
        rnn_output, _ = self.rnn(input_data)
        out = self.fc(rnn_output[:, -1, :])
        return out

def train_model(model, training_loader, num_epochs=5, learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    for epoch in trange(num_epochs):
        for batch, data in enumerate(training_loader):
            instructions, expected_output = data
            instructions = instructions.float()

            instructions = instructions.to(device)
            expected_output = expected_output.to(device)
            
            optimizer.zero_grad()
            pred = model(instructions)
            expected_output = expected_output.unsqueeze(dim=1)
            loss = criterion(pred, expected_output)
            loss.backward()
            optimizer.step()
            train_loss.append(float(loss))

            if batch == 3:
                torch.set_printoptions(precision=10)
                torch.set_printoptions(edgeitems=100)
                print(loss)
                print(pred)
                print(expected_output)

    plt.plot(train_loss)
    plt.show()

model = rnn(21, 256)
model = model.to(device)
train_model(model, data_loader, num_epochs=5, learning_rate=1e-1)