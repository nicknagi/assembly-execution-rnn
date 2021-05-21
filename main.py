import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

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
for _ in range(100):
    instructions = [generate_assembly_instruction() for _ in range(20)]

    result = execute_assembly(instructions)
    result = torch.tensor(result)

    instructions = "~".join(instructions)
    instructions = s_to_i(instructions)
    instructions = torch.nn.functional.one_hot(torch.tensor(instructions))

    x.append(instructions)
    y.append(result)

dataset = TensorDataset(torch.stack(x), torch.stack(y))
data_loader = DataLoader(dataset, batch_size=32)