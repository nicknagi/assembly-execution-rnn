import random
import torch
from tqdm import trange

all_chars = ["A", "D", "S", "U", "B", "M", "O", "V", "1", "2",
             "3", "4", "5", "6", "7", "8", "9", "0", "R", " ", "-", "~"]


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
def generate_assembly_instruction(possible_instructions):
    instr = random.choice(possible_instructions)
    value = random.randint(1, 9)
    destination_reg = "R" + str(random.randint(1, 5))
    source_reg = "R" + str(random.randint(0, 5))

    source_reg_or_val = random.choice([str(value), source_reg])
    return instr + " " + source_reg_or_val + " " + destination_reg


# Convert string to numerical list based on char mapping
def s_to_i(s):
    instr_num = []
    for char in s:
        instr_num.append(all_chars.index(char))
    return instr_num


# Given a list of register values convert to one hot encoding
def convert_registers_to_one_hot(registers_list):
    result_string = ""
    for register in registers_list:
        result_string += str(register)
        result_string += " "
    result_string += "~"
    result_numerical = s_to_i(result_string)
    result = torch.nn.functional.one_hot(torch.tensor(result_numerical))
    return result


# Create a dataset
def create_dataset(num_instrs, possible_instructions, num_samples=10000):
    x, y = [], []
    for _ in trange(num_samples):
        legal = False
        result = None
        while not legal:
            instructions = [generate_assembly_instruction(possible_instructions)
                            for _ in range(num_instrs)]
            result = execute_assembly(instructions)
            legal = sum(result) != 0

        target = convert_registers_to_one_hot(result)

        instructions = "~".join(instructions) + "~"
        instructions = s_to_i(instructions)
        instructions = torch.nn.functional.one_hot(torch.tensor(instructions))

        x.append(instructions)
        y.append(target)

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return x, y
