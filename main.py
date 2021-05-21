import random

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
    value = random.randint(0, 1000)
    destination_reg = "R" + str(random.randint(1,5))
    source_reg = "R" + str(random.randint(0,5))
    
    source_reg_or_val = random.choice([str(value), source_reg])
    return instr + " " + source_reg_or_val + " " + destination_reg

x,y = [], []
for _ in range(10000):
    instructions = [generate_assembly_instruction() for _ in range(random.randint(3, 25))]
    result = execute_assembly(instructions)

    x.append(instructions)
    y.append(result)

instructions = " <exec> ".join([generate_assembly_instruction() for _ in range(random.randint(3, 25))])
print(instructions)
