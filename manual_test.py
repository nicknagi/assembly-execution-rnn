from main import execute_assembly, s_to_i, all_chars
import torch
from model import lstm_seq2seq

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

instrs = ["ADD 56 R2", "MOV R2 R4", "MOV R4 R3"]
expected = execute_assembly(instrs)

expected_string = ""
for register in expected:
    expected_string += str(register)
    expected_string += " "
expected_string += "~"

instructions = "~".join(instrs) + "~"
instructions = s_to_i(instructions)
instructions_tensor = torch.nn.functional.one_hot(torch.tensor(instructions))

model = lstm_seq2seq(len(all_chars), 512)
model.load_state_dict(torch.load("models/incremental-1/23 May 09:39:32/bs_128_epochs_99_lr_0.01_valloss_0.5769427094248033"))
model.to(device)
model.eval()

pred = model.predict(instructions_tensor, all_chars, "~", temperature=0.001)
print(f"Expected: {expected_string}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=0.01)
print(f"Expected: {expected_string}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=0.2)
print(f"Expected: {expected_string}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=0.5)
print(f"Expected: {expected_string}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=0.8)
print(f"Expected: {expected_string}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=1)
print(f"Expected: {expected_string}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=1.5)
print(f"Expected: {expected_string}, Prediction: {pred}")

pred = model.predict(instructions_tensor, all_chars, "~", temperature=2.0)
print(f"Expected: {expected_string}, Prediction: {pred}")

'''
* Good Models:
1. models/22 May 19:27/bs_128_epochs_99_lr_0.01_valloss_0.8033977196581902 -- trained for 2 instructions only
2. models/22 May 21:25/bs_128_epochs_54_lr_0.01_valloss_2.0715185201937154 -- trained using incremental learning from 2 to 3
3. models/22 May 21:45/bs_128_epochs_54_lr_0.01_valloss_1.764981908182944 -- trained even more on 3 instructions using model 2
'''
