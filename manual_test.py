from main import execute_assembly, s_to_i, all_chars
import torch
from model import lstm_seq2seq

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

instrs = ["SUB 22 R2"]
expected = execute_assembly(instrs)

instructions = "~".join(instrs) + "~"
instructions = s_to_i(instructions)
instructions_tensor = torch.nn.functional.one_hot(torch.tensor(instructions))

model = lstm_seq2seq(len(all_chars), 512)
model.load_state_dict(torch.load("models/22 May 19:27/bs_128_epochs_99_lr_0.01_valloss_0.8033977196581902"))
model.to(device)

pred = model.predict(instructions_tensor, all_chars, "~", temperature=0.5)
print(f"Expected: {expected}, Prediction: {pred}")