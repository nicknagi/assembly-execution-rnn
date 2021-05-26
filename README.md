# assembly-execution-rnn

A character level seq2seq model to execute simple assembly instructions. The model outputs the corresponding register values as its output.

## Example

Input: ["ADD 4 R2", "SUB 3 R2", "MOV R2 R4", "ADD 1 R4"]
Output: 0 0 1 0 2 0 

NoreL Each integer in the output is the corresponding register value. eg. 2 is R4

## How to run
Run `python3 manual_test.py` to see the output of the model for a given sequence of assembly instructions.
