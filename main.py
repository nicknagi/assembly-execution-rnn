from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
from model import lstm_seq2seq
import time
from training import train_model
from dataset import *

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

torch.set_printoptions(precision=10)
torch.set_printoptions(edgeitems=100)

print(f"Running on {device}")


def run_training_for_model(model, num_instrs, possible_instructions=None, data_factor=1):
    if possible_instructions is None:
        possible_instructions = ["ADD", "SUB", "MOV"]
    train_x, train_y = create_dataset(num_instrs, num_samples=10000 * num_instrs * data_factor,
                                      possible_instructions=possible_instructions)
    val_x, val_y = create_dataset(num_instrs, num_samples=1000 * num_instrs * data_factor,
                                  possible_instructions=possible_instructions)

    train_dataset = TensorDataset(train_x, train_y)
    validation_dataset = TensorDataset(val_x, val_y)

    train_loss, val_loss = train_model(model, train_dataset=train_dataset, batch_size=128, n_epochs=4,
                                       target_len=train_y.size()[1],
                                       validation_dataset=validation_dataset,
                                       training_prediction="teacher_forcing", learning_rate=0.01)

    return train_loss, val_loss


if __name__ == "__main__":
    model = lstm_seq2seq(len(all_chars), 256)
    model = model.to(device)

    NUM_INSTRS = 4

    # for num_instrs in range(1,NUM_INSTRS+1):
    #     print(f"\n\nStarting training for {num_instrs}")
    #     training_loss, validation_loss = run_training_for_model(model, num_instrs, possible_instructions=["ADD"])
    #
    #     plt.plot(training_loss, label=f"training loss {num_instrs} 1")
    #     plt.plot(validation_loss, label=f"validation loss {num_instrs} 1")
    #
    # for num_instrs in range(1,NUM_INSTRS+1):
    #     print(f"\n\nStarting training for {num_instrs}")
    #     training_loss, validation_loss = run_training_for_model(model, num_instrs, possible_instructions=["ADD", "SUB"], data_factor=2)
    #
    #     plt.plot(training_loss, label=f"training loss {num_instrs} 2")
    #     plt.plot(validation_loss, label=f"validation loss {num_instrs} 2")

    # for num_instrs in range(1,NUM_INSTRS+1):
    #     print(f"\n\nStarting training for {num_instrs}")
    #     training_loss, validation_loss = run_training_for_model(model, num_instrs, possible_instructions=["ADD", "SUB", "MOV"], data_factor=3)
    #
    #     plt.plot(training_loss, label=f"training loss {num_instrs} 3")
    #     plt.plot(validation_loss, label=f"validation loss {num_instrs} 3")
    #
    # plt.legend(loc="upper left")
    # plt.savefig(f"results_{time.time()}.png")
    # plt.show()

    training_loss, validation_loss = run_training_for_model(model, 2,
                                                            possible_instructions=["ADD", "SUB", "MOV"], data_factor=1)
    plt.plot(training_loss, label=f"training loss")
    plt.plot(validation_loss, label=f"validation loss")

    plt.legend(loc="upper left")
    plt.savefig(f"results_{time.time()}.png")
    plt.show()
