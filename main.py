from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
from model import lstm_seq2seq
import time
from training import train_model
from dataset import *
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


using_gpu = False
if torch.cuda.is_available():
    device = "cuda:0"
    using_gpu = True
else:
    device = "cpu"

torch.set_printoptions(precision=10)
torch.set_printoptions(edgeitems=100)

print(f"Running on {device}")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(dist.Backend.GLOO, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_training_for_model(rank, world_size, num_instrs, possible_instructions=None, data_factor=1):
    setup(rank, world_size)

    model = lstm_seq2seq(len(all_chars), 256)
    model = model.to(device)
    device_ids = []
    if using_gpu:
        device_ids.append(0)
    model = DDP(model, device_ids=device_ids)  # Change if using GPU

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
    cleanup()

    return train_loss, val_loss


if __name__ == "__main__":

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

    # training_loss, validation_loss = run_training_for_model(model, 2,
    #                                                         possible_instructions=["ADD", "SUB", "MOV"], data_factor=1)
    world_size = 1
    mp.spawn(run_training_for_model,
             args=(world_size, NUM_INSTRS, ["ADD", "SUB", "MOV"], 1),
             nprocs=world_size,
             join=True)
