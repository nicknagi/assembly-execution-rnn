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
import argparse
from config import FORCE_CPU, MASTER_IP, MASTER_PORT, BATCH_SIZE, NUM_EPOCHS, ENABLE_GRAPHS


using_gpu = False
if torch.cuda.is_available():
    device = "cuda:0"
    using_gpu = True
else:
    device = "cpu"

if FORCE_CPU:
    device = "cpu"
    using_gpu = False

torch.set_printoptions(precision=10)
torch.set_printoptions(edgeitems=100)

print(f"Running on {device}")

parser = argparse.ArgumentParser(description='PyTorch DDP Test')
parser.add_argument('--enable-ddp', action="store_true",
                    help='use ddp or normal training')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--machine-num', default=0, type=int,
                    help='machine identifier, starting with 0')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = MASTER_IP
    os.environ['MASTER_PORT'] = MASTER_PORT

    # initialize the process group
    dist.init_process_group(dist.Backend.GLOO, rank=rank, world_size=world_size,
                            init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}")


def cleanup():
    dist.destroy_process_group()


def run_training_for_model(rank, machine_number, world_size, enable_ddp, num_instrs, possible_instructions=None,
                           data_factor=1):
    if enable_ddp:
        setup(rank+machine_number, world_size)

    model = lstm_seq2seq(len(all_chars), 256, enable_ddp=enable_ddp)
    model = model.to(device)

    if possible_instructions is None:
        possible_instructions = ["ADD", "SUB", "MOV"]
    train_x, train_y = create_dataset(num_instrs, num_samples=10000 * num_instrs * data_factor,
                                      possible_instructions=possible_instructions)
    val_x, val_y = create_dataset(num_instrs, num_samples=1000 * num_instrs * data_factor,
                                  possible_instructions=possible_instructions)

    train_dataset = TensorDataset(train_x, train_y)
    validation_dataset = TensorDataset(val_x, val_y)

    train_loss, val_loss = train_model(model, train_dataset=train_dataset, batch_size=BATCH_SIZE, n_epochs=NUM_EPOCHS,
                                       target_len=train_y.size()[1],
                                       validation_dataset=validation_dataset, is_distributed=enable_ddp,
                                       training_prediction="teacher_forcing", learning_rate=0.01)
    if enable_ddp:
        cleanup()

    return train_loss, val_loss


if __name__ == "__main__":

    NUM_INSTRS = 4
    random.seed(42)
    torch.manual_seed(42)
    args = parser.parse_args()

    if args.enable_ddp:
        mp.spawn(run_training_for_model,
                 args=(args.machine_num, args.world_size, args.enable_ddp, NUM_INSTRS, ["ADD", "SUB", "MOV"], 1),
                 nprocs=1,
                 join=True)
    else:
        training_loss, validation_loss = run_training_for_model(-1,-1,-1,False,NUM_INSTRS, ["ADD", "SUB", "MOV"], 1)
        if ENABLE_GRAPHS:
            plt.plot(training_loss, label=f"training loss")
            plt.plot(validation_loss, label=f"validation loss")
            plt.legend(loc="upper left")
            plt.savefig(f"results_{time.time()}.png")
            plt.show()
