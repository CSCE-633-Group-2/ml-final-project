import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import importlib
import matplotlib.pyplot as plt
import os
# from transformers import AutoModel

from lstm import MyModel, Trainer

data_processor = importlib.import_module('lstm-data-processor')
hyperparams = importlib.import_module('lstm-hyperparams')

def plot_and_save_metrics(history, save_dir="./data/"):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Plot Loss
    axes[0].plot(epochs, history["train_loss"], label='Train Loss', color='blue', marker='o')
    if any(history["val_loss"]):  # Only plot if validation was run
        axes[0].plot(epochs, history["val_loss"], label='Val Loss', color='red', marker='o')
    axes[0].set_title('Loss vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Plot Accuracy
    axes[1].plot(epochs, history["train_acc"], label='Train Acc', color='blue', marker='o')
    if any(history["val_acc"]): 
        axes[1].plot(epochs, history["val_acc"], label='Val Acc', color='red', marker='o')
    axes[1].set_title('Accuracy vs. Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Plot F1 Score
    axes[2].plot(epochs, history["train_f1"], label='Train F1', color='blue', marker='o')
    if any(history["val_f1"]):
        axes[2].plot(epochs, history["val_f1"], label='Val F1', color='red', marker='o')
    axes[2].set_title('F1 Score vs. Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    
    # Save and close
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300) # dpi=300 ensures a high-quality, crisp image
    plt.close(fig) # Free up memory
    print(f"Training curves successfully saved to {save_path}")

def load_train_objs(infiles_list : list[str], data_type : str, rank : int):
    *output, tokenizer = data_processor.load_and_preprocess_data(
        data_paths=infiles_list, 
        data_type=data_type,
        model_type='transformer'
    )
    
    # 1. Initialize model
    model = MyModel(bidirectional=True)
    
    # 2. Move model to GPU FIRST
    model = model.to(rank)

    # 3. Initialize optimizer with GPU parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=hyperparams.OPTIMIZER_LR, 
        weight_decay=hyperparams.OPTIMIZER_WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )
    
    return *output, model, optimizer, scheduler

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12281"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, infiles : str, data_type : str):
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    infiles_list = infiles.split(",")

    train_dataset, *output, model, optimizer, scheduler = load_train_objs(infiles_list, data_type, rank)
    val_dataset = output[0] if output else None

    train_data = data_processor.prepare_distributed_dataloader(train_dataset, batch_size)
    val_data = data_processor.prepare_distributed_dataloader(val_dataset, batch_size) if val_dataset is not None else None

    trainer = Trainer(model, train_data, val_data, optimizer, scheduler, rank, save_every)
    
    history = trainer.train(total_epochs)
    
    if rank == 0:
        plot_and_save_metrics(history, save_dir='./data/plots/')

    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--infiles', type=str, help='Comma-separated list of input files')
    parser.add_argument('--data_type', type=str, choices=['train', 'train_val'], help='Whether validation needs to be carried out')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, args.infiles, args.data_type), nprocs=world_size) # type: ignore