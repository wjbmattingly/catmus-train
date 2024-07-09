import argparse
from src.data import load_data
from src.models import train_model

def main():
    parser = argparse.ArgumentParser(description="Train a model on medieval script data")

    # Add arguments for all kwargs
    parser.add_argument('--shuffle_seed', type=int, default=42, help="Seed for shuffling the data")
    parser.add_argument('--select_range', type=int, default=1000, help="Number of samples to select from the dataset")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--logging_steps', type=int, default=100, help="Logging steps")
    parser.add_argument('--save_steps', type=int, default=100, help="Save steps")
    parser.add_argument('--save_limit', type=int, default=2, help="Total number of checkpoints to keep")
    parser.add_argument('--device', type=str, default='mps:0', help="Device to use for training")
    parser.add_argument('--version', type=str, default=None, help="Version of the model to save")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument('--from_pretrained_model', type=str, default="medieval-data/trocr-medieval-base", help="Path to pretrained model")
    parser.add_argument('--scripts', type=str, nargs='+', default=["Caroline"], help="List of scripts to train on")

    args = parser.parse_args()

    # Convert argparse Namespace to dictionary
    kwargs = vars(args)

    scripts = kwargs.pop('scripts')

    for script in scripts:
        train_model.train_model(script=script, **kwargs)

if __name__ == "__main__":
    main()
