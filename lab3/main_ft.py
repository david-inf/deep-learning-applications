"""Main program for finetuning BERT for sentiment analysis"""

from comet_ml import start
import torch
from mydata import get_loaders
from lab3.models.distilbert import get_bert
from train import train_loop, test
from utils import LOG, set_seeds, visualize, update_yaml


def get_model(opts):
    if opts.model == "distilbert":
        tokenizer, model = get_bert(opts)
    else:
        raise ValueError(f"Unknown model {opts.model}")

    return tokenizer, model


def main(opts, experiment):
    set_seeds(opts.seed)
    # Get BERT and tokenizer
    tokenizer, model = get_model(opts)
    # Get loaders
    train_loader, val_loader, test_loader = get_loaders(opts, tokenizer)
    # Training
    with experiment.train():
        LOG.info(f"Running experiment_name={opts.experiment_name}")
        train_loop(opts, model, train_loader, val_loader, experiment)
    # Testing
    with experiment.test():
        _, test_acc = test(opts, model, test_loader)
        experiment.log_metric("acc", test_acc)
        LOG.info(f"Test: accuracy={100.*test_acc:.1f}%")


def view_model(opts):
    # Get BERT and tokenizer
    tokenizer, model = get_model(opts)
    # Random data for input_ids and attention_mask
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (opts.batch_size, 70)).to(opts.device)
    attention_mask = torch.ones(
        (opts.batch_size, 70), dtype=torch.int64).to(opts.device)
    input_data = {"input_ids": input_ids, "attention_mask": attention_mask}
    # Visualize model
    visualize(model, opts.model, input_data)


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception

    opts = parse_args()

    with launch_ipdb_on_exception():
        if opts.visualize:
            view_model(opts)
        else:
            LOG.info(f"Training for num_epochs={opts.num_epochs}")
            # LOG.info(f"Checkpoint checkpoint_every={opts.checkpoint_every}")
            if not opts.experiment_key:
                experiment = start(project_name=opts.comet_project)
                experiment.set_name(opts.experiment_name)
                update_yaml(opts, "experiment_key", experiment.get_key())
                LOG.info("Added experiment key")
            else:
                experiment = start(project_name=opts.comet_project,
                                   mode="get", experiment_key=opts.experiment_key)
            main(opts, experiment)
            experiment.log_parameters(vars(opts))
            experiment.end()
