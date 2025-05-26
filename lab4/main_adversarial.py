"""Generate adversarial examples for a given model and dataset"""

import sys
import os
from types import SimpleNamespace
import random

import torch
import yaml
# from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab4.mydata import get_loaders, ID_CLASSES

from lab1.main_train import get_model
from lab1.utils.train import load_checkpoint
from lab1.utils import set_seeds, LOG, N


def attack(pred: int, gt: int, image, model, eps, target=None):
    """Do a targeted or untargeted adversarial attack"""
    pred_x = pred  # prediction with original image
    if pred_x.item() != gt.item() or gt.item() == target:
        print("Classifier is already wrong ot target label same as GT!")
    else:
        done = False
        n = 0  # amount of budget spent
        loss_fn = torch.nn.CrossEntropyLoss()
        while not done:
            image.retain_grad()  # ??
            output = model(image.unsqueeze(0))  # [1, K]

            if target is None:
                # proceed with untargeted attack
                yt = gt.unsqueeze(0)
            else:
                # proceed with targeted attack
                yt = target.unsqueeze(0)

            model.zero_grad()
            loss = loss_fn(output, yt)
            loss.backward()

            if target is None:
                # untargeted FGSM
                # image += eps * torch.sign(image.grad)
                image = image + eps * torch.sign(image.grad)
            else:
                # targeted FGSM
                image = image - eps * torch.sign(image.grad)

            # TODO: do again prediction
            output = model(image.unsqueeze(0))  # [1, K]
            pred = N(output).argmax()  # prediction with corrupted image

            n += 1

            if target is None and pred != gt:
                # did the prediction change?
                budget = int(255*n*eps)
                print(f'Untargeted attack success! budget:{budget}/255')
                done = True

            if target is not None and pred == target:
                # l'idea è di farlo andare verso la classe specificata
                budget = int(255*n*eps)
                print(f"Targeted attack ({ID_CLASSES[pred]})"
                      f" success! budget:{budget}/255")
                done = True

    return image, pred


def adversarials(opts, model, loader):
    """Generate adversarial examples"""
    from rich.console import Console
    from rich.table import Table
    n_samples = 5
    images, labels = next(iter(loader))
    # TODO: put on device just the images and labels needed
    images, labels = images.to(opts.device), labels.to(opts.device)
    samples = [random.randint(0, images.size(0)) for _ in range(n_samples)]  # n

    # original images
    images_orig = images[samples]  # [n]
    # images to be corrupted
    images_adv = images_orig.clone()  # [n]
    images_adv.requires_grad = True  # compute the gradient over these

    # do as many steps as needed for making the classifier do a wrong prediction
    model.eval()
    pred_x = model(images_orig).argmax(1)  # [n]
    pred_adv = 15+torch.zeros_like(pred_x)  # [n]
    corrupted = []
    for i, sample_id in enumerate(samples):
        if opts.target is None:
            target = None
        else:
            target = torch.tensor(ID_CLASSES.index(opts.target)).to(opts.device)
        image, pred = attack(
            pred_x[i], labels[sample_id], images_adv[i],
            model, opts.budget/255, target)
        # images_adv[i] = image
        corrupted.append(image.detach().cpu().unsqueeze(0))
        pred_adv[i] = pred


    # print results
    table = Table(title="Adversarial attacks")
    table.add_column("id", justify="right")  # sample_id, image
    table.add_column("gt")  # groundtruth, actual label
    table.add_column("pred x")  # model prediction on original image
    table.add_column("pred x'")  # model prediction on attacked image

    for i, sample_id in enumerate(samples):
        gt = ID_CLASSES[N(labels)[sample_id]]
        table.add_row(str(sample_id), gt, str(pred_x[i].item()), str(pred_adv[i].item()))
    console = Console()
    console.print(table)

    # Print adversarial examples
    advs = torch.cat(corrupted)
    images_orig = images_orig.detach().cpu()
    diff = images_orig - advs
    fig, axs = plt.subplots(n_samples, 3)
    plt.tight_layout()
    for i in range(n_samples):
        axs[i, 0].imshow(images_orig[i].permute(1,2,0))
        axs[i, 0].set_title(f"pred: {ID_CLASSES[pred_x[i]]}")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(torch.clamp(advs[i].permute(1,2,0), 0., 1.))
        axs[i, 1].set_title(f"pred: {ID_CLASSES[pred_adv[i]]}")
        axs[i, 1].axis("off")
        axs[i, 2].imshow(torch.clamp(diff[i].permute(1,2,0), 0., 1.))
        axs[i, 2].set_title("diff")
        axs[i, 2].axis("off")

    # grid = make_grid(torch.cat((images_orig, advs, diff), dim=0),
    #                  nrow=3, padding=3, normalize=True)
    # plt.imshow(grid.permute(1, 2, 0))
    # plt.axis("off")
    output_path = os.path.join("lab4/plots/adversarial", f"{opts.attack}.svg")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Printed img={output_path}")


def main(opts):
    """Main function to generate adversarial examples"""
    set_seeds(opts.seed)
    # Load model checkpoint
    with open(opts.model_configs, "r", encoding="utf-8") as f:
        model_configs = yaml.safe_load(f)
    model_opts = SimpleNamespace(**model_configs)
    model_opts.device = opts.device
    
    # Load model
    model = get_model(model_opts)
    load_checkpoint(model_opts.checkpoint, model, opts.device)    

    # Load data
    # train split so we have garauntees that the model will
    # be predicting the correct labels with original images
    loader = get_loaders(model_opts, train=True)

    # Generate adversarial examples
    adversarials(opts, model, loader)
    
    print("Adversarial examples generated and saved successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_configs", type=str,
                        help="Path to model configuration file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--attack", type=str, default="untargeted", choices=["untargeted", "targeted"],
                        help="Type of adversarial attack to perform")
    parser.add_argument("--target", type=str, default=None,
                        help="Class that the model should predict")
    parser.add_argument("--budget", type=int, default=1, help="Budget per each step")
    args = parser.parse_args()
    args.seed = 42

    try:
        main(args)
    except Exception:
        import ipdb
        import traceback
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
