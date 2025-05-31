"""Generate adversarial examples for a given model and dataset"""

import sys
import os
from types import SimpleNamespace
import random

import torch
import yaml

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab1.main_train import get_model
from lab1.utils.train import load_checkpoint
from lab1.utils import set_seeds, N
from lab1.mydata import MyCIFAR10

from lab4.mydata import ID_CLASSES
from lab4.utils.adversarial import plot_attacked_images, print_summary_table


def attack(gt: int, image, model, eps, target=None):
    """Do a targeted or untargeted adversarial attack"""
    n = 0  # amount of budget spent
    done = False
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

        output = model(image.unsqueeze(0))  # [1, K]
        pred = N(output).argmax()  # prediction with corrupted image
        n += 1

        if target is None and pred != gt:
            # did the prediction change?
            budget = int(255*n*eps)
            # print(f'Untargeted attack success! budget: {budget}/255')
            done = True

        if target is not None and pred == target:
            # l'idea è di farlo andare verso la classe specificata
            budget = int(255*n*eps)
            # print(f"Targeted attack ({ID_CLASSES[pred]})"
            #         f" success! budget: {budget}/255")
            done = True

    return image, pred, n


def adversarials(opts, model, dataset):
    """Generate adversarial examples"""
    set_seeds(opts.seed)
    samples = [dataset[random.randint(0, len(dataset))]
               for _ in range(opts.n_samples)]  # list of (image, label)

    images_orig = []  # original images
    preds_orig = []  # predictions on original images
    images_adv = []  # attacked images
    preds_adv = []  # predictions on attacked images

    # do as many steps as needed for making the classifier do a wrong prediction
    model.eval()
    iters_list = []

    for i, (image, label) in enumerate(samples):
        if opts.target is None:
            target = None
        else:
            target = torch.tensor(ID_CLASSES.index(opts.target)).to(opts.device)

        image, label = image.to(opts.device), label.to(opts.device)
        image.requires_grad = True  # already tensor in [0,1]

        images_orig.append(N(image))
        output = model(image.unsqueeze(0))  # needs the batch dimension
        pred = output.argmax()  # prediction
        preds_orig.append(N(pred))

        # attack current image
        if pred.item() != label.item():
            print(f"Image {i} classifier is already wrong")
            pred = N(pred)
            iters = 0
            # image remains unchanged
        elif label.item() == target:
            print(f"Image {i} target label same as GT")
            pred = N(pred)
            iters = 0
            # image remains unchanged
        # if pred.item() != label.item() or label.item() == target:
        #     print("Classifier is already wrong or target label same as GT!")
        #     pred = N(pred)
        #     iters = 0
        #     # image remains unchanged
        else:
            image, pred, iters = attack(
                label, image.clone(),
                model, opts.budget/255, target)

        images_adv.append(N(image))
        preds_adv.append(pred)
        iters_list.append(iters)

    # Print summary table with attack results
    print_summary_table(samples, preds_orig, preds_adv, iters_list)
    # Plot attacked images
    plot_attacked_images(opts, images_orig, images_adv, preds_orig, preds_adv, iters_list)


def main(opts):
    """Main function to generate adversarial examples"""
    # Load model checkpoint
    with open(opts.model_configs, "r", encoding="utf-8") as f:
        model_configs = yaml.safe_load(f)
    model_opts = SimpleNamespace(**model_configs)
    model_opts.device = opts.device
    
    # Load model
    model = get_model(model_opts)
    load_checkpoint(model_opts.checkpoint, model, opts.device)    

    # Load data
    id_set = MyCIFAR10(model_opts, train=False)  # in-distribution data

    # Generate adversarial examples
    adversarials(opts, model, id_set)
    print("Adversarial examples generated and saved successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed for changing images")
    parser.add_argument("--model_configs", type=str,
                        help="Path to model configuration file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--attack", type=str, default="untargeted", choices=["untargeted", "targeted"],
                        help="Type of adversarial attack to perform")
    parser.add_argument("--target", type=str, default=None,
                        help="Class that the model should predict")
    parser.add_argument("--budget", type=int, default=1, help="Budget per each step")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of samples to attack and visualize")
    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        import ipdb
        import traceback
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
