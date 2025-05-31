
import sys
import os

import torch
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab4.mydata import ID_CLASSES


def print_summary_table(samples, preds_orig, preds_adv, iters_list):
    """Table summarizing the attacks"""
    table = Table(title="Adversarial attacks")
    table.add_column("id", justify="right")  # sample_id, image
    table.add_column("gt")  # groundtruth, actual label
    table.add_column("pred x")  # model prediction on original image
    table.add_column("pred x'")  # model prediction on attacked image
    table.add_column("iters")

    for i, (_, label) in enumerate(samples):
        gt = ID_CLASSES[label] + f" ({label.item()})"  # string
        table.add_row(str(i), gt, str(preds_orig[i]), str(preds_adv[i]), str(iters_list[i]))
    console = Console()
    console.print(table)


def plot_attacked_images(opts, images_orig, images_adv, preds_orig, preds_adv, budgets):
    """Grid with attacked images evaluation"""
    fig, axs = plt.subplots(opts.n_samples, 5, figsize=(12, 10))
    fig.suptitle(f"Attack type: {opts.attack}")

    for i, (image_orig, image_adv) in enumerate(zip(images_orig, images_adv)):
        image_orig = torch.from_numpy(image_orig)
        image_adv = torch.from_numpy(image_adv)
        diff = image_orig - image_adv
        # original image
        axs[i, 0].imshow(image_orig.permute(1,2,0))
        axs[i, 0].set_title(f"pred: {ID_CLASSES[preds_orig[i]]}")
        axs[i, 0].axis("off")

        # attacked image
        axs[i, 1].imshow(torch.clamp(image_adv.permute(1,2,0), 0., 1.))
        axs[i, 1].set_title(f"pred: {ID_CLASSES[preds_adv[i]]}")
        axs[i, 1].axis("off")

        # # difference, with 3 channels
        axs[i, 2].imshow(torch.clamp(diff, 0., 1.).permute(1,2,0))
        axs[i, 2].set_title("diff")
        axs[i, 2].axis("off")

        # # difference, reducted on 1 channel
        axs[i, 3].imshow(255*torch.clamp(diff, 0., 1.).squeeze().mean(0), cmap=plt.get_cmap('PuOr'))
        axs[i, 3].set_title("reduction")
        axs[i, 3].axis("off")

        # attack magnitude distribution
        axs[i, 4].hist(255*torch.clamp(diff, 0., 1.).flatten(), density=True)
        axs[i, 4].set_xlabel("magnitude")
        axs[i, 4].set_title(f"steps: {budgets[i]}")

    plt.tight_layout()
    output_path = os.path.join("lab4/plots/adversarial", f"{opts.attack}.svg")
    plt.savefig(output_path)
    print(f"Printed img={output_path}")
