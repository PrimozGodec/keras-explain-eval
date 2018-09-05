import sys
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from skimage import measure
import math
import pickle

approaches_name = {
    'SundararajanIntegrated': "Integrated gradients",
    'Bach_LRP': "Layer-wise rel. prop.",
    'SelvarjuGuidedGradCam': "Guided GradCam" ,
    'Simonyan': "Saliency",
    'SpingenbergGuidedBP': "Guided back propagation",
    'zeiler': "Basic graying out",
    'Zintgraf': "Prediction difference",
    'ribeiro_lime': "LIME",
    'SelvarjuGradCam': "GradCam",
    'original': 'Original',
}


def read_images(im_names, shape):
    h, w, d = shape
    im_orig = np.zeros((len(im_names), h, w, d))
    for i, img in enumerate(im_names):
        im = Image.open(os.path.join(location, "original", img))
        im = im.resize((h, w), Image.ANTIALIAS)
        im_orig[i] = np.array(im)
    return im_orig / 255.0


def read_masks(approaches, images, shape):
    h, w = shape
    masks = {}
    for approach in approaches:
        masks_ap = np.zeros((len(images), h, w))
        for i, img in enumerate(images):
            im = Image.open(os.path.join(location, approach, "pos", img))
            im = im.resize((h, w), Image.ANTIALIAS)
            masks_ap[i] = np.array(im)
        masks_ap = masks_ap / 255.0
        masks[approach] = masks_ap
    return masks


def compare(location, model, patch_size):
    approaches = sorted(
        list(set(os.listdir(location)) -
             {"all", "original", ".DS_Store", "classes.json", "paper"}))
    if not os.path.isfile("acc.pkl"):

        # list original images
        images = os.listdir(os.path.join(location, "all"))
        im_names = ["-".join(x.split("-")[:-2]) for x in images]
        im_targets = [int((x.split("-")[-1]).split(".")[0]) for x in images]
        im_predictions = [int((x.split("-")[-2])) for x in images]

        h, w, d = model.input_shape

        # read original images
        im_orig = read_images(im_names, model.input_shape)

        # read masks
        masks = read_masks(approaches, images, (h, w))

        pooled_masks = {n: measure.block_reduce(
            mask, (1, patch_size,patch_size), np.sum) for n, mask in masks.items()}

        # for each threshold
        model.load()

        replacement_patch = np.zeros((patch_size, patch_size))
        accuracies = {approach: [] for approach in approaches}

        # filter relevant images
        cond = np.array(im_targets) == np.array(im_predictions)
        tmp_orig = im_orig[cond]
        target_classes = np.array(im_targets)[cond]
        tmp_masks = {n: v[cond] for n, v in masks.items()}
        tmp_pool_masks = {n: v[cond] for n, v in pooled_masks.items()}
        removed = {n: np.zeros_like(mask).astype(bool)
                   for n, mask in tmp_masks.items()}

        r = int(math.ceil(h / patch_size) * math.ceil(w / patch_size))
        for patch_no in range(
                int(math.ceil(h / patch_size) * math.ceil(w / patch_size))):
            print(patch_no, "/", r)
            # for each approach
            for approach in approaches:
                # remove based on the contribution
                p_mask = tmp_pool_masks[approach]
                h, w = p_mask.shape[1:3]

                for im in range(p_mask.shape[0]):
                    x, y = np.unravel_index(
                        np.argmax(p_mask[im], axis=None), (h, w))
                    removed[approach][im, x * patch_size: x * patch_size + patch_size,
                        y * patch_size: y * patch_size + patch_size] = True
                    tmp_pool_masks[approach][im, x, y] = -1

                temp_images = tmp_orig * (~removed[approach])[..., None]

                # precict
                pr = model.predict(temp_images)

                pr_target_classes = pr[np.arange(pr.shape[0]), target_classes]
                accuracies[approach].append(np.mean(pr_target_classes))

        with open("acc.pkl", "wb") as f:
            pickle.dump(accuracies, f)
    else:
        with open("acc.pkl", "rb") as f:
            accuracies = pickle.load(f)

    # plot predictions
    for approach in approaches:
        plt.plot(np.arange(len(accuracies[approach])) / len(accuracies[approach]) * 100,
                 accuracies[approach], label=approaches_name[approach])
    plt.legend(loc='best')
    plt.xlabel("Percent of removed patches")
    plt.ylabel("Classification accuracy")
    plt.show()

    if not os.path.isdir("../data/plots/"):
        os.makedirs("../data/plots/")

    plt.savefig("../data/plots/error_drop.png")


if __name__ == "__main__":
    patch_size = 4

    # dir with explanation results
    # each sub-dir present explanation from an approach
    # there must be also originl dir which contains original images
    # for visualisation
    location = sys.argv[1]

    # class with Keras model
    model = ExampleModel(4)
    compare(location, model, patch_size)