import random
from typing import Iterable, Tuple, List
import os 
def get_images(
    paths: Iterable[str], labels: Iterable[any], nb_samples=None, shuffle=True
) -> List[Tuple]:
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """

    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels: List[tuple] = []

    for i, path in zip(labels, paths):
        for image in sampler(os.listdir(path)):
            images_labels.append((i, os.path.join(path, image),))

    if shuffle:
        random.shuffle(images_labels)
    return images_labels
