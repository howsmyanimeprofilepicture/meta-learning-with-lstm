from pathlib import Path
import random
from typing import *


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self,
        num_classes: int,
        num_samples_per_class: int,
        data_folder: str = "./omniglot_resized",
        img_size=(28, 28),
    ):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """


        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.img_size: Tuple[int, int] = img_size
        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = []

        for family in Path(data_folder).glob("*"):
            if not family.is_dir():
                for character in family.glob("*"):
                    if character.is_dir():
                        character_folders.append(character)

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[
            num_train : num_train + num_val
        ]
        self.metatest_character_folders = character_folders[num_train + num_val :]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        pass
        #############################

        return (
            all_image_batches.astype(np.float32),
            all_label_batches.astype(np.float32),
        )

