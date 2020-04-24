import numpy as np
import os
import random
from matplotlib.pyplot import imread
import torch
import debug as db
import ImUtil

def get_images(paths, labels, nb_samples=None, shuffle=True):
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
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32)
    image = 1.0 - image
    return image


def scramble(X, axis=0, inline=True):
    """Shuffle along any axis of a tensor."""
    if not inline:
        X = X.copy()
    np.apply_along_axis(np.random.shuffle, axis, X)
    if not inline:
        return X

class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))
        self.imnumel = self.img_size[0] * self.img_size[1]

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

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
        #
        # Sample class.
        folders = np.array(folders)
        class_idx = np.random.randint(0, len(folders), size=(batch_size * self.num_classes))
        # selected_folders = folders.take(class_idx)
        # labels = np.concatenate([np.arange(self.num_classes)] * batch_size, 0)
        # sample_list = get_images(selected_folders, labels, self.num_samples_per_class, shuffle=True)
        # label, impth = zip(*sample_list)
        #
        # Generate image and label tensors.
        all_image_batches = []
        all_label_batches_onehot = []
        all_label_batches = []
        for b_idx in range(batch_size):
            labels = np.arange(self.num_classes)
            selected_folders = folders.take(class_idx[b_idx*self.num_classes:(b_idx+1)*self.num_classes])
            samples = get_images(selected_folders, labels, self.num_samples_per_class, shuffle=False)
            idx, sample_file = zip(*samples)
            #
            # Get the scramble indexes per sequence.
            idx_sample = np.arange(self.num_classes).reshape(-1, 1)
            idx_sample = np.repeat(idx_sample, self.num_samples_per_class, axis=1)
            scramble(idx_sample)
            sample_file = np.array(sample_file)
            sample_file = sample_file.reshape(self.num_classes, self.num_samples_per_class)
            idx_sample = idx_sample.T
            task_imgs = []
            for col_idx, sample_idxr in enumerate(idx_sample):
                for lbl in sample_idxr:
                    f = sample_file[lbl, col_idx]
                    # task_imgs.append(f)
                    task_imgs.append(image_file_to_array(f, self.imnumel))
            task_imgs = np.array(task_imgs)
            # task_imgs = np.array(task_imgs).reshape(, self.num_classes, -1)
            one_hot = np.zeros([self.num_samples_per_class, self.num_classes, self.num_classes], dtype=np.float32)
            np.put_along_axis(one_hot, np.expand_dims(idx_sample, 2), 1, 2)


            all_image_batches.append(task_imgs)
            all_label_batches.append(idx_sample)
            all_label_batches_onehot.append(one_hot)
        all_label_batches_onehot = np.stack(all_label_batches_onehot, 0)
        all_label_batches = np.stack(all_label_batches, 0)
        all_image_batches = np.stack(all_image_batches, 0)
        # import torch
        # db.printInfo(all_label_batches)
        # ImUtil.showBatch(torch.from_numpy(all_image_batches).view(-1, 1, 28, 28), show=True)
        return all_image_batches, all_label_batches_onehot, all_label_batches
