"""
This module defines the function `download_and_prepare_dataset`, which
gets executed when runing this module from the command line. It does
the following:
 - Downloads the dataset from the repository
 - Creates metadata.csv, which contains the name of files, their label
 and the split they correspond to (train, valid, test).
 - Removes any temporary files created during the process.

"""
import os
import random
import shutil
import sys

import pandas as pd
import tensorflow as tf

def prepare_dataset(data_dir, exclude_dirs=None):
    
    sources = find_sources(data_dir, exclude_dirs=exclude_dirs)
    labels, filepaths = zip(*sources)
    fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.JPG'
    names = [fn(name) for name in filepaths]
    splits = ['train' if random.random() <= 0.7 else 'valid' for _ in names]
    metadata = pd.DataFrame({'label': labels, 'image_name': names, 'split': splits})
    metadata.to_csv('metadata.csv', index=False)

    for name, fpath in zip(names, filepaths):
        shutil.copy(fpath, os.path.join('images', name))

def build_sources_from_metadata(metadata, data_dir, mode='train', exclude_labels=None):
    
    if exclude_labels is None:
        exclude_labels = set()
    if isinstance(exclude_labels, (list, tuple)):
        exclude_labels = set(exclude_labels)

    df = metadata.copy()
    df = df[df['split'] == mode]
    df['filepath'] = df['image_name'].apply(lambda x: os.path.join(data_dir, x))
    include_mask = df['label'].apply(lambda x: x not in exclude_labels)
    df = df[include_mask]

    sources = list(zip(df['filepath'], df['label']))
    return sources


def find_sources(data_dir, exclude_dirs=None, file_ext='.JPG', shuffle=True):
    """Find all files with its label.

    This function assumes that data_dir is set up in the following format:
    data_dir/
        - label 1/
            - file1.ext
            - file2.ext
            - ...
        - label 2/
        - .../
    Args:
        data_dir (str): the directory with the data sources in the format
            specified above.
        exclude_dirs (set): A set or iterable with the name of some directories
            to exclude. Defaults to None.
        file_ext (str): Defaults to '.JPG'
        shuffle (bool): whether to shuffle the resulting list. Defaults to True
    
    Returns:
        A list of (lable_id, filepath) pairs.
        
    """
    if exclude_dirs is None:
        exclude_dirs = set()
    if isinstance(exclude_dirs, (list, tuple)):
        exclude_dirs = set(exclude_dirs)

    sources = [
        (os.path.join(data_dir, label_dir, name), int(label_dir))
        for label_dir in os.listdir(data_dir)
        for name in os.listdir(os.path.join(data_dir, label_dir))
        if label_dir not in exclude_dirs and name.endswith(file_ext)
    ]

    random.shuffle(sources)

    return sources 

def preprocess_image(image):
    image = tf.image.resize(image, size=(32, 32))
    image = image / 255.0
    return image

def make_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
    """
    Returns an operation to iterate over the dataset specified in sources

    Args:
        sources (list): A list of (lable_id, filepath) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.

    Returns:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """
    def load(row):
        filepath = row['image']
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img)
        return img, row['label']

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4

    images, labels = zip(*sources)
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'label': list(labels)}) 

    if training:
        ds = ds.shuffle(shuffle_buffer_size)
    
    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(preprocess_image)
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download', action='store_true')
    args = parser.parse_args()

    if args.download:
        import git
        git.Git('').clone('https://github.com/ardamavi/Sign-Language-Digits-Dataset.git')

    os.mkdir('images')
    prepare_dataset('Sign-Language-Digits-Dataset/Dataset')
    shutil.rmtree('Sign-Language-Digits-Dataset')