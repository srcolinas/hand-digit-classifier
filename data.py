import os
import random

import tensorflow as tf

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
        (int(label_dir), os.path.join(data_dir, label_dir, name))
        for label_dir in os.listdir(data_dir)
        for name in os.listdir(os.path.join(data_dir, label_dir))
        if label_dir not in exclude_dirs and name.endswith(file_ext)
    ]

    random.shuffle(sources)

    return sources 


def dataset_iterator(sources, training=False, batch_size=1,
    num_batches=1, num_parallel_calls=None, shuffle_buffer_size=None):
    """
    Returns an operation to iterate over the dataset specified in sources

    Args:
        sources (list): A list of (lable_id, filepath) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.

    Returns:
        A tf.Tensor which moves over the dataset every time it is executed
            (i.e. tf.data.Iterator().get_next())
    """
    def parse_fn(filepath):
        out = tf.read_file(filepath)
        out = tf.image.decode_jpeg(out)
        return out

    if num_parallel_calls is None:
        num_parallel_calls = tf.data.experimental.AUTOTUNE
    if shuffle_buffer_size is None:
        shuffle_buffer_size = tf.data.experimental.AUTOTUNE

    labels, images = zip(*sources)

    images = tf.data.Dataset.from_tensor_slices(images)    
    images = images.map(parse_fn, num_parallel_calls=num_parallel_calls)

    ds = tf.data.Dataset.zip((labels, images))
    if training:
        ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size=batch_size)
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element