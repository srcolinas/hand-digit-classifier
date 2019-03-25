"""This module trains the model.

This module should be seen as a template, that is, it may change for
every experiment. And it is the developers responsibility to make
the changes that are necessary for the experiment to work as expected.
However, the changes are expected to be minimal.

"""
import argparse
import os

import pandas as pd
import tensorflow as tf

import data
import model
import train


def main(args):

    metadata = pd.read_csv(args.metadata)
    exclude_labels = set(int(lbl) for lbl in args.exclude_labels.split('|'))
    num_labels = len(set(metadata['label'].values.tolist())) - len(exclude_labels)
    # TODO: define training and validation iteratos
    train_dataset = data.build_sources_from_metadata(metadata, args.data_dir,
        exclude_labels=exclude_labels)

    train_dataset = data.make_dataset(train_dataset, training=True,
        batch_size=args.batch_size, num_epochs=1,
        num_parallel_calls=3)
    #

    # TODO: instantiate the model of your choice
    net = model.Baseline(num_labels=num_labels)
    #

    # TODO: define loss function
    
    #

    # TODO: define optimzer
    
    #

    checkpoint_path = os.path.join(args.checkpoint_dir, 'tf_ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
            save_weights_only=True, verbose=1)

    net.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    net.fit(x=train_dataset, epochs=args.num_epochs, callbacks=[cp_callback])



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run an experiment")
    parser.add_argument('metadata')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--data-dir', default='image_files')
    parser.add_argument('--exclude-labels', default='6|7|8|9')
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.0001, type=float)
    parser.add_argument('--log-every-n-steps', default=10, type=int)

    args = parser.parse_args()
    print(args)

    main(args)