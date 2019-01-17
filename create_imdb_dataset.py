from tensor2tensor.data_generators.imdb import SentimentIMDB
import argparse
import tensorflow as tf
import tempfile
import pandas as pd
from pathlib import Path
import sys


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    output_dir = Path(args.output_dir)

    tf.gfile.MakeDirs(str(output_dir))

    data_dir = tempfile.gettempdir()
    temp_dir = tempfile.gettempdir()

    imdb = SentimentIMDB()
    dataset_splits = ['train', 'test']

    for dataset_split in dataset_splits:
        tf.logging.info(f'Creating {dataset_split} split ...')
        data_df = imdb.generate_samples(data_dir=data_dir,
                                        tmp_dir=temp_dir,
                                        dataset_split=dataset_split)

        data_df = pd.DataFrame(data_df)
        data_df['id'] = data_df.index.values
        data_df = data_df.rename(columns={'inputs': 'text'})
        data_df = data_df[['id', 'label', 'text']]

        output_file = output_dir / f'{dataset_split}.tsv'
        data_df.to_csv(output_file, index=False, sep='\t')

    tf.logging.info('Finished creating the IMDB dataset!')


if __name__ == '__main__':
    assert sys.version_info >= (3, 6)
    parser = argparse.ArgumentParser(description='Use tensor2tensor to create the imdb training and test set.')
    parser.add_argument('--output_dir', help='Path to save the training and test files', required=True)
    args = parser.parse_args()
    main(args)
