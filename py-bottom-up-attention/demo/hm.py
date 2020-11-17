import argparse
import os

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--split', default='img')
parser.add_argument('--split_json_file', default='train')
parser.add_argument('--d2_file_suffix', default='d2_36-36')
parser.add_argument('--data_path', help='path to the train/dev/test split images folders')
parser.add_argument('--output_path', help='path to the output for the extracted features')
args = parser.parse_args()


def read_img_feat(args):
    return pd.read_csv(os.path.join(args.output_path, args.d2_file_suffix, 'tsv', args.split + '.tsv'), sep='\t',
                       header=None, names=["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                                           "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"])


def merge_data(df_a, df_b, args, tiny_dataset_size=32):
    df_merged = pd.merge(df_a, df_b, left_on='id', right_on='img_id')
    df_merged.to_csv(
        os.path.join(args.data_path, 'data_' + args.split_json_file.split('.')[0] + '_' + args.d2_file_suffix + '.tsv'),
        sep='\t', header=False, index=False)
    df_merged.sample(tiny_dataset_size).to_csv(
        os.path.join(args.data_path, 'tiny_data_' + args.split_json_file.split('.')[0] + '_' + args.d2_file_suffix + '.tsv'),
        sep='\t', header=False, index=False)


def merge(df, args):
    df_boxes = read_img_feat(args)
    merge_data(df, df_boxes, args)


if __name__ == "__main__":
    df_json = pd.read_json(os.path.join(args.data_path, args.split_json_file), lines=True)
    merge(df_json, args)
