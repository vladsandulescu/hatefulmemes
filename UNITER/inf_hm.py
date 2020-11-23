"""UNITER inference for HM """
import argparse
import json
import os
from os.path import exists
from time import time

import numpy as np
import torch
from apex import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import (HMTestDataset,
                  hm_test_collate,
                  HMPairedTestDataset,
                  hm_paired_test_collate,
                  PrefetchLoader, TokenBucketSampler)
from model.hm import (UniterForHm, UniterForHmPaired, UniterForHmPairedAttn)
from model.model import UniterConfig
from utils.const_hm import IMG_DIM, BUCKET_SIZE
from utils.misc import Struct


def create_dataloader(opts, dataset_cls, collate_fn, mode='test'):
    assert mode in ['test']

    image_set = opts.test_image_set
    batch_size = opts.batch_size

    dataset = dataset_cls(image_set, opts.root_path, opts.dataset_path,
                          use_img_type=opts.use_img_type, test_mode=(mode == 'test'))
    sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=False)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)


def main(opts):
    device = torch.device("cuda")  # support single GPU only
    train_opts = Struct(json.load(open(f'{opts.train_dir}/log/hps.json')))

    if 'paired' in train_opts.model:
        TestDatasetCls = HMPairedTestDataset
        test_collate_fn = hm_paired_test_collate
    if train_opts.model == 'paired':
        ModelCls = UniterForHmPaired
    elif train_opts.model == 'paired-attn':
        ModelCls = UniterForHmPairedAttn
    elif train_opts.model == 'cls':
        TestDatasetCls = HMTestDataset
        test_collate_fn = hm_test_collate
        ModelCls = UniterForHm
    else:
        raise ValueError('unrecognized model type')

    # data loaders
    opts.batch_size = (train_opts.val_batch_size if opts.batch_size is None else opts.batch_size)
    opts.use_img_type = train_opts.use_img_type
    test_dataloader = create_dataloader(opts, TestDatasetCls, test_collate_fn, mode='test')

    # Prepare model
    ckpt_file = f'{opts.train_dir}/ckpt/model_step_{opts.ckpt}.pt'
    checkpoint = torch.load(ckpt_file)
    model_config = UniterConfig.from_json_file(
        f'{opts.train_dir}/log/model.json')
    model = ModelCls(model_config, img_dim=IMG_DIM)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    results, results_logits = inference(model, test_dataloader)
    if not exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    with open(f'{opts.output_dir}/results.csv', 'w') as f:
        f.write("id,proba,label\n")
        for id_, pred, prob in results:
            padded_id = id_
            if len(id_) == 4:
                padded_id = "0" + str(id_)
            f.write(f'{padded_id},{str(np.round(prob, 4))},{pred}\n')


@torch.no_grad()
def inference(model, test_loader):
    print("start running inference...")
    model.eval()
    n_ex = 0
    st = time()
    results = []
    results_logits = []
    for i, batch in enumerate(test_loader):
        img_ids = batch['img_ids']
        del batch['img_ids']
        scores = model(**batch, targets=None, compute_loss=False)
        predictions = scores.max(dim=-1, keepdim=False)[1].cpu().tolist()
        probs = F.softmax(scores, dim=1)[:, 1].cpu().tolist()
        logits = scores.cpu().tolist()
        results.extend(zip(img_ids, predictions, probs))
        results_logits.extend(zip(img_ids, predictions, probs, logits))
        n_ex += len(img_ids)
    tot_time = time() - st
    model.train()
    print(f"inference finished in {int(tot_time)} seconds "
          f"at {int(n_ex / tot_time)} examples per second")

    return results, results_logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default=None, type=str,
                        help="The root path of the project.")
    parser.add_argument("--dataset_path",
                        default=None, type=str,
                        help="The path to the dataset test file.")
    parser.add_argument("--test_image_set",
                        default=None, type=str,
                        help="The input test images tsv file name.")
    parser.add_argument("--batch_size", type=int,
                        help="batch size for evaluation")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--fp16', action='store_true',
                        help="fp16 inference")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="The directory storing HM finetuning output")
    parser.add_argument("--ckpt", type=int, required=True,
                        help="specify the checkpoint to run inference")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the prediction results will be written.")

    args = parser.parse_args()

    main(args)
