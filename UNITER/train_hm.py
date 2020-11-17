"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for HM
"""
import argparse
import os
from os.path import exists, join
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (DistributedTokenBucketSampler,
                  HMDataset, HMEvalDataset, HMTestDataset,
                  hm_collate, hm_eval_collate, hm_test_collate,
                  HMPairedDataset, HMPairedEvalDataset, HMPairedTestDataset,
                  hm_paired_collate, hm_paired_eval_collate, hm_paired_test_collate,
                  PrefetchLoader)
from model.hm import (UniterForHm, UniterForHmPaired, UniterForHmPairedAttn)
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const_hm import IMG_DIM, BUCKET_SIZE

import numpy as np
from sklearn.metrics import roc_auc_score


def create_dataloader(opts, dataset_cls, collate_fn, mode='train'):
    assert mode in ['train', 'val', 'test']
    if mode == 'train':
        image_set = opts.train_image_set
        batch_size = opts.train_batch_size
    elif mode == 'val':
        image_set = opts.val_image_set
        batch_size = opts.val_batch_size
    else:
        image_set = opts.test_image_set
        batch_size = opts.val_batch_size

    dataset = dataset_cls(image_set, opts.root_path, opts.dataset_path,
                       use_img_type=opts.use_img_type, test_mode=(mode == 'test'))
    sampler = DistributedTokenBucketSampler(
        hvd.size(), hvd.rank(), dataset.lens,
        bucket_size=BUCKET_SIZE, batch_size=batch_size,
        droplast=False, shuffle=(mode == 'train'))
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    if 'paired' in opts.model:
        DatasetCls = HMPairedDataset
        EvalDatasetCls = HMPairedEvalDataset
        TestDatasetCls = HMPairedTestDataset
        collate_fn = hm_paired_collate
        eval_collate_fn = hm_paired_eval_collate
        test_collate_fn = hm_paired_test_collate
    if opts.model == 'paired':
        ModelCls = UniterForHmPaired
    elif opts.model == 'paired-attn':
        ModelCls = UniterForHmPairedAttn
    elif opts.model == 'cls':
        DatasetCls = HMDataset
        EvalDatasetCls = HMEvalDataset
        TestDatasetCls = HMTestDataset
        collate_fn = hm_collate
        eval_collate_fn = hm_eval_collate
        test_collate_fn = hm_test_collate
        ModelCls = UniterForHm
    else:
        raise ValueError('unrecognized model type')

    # data loaders
    train_dataloader = create_dataloader(opts, DatasetCls, collate_fn, mode='train')
    val_dataloader = create_dataloader(opts, EvalDatasetCls, eval_collate_fn, mode='val')
    test_dataloader = create_dataloader(opts, TestDatasetCls, test_collate_fn, mode='test')

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}

    model = ModelCls.from_pretrained(opts.model_config, state_dict=checkpoint,
                                     img_dim=IMG_DIM)
    # model.init_type_embedding()
    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2',
                                      keep_batchnorm_fp32=False,
                                      loss_scale=128.0,
                                      min_loss_scale=128.0
                                      )

    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        os.makedirs(join(opts.output_dir, 'results'))  # store val predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataloader.dataset))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    model.train()
    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    while True:
        for step, batch in enumerate(train_dataloader):
            targets = batch['targets']
            n_examples += targets.size(0)

            loss = model(**batch, compute_loss=True)
            loss = loss.mean()
            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())

            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                losses = all_gather_list(running_loss)
                running_loss = RunningMeter(
                    'loss', sum(l.val for l in losses)/len(losses))
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time()-start))
                    LOGGER.info(f'Step {global_step}: '
                                f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s, '
                                f'Loss: {loss}')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)

                if global_step % opts.valid_steps == 0:
                    for split, loader in [('val', val_dataloader), ('test', test_dataloader)]:
                        LOGGER.info(f"Step {global_step}: start running "
                                    f"validation on {split} split...")
                        log, results, results_logits = validate(model, loader, split)
                        with open(f'{opts.output_dir}/results/'
                                  f'{split}_results_{global_step}_'
                                  f'rank{rank}.csv', 'w') as f:
                            if split != 'test':
                                f.write("id,proba,label,target\n")
                                for id_, pred, prob, label in results:
                                    padded_id = id_
                                    if len(id_) == 4:
                                        padded_id = "0" + str(id_)
                                    f.write(f'{padded_id},{str(np.round(prob, 4))},{pred},{label}\n')
                            else:
                                f.write("id,proba,label\n")
                                for id_, pred, prob in results:
                                    padded_id = id_
                                    if len(id_) == 4:
                                        padded_id = "0" + str(id_)
                                    f.write(f'{padded_id},{str(np.round(prob, 4))},{pred}\n')

                        with open(f'{opts.output_dir}/results/'
                                  f'{split}_results_logits_{global_step}_'
                                  f'rank{rank}.csv', 'w') as f:
                            if split != 'test':
                                f.write("id,proba,label,target,logit_0,logit_1\n")
                                for id_, pred, prob, label, logits in results_logits:
                                    padded_id = id_
                                    if len(id_) == 4:
                                        padded_id = "0" + str(id_)
                                    f.write(f'{padded_id},{str(np.round(prob, 4))},{pred},{label},{logits[0]},{logits[1]}\n')
                            else:
                                f.write("id,proba,label,logit_0,logit_1\n")
                                for id_, pred, prob, logits in results_logits:
                                    padded_id = id_
                                    if len(id_) == 4:
                                        padded_id = "0" + str(id_)
                                    f.write(f'{padded_id},{str(np.round(prob, 4))},{pred},{logits[0]},{logits[1]}\n')

                        TB_LOGGER.log_scaler_dict(log)
                    model_saver.save(model, global_step)
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        if global_step % opts.valid_steps == 0:
            LOGGER.info(f"Step {global_step}: finished {n_epoch} epochs")
    for split, loader in [('val', val_dataloader), ('test', test_dataloader)]:
        LOGGER.info(f"Step {global_step}: start running "
                    f"validation on {split} split...")
        log, results, results_logits = validate(model, loader, split)
        with open(f'{opts.output_dir}/results/'
                  f'{split}_results_{global_step}_'
                  f'rank{rank}_final.csv', 'w') as f:
            if split != 'test':
                f.write("id,proba,label,target\n")
                for id_, pred, prob, label in results:
                    padded_id = id_
                    if len(id_) == 4:
                        padded_id = "0" + str(id_)
                    f.write(f'{padded_id},{str(np.round(prob, 4))},{pred},{label}\n')
            else:
                f.write("id,proba,label\n")
                for id_, pred, prob in results:
                    padded_id = id_
                    if len(id_) == 4:
                        padded_id = "0" + str(id_)
                    f.write(f'{padded_id},{str(np.round(prob, 4))},{pred}\n')
        TB_LOGGER.log_scaler_dict(log)
    model_saver.save(model, f'{global_step}_final')


@torch.no_grad()
def validate(model, val_loader, split):
    model.eval()
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    results = []
    results_logits = []
    test_mode = (split == 'test')
    for i, batch in enumerate(val_loader):
        img_ids = batch['img_ids']
        if not test_mode:
            targets = batch['targets']
            del batch['targets']
        del batch['img_ids']
        scores = model(**batch, targets=None, compute_loss=False)
        if not test_mode:
            # loss = F.binary_cross_entropy_with_logits(scores, targets.to(dtype=scores.dtype), reduction='sum')
            loss = F.cross_entropy(scores, targets, reduction='sum')
            val_loss += loss.item()
            tot_score += (scores.max(dim=-1, keepdim=False)[1] == targets).sum().item()
        predictions = scores.max(dim=-1, keepdim=False)[1].cpu().tolist()
        probs = F.softmax(scores, dim=1)[:, 1].cpu().tolist()
        logits = scores.cpu().tolist()
        #     tot_score += ((scores > 0.5).to(dtype=targets.dtype) == targets).sum().item()
        # probs = torch.sigmoid(scores).cpu().tolist()
        # predictions = [1 if prob > 0.5 else 0 for prob in probs]
        if not test_mode:
            labels = targets.cpu().tolist()
            results.extend(zip(img_ids, predictions, probs, labels))
            results_logits.extend(zip(img_ids, predictions, probs, labels, logits))
        else:
            results.extend(zip(img_ids, predictions, probs))
            results_logits.extend(zip(img_ids, predictions, probs, logits))
        n_ex += len(img_ids)
    if not test_mode:
        val_loss = sum(all_gather_list(val_loss))
        tot_score = sum(all_gather_list(tot_score))
        n_ex = sum(all_gather_list(n_ex))
        tot_time = time()-st
        val_loss /= n_ex
        val_acc = tot_score / n_ex

        res_img_ids, res_predictions, res_probs, res_labels = list(zip(*results))
        val_auroc = roc_auc_score(res_labels, res_probs)

        val_log = {f'valid/{split}_loss': val_loss,
                   f'valid/{split}_acc': val_acc,
                   f'valid/{split}_AUROC': val_auroc,
                   f'valid/{split}_ex_per_s': n_ex/tot_time}
        model.train()
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"val Acc: {val_acc:.4f}, val AUROC: {val_auroc:.4f}")
        return val_log, results, results_logits
    else:
        return {f'generate/{split}': 1}, results, results_logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--root_path",
                        default=None, type=str,
                        help="The root path of the project.")
    parser.add_argument("--dataset_path",
                        default=None, type=str,
                        help="The path to the dataset files train/dev/test.")
    parser.add_argument("--train_image_set",
                        default=None, type=str,
                        help="The input train images tsv file name.")
    parser.add_argument("--val_image_set",
                        default=None, type=str,
                        help="The input validation images tsv file name.")
    parser.add_argument("--test_image_set",
                        default=None, type=str,
                        help="The input test images tsv file name.")
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")
    parser.add_argument("--model", default='paired',
                        choices=['paired', 'paired-attn'],
                        help="choose from 2 model architecture")
    parser.add_argument('--use_img_type', action='store_true',
                        help="expand the type embedding for 2 image types")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size",
                        default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size",
                        default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=8.00e-7,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps",
                        default=1000,
                        type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps",
                        default=100000,
                        type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+', type=float,
                        help="beta for adam optimizer")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm",
                        default=0.25,
                        type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps",
                        default=4000,
                        type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
