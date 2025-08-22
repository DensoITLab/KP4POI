import argparse
import collections
import os
import random
from pathlib import Path
import logging
from datetime import datetime
from packaging import version

#import torch_npu
#from torch_npu.contrib import transfer_to_npu

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from pprint import pprint

from param import get_optimizer
from pretrain_data import get_loader
from utils import load_state_dict,LossMeter
from dist_utils import reduce_dict

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        assert args.whole_word_embed
        from pretrain_model import P5Pretraining

        model_kwargs = {}
        model_class = P5Pretraining

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config, **model_kwargs)

        if 'p5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = 0
        if args.load is not None:
            if ',' in args.load:
                t = args.load.split(',')
                args.load = t[0]
                self.start_epoch = int(t[1]) + 1
            else:
                self.start_epoch = int(args.load.split('Epoch-')[-1]) + 1
            logging.info('loading model from {}; start epoch is {}'.format(args.load+'.pth',self.start_epoch))
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.cuda(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
            if args.load is not None and os.path.isfile(args.load + '_opt.pth'):
                logging.info('loading optimizer from {}'.format(args.load + '_opt.pth'))
                self.optim.load_state_dict((torch.load(args.load + '_opt.pth')))

        if args.fp16:
            from apex import amp
            self.amp_scale_loss = amp.scale_loss
            self.amp_master_params = amp.master_params


        if args.multiGPU:
            if args.distributed:
                if args.fp16:
                    self.model, self.optim = amp.initialize(self.model, self.optim, opt_level='O2', loss_scale=128.0)
                self.model = DDP(self.model, device_ids=[args.gpu])
        else:
            if args.fp16:
                self.model, self.optim = amp.initialize(self.model, self.optim) #, combine_grad=True)
            
        if self.verbose:
            logging.info(f'It took {time() - start:.1f}s')

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        best_eval_loss = 100000.
        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

            if 't5' in self.args.backbone:
                project_name = "P5_Pretrain"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        # forward scheduler
        if self.lr_scheduler:
            for __ in range(0,self.start_epoch*len(self.train_loader)):
                self.lr_scheduler.step()

        global_step = 0
        # endure_count = 0
        for epoch in range(self.start_epoch,self.args.epoch):
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            if self.verbose:
                logging.info(now_time() + 'Epoch: {}'.format(epoch))
            # Train
            self.model.train()
            # logging.info(self.model)


            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):

                if self.args.distributed:
                    results = self.model.module.train_step(batch)
                else:
                    results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16:
                    with self.amp_scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(self.amp_master_params(self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            if self.args.distributed:
                dist.barrier()

            results = reduce_dict(epoch_results, average=False)
            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}, lr: {lr:.6f} \n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                losses_str += '\n'
                logging.info(now_time() + losses_str)

            if self.args.distributed:
                dist.barrier()

            if (epoch + 1) % self.args.eval_interval == 0:
                # Validation
                valid_results = self.evaluate_epoch(epoch=epoch)

                valid_results = reduce_dict(valid_results, average=False)
                
                if self.verbose:
                    valid_loss = valid_results['total_loss']
                    valid_loss_count = valid_results['total_loss_count']

                    avg_valid_loss = valid_loss / valid_loss_count
                    losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                    for name, loss in valid_results.items():
                        if name[-4:] == 'loss':
                            loss_count = int(valid_results[name+'_count'])
                            if loss_count > 0:
                                avg_loss = loss / loss_count
                                losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                    losses_str += '\n'
                    logging.info(now_time() + losses_str)

                if self.args.distributed:
                    dist.barrier()

                if self.verbose:
                    # Save
                    if avg_valid_loss < best_eval_loss:
                        best_eval_loss = avg_valid_loss
                    
                        self.save("BEST_EVAL_LOSS")
                        logging.info('==== best eval loss {}, epoch {} ==== \n'.format(best_eval_loss, epoch))

                if self.args.distributed:
                    dist.barrier()

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=275)

            for step_i, batch in enumerate(self.val_loader):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Valid Epoch {epoch} |'
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()
            if self.args.distributed:
                dist.barrier()

            return epoch_results

def now_time():
    return '[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '

def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    train_task_list = {
    # 'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
    'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']
    }

    train_sample_numbers = {'rating': 1, 'sequential': 1, 'explanation': 1, 'review': 1, 'traditional': (10, 5)}
    train_loader = get_loader(
        args,
        train_task_list,
        train_sample_numbers,
        split=args.train, 
        mode='train',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )

    print(f'Building val loader at GPU {gpu}')
    val_task_list = {
    # 'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
    'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10'],
    }
    val_sample_numbers = {'rating': 1, 'sequential': 1, 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
    val_loader = get_loader(
        args,
        val_task_list,
        val_sample_numbers,
        split=args.valid, 
        mode='val',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )

    trainer = Trainer(args, train_loader, val_loader, train=True)
    trainer.train()

def parse_args(parse=True, parser=None, **optional_kwargs):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    # Data Splits
    parser.add_argument("--data_url", default='data')
    parser.add_argument("--train", default='books')
    parser.add_argument("--train_url", default='out')
    parser.add_argument("--valid", default='books')
    parser.add_argument("--test", default=None)
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--submit', action='store_true')

    # Checkpoint
    parser.add_argument('--log_url', type=str, default='log')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model) or resume training (model ID,epoch).')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_true')
    parser.add_argument('--fp16', action='store_false')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=160, type=int)
    parser.add_argument('--local_rank', type=int, default=0)

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-small')
    parser.add_argument('--tokenizer', type=str, default='p5')
    parser.add_argument('--whole_word_embed', action='store_false')

    parser.add_argument('--max_text_length', type=int, default=512)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument("--losses", default='sequential', type=str)
    parser.add_argument('--log_train_accuracy', action='store_true')

    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--degree', type=int, default=8, help='relation number of kg')
    parser.add_argument('--hop_num', type=int, default=2, help='number of kg hop')
    parser.add_argument('--max_items', type=int, default=5, help='number of items')

    parser.add_argument("--test_random", action='store_false')
    parser.add_argument("--tree_mask", action='store_true')
    parser.add_argument("--cross_mask", action='store_true')

    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--gen_max_length', type=int, default=64)

    # Data
    parser.add_argument('--do_lower_case', action='store_true')

    # Etc.
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        # args = parser.parse_known_args()[0]
        args = parser.parse_args(args=[])
 
    # Namespace => Dictionary
    #kwargs = vars(args)
    #kwargs.update(optional_kwargs)

    #args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    return args


if __name__ == "__main__":
    #torch_npu.npu.set_compile_mode(jit_compile=False)
    cudnn.benchmark = True
    args = parse_args(parse=True,parser=None)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    
    args.output_path = os.path.join(args.train_url, args.train + '_' + args.backbone + args.prefix)
    if args.local_rank in [0, -1]:
        os.makedirs(args.output_path, exist_ok=True)
        if args.load is not None:
            logging.basicConfig(filename = os.path.join(args.output_path, "train.log"), filemode = 'a', level = logging.INFO)
        else:
            logging.basicConfig(filename = os.path.join(args.output_path, "train.log"), filemode = 'w', level = logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(now_time() + 'begin work')
        logging.info(args)

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        logging.info(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')

    args.LOSSES_NAME = LOSSES_NAME

    main_worker(args.local_rank, args)
