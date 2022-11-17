from copy import copy, deepcopy
import numpy as np
import json
import argparse
import os
import multiprocessing
from time import time
from shutil import copy2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from utils.vocab import create_vocab
from utils.eval import calculate_accuracy, calculate_agreement
from utils.model_loading import load_model
from utils.gameplayutils import *
from train.GamePlay.parser import preprocess_config

from utils.datasets.SL.N2NDataset import N2NDataset
from utils.datasets.SL.N2NResNetDataset import N2NResNetDataset

from models.Oracle import Oracle
from models.Ensemble import Ensemble
from models.CNN import ResNet

from train.GamePlay.stochastic import make_stochastic
from train.GamePlay.stochastic import sample as sample_model

NUM_SCRIPTS = 6 # 2 times the actual number of scripts to be safe

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

#TODO: Move this code from the train folder

SIGMA = 0.01

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/GamePlay/config.json", help=' General config file')
    parser.add_argument("-ens_config", type=str, default="config/GamePlay/ensemble.json", help=' Ensemble config file')
    parser.add_argument("-or_config", type=str, default="config/GamePlay/oracle.json", help=' Oracle config file')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-my_cpu", action='store_true', help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-breaking", action='store_true',
                        help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-resnet", action='store_true', help='This flag will cause the program to use the image features from the ResNet forward pass instead of the precomputed ones.')
    parser.add_argument("-dataparallel", action='store_true', help='This for model files which were saved with Dataparallel')
    parser.add_argument("-log_enchidden", action='store_true', help='This flag saves the encoder hidden state. WARNING!!! This might cause the resulting json file to blow up!')

    # --------Arguments from config.json that can be overridden here. Similar changes have to be made in the util file and not here--------------------
    parser.add_argument("-batch_size", type=int, help='Batch size for the gameplay')
    parser.add_argument("-load_bin_path", type=str, help='Bin file path for the saved model. If this is not given then one provided in ensemble.json will be taken ')

    args = parser.parse_args()
    print(args.exp_name)

    # Load the Arguments and Hyperparamters
    ensemble_args, dataset_args, optimizer_args, exp_config, oracle_args, word2i, i2word, catid2str = preprocess_config(args)
    dataset_args['my_cpu'] = False
    pad_token= word2i['<padding>']

    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    if exp_config['logging']:
        log_dir = exp_config['logdir']+str(args.exp_name)+exp_config['ts']+'/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        copy2(args.config, log_dir)
        copy2(args.ens_config, log_dir)
        copy2(args.or_config, log_dir)
        with open(log_dir+'args.txt', 'w') as f:
            f.write(str(vars(args))) # converting args.namespace to dict

    model = Ensemble(**ensemble_args)
    model = load_model(model, ensemble_args['bin_file'], use_dataparallel=False)
    model.eval()

    oracle = Oracle(
        no_words            = oracle_args['vocab_size'],
        no_words_feat       = oracle_args['embeddings']['no_words_feat'],
        no_categories       = oracle_args['embeddings']['no_categories'],
        no_category_feat    = oracle_args['embeddings']['no_category_feat'],
        no_hidden_encoder   = oracle_args['lstm']['no_hidden_encoder'],
        mlp_layer_sizes     = oracle_args['mlp']['layer_sizes'],
        no_visual_feat      = oracle_args['inputs']['no_visual_feat'],
        no_crop_feat        = oracle_args['inputs']['no_crop_feat'],
        dropout             = oracle_args['lstm']['dropout'],
        inputs_config       = oracle_args['inputs'],
        scale_visual_to     = oracle_args['inputs']['scale_visual_to']
        )

    oracle = load_model(oracle, oracle_args['bin_file'], use_dataparallel=False)
    oracle.eval()

    print(model)
    print(oracle)

    if args.resnet:
        # This was for the new image case, we don't use it
        # Takes too much time.
        dataset_test = N2NResNetDataset(split='test', **dataset_args)
    else:
        dataset_test = N2NDataset(split='test', **dataset_args)
    
    eval_log = dict()
    
    dataloader = DataLoader(
            dataset=dataset_test,
            batch_size=optimizer_args['batch_size'],
            shuffle=True,
            num_workers=1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count() // NUM_SCRIPTS,
            pin_memory= use_cuda,
            drop_last=False)
    
    softmax = nn.Softmax(dim=-1)
    accuracy = []

    for i_batch, sample in enumerate(dataloader):
        # avg_img_features = sample['image']
        for k, v in sample.items():
            if torch.is_tensor(v):
                sample[k] = to_var(v, False)
        with torch.no_grad():
            encoder_hidden = model.encoder(history=sample['history'], visual_features=sample['image'],
                history_len=sample['history_len'])
            guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], 
                objects=sample['objects'], regress= False)
            batch_accuracy = calculate_accuracy(softmax(guesser_logits*sample['objects_mask'].float()), sample['target_obj'])
            accuracy.append(batch_accuracy.cpu().item())
            for bidx, x in enumerate(encoder_hidden):
                eval_log[sample['game_id'][bidx]] = dict()
                eval_log[sample['game_id'][bidx]]['enc_hidden'] = x.data.tolist()
    
    print(sum(accuracy) / len(accuracy))
    file_name = log_dir+'test'+'_GPinference_'+str(args.exp_name)+'_'+exp_config['ts']+'.json'
    with open(file_name, 'w') as f:
        json.dump(eval_log, f)