import numpy as np
import copy
import datetime
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
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from utils.vocab import create_vocab
from utils.eval import calculate_accuracy
from utils.wrap_var import to_var
from train.CL.parser import preprocess_config
from utils.gameplayutils import *
from utils.model_loading import load_model

from utils.datasets.CL.RndObjSampDataset import RndObjSampDataset # For Guesser Accuracy training
from utils.datasets.CL.QGenDataset import QGenDataset #For SL QGen training
from utils.datasets.GamePlay.GamePlayDataset import GamePlayDataset # For validation
from utils.datasets.SL.N2NDataset import N2NDataset # For aux games

from models.Oracle import Oracle
from models.Ensemble import Ensemble
# from models.N2N.CNN import ResNet

from train.CL.gameplay import gameplay_fwpass
from train.CL.qgen import qgen_fwpass
from train.CL.auxloss import aux_loss

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

NUM_SCRIPTS = 4 # 2 times the actual number of scripts to be safe

def cycle(iterable):
    # idea courtesy of:
    # https://discuss.pytorch.org/t/in-what-condition-the-dataloader-would-raise-stopiteration/17483/2
    while True:
        for x in iterable:
            yield x

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/CL/config.json", help=' General config file')
    parser.add_argument("-ens_config", type=str, default="config/CL/ensemble.json", help=' Ensemble config file')
    parser.add_argument("-or_config", type=str, default="config/CL/oracle.json", help=' Oracle config file')
    parser.add_argument("-modulo", type=int, default=5, help='This flag will cause the guesser to be updated every modulo number of epochs. If this flag is on then automatically epoch flag will overridden to 0') # TODO update help
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("-eval_newobj", action='store_true', help='To evaluate new object score for the model')
    parser.add_argument("-my_cpu", action='store_true', help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-breaking", action='store_true',
        help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-dataparallel", action='store_true', help='This for model files which     were saved with Dataparallel')

    # my args
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("-aux_data", action='store_true', 
        help='Flag to include human data when training guesser.')
    parser.add_argument("-adv_weight", type=float, default=1.0)
    # parser.add_argument('-beta', type=float, default=0.0)
    parser.add_argument("-guesser_modulo", type=int, default=1, help='This flag will cause the normal data'
        ' to be included every modulo number of epochs.')
    parser.add_argument("-aux_modulo", type=int, default=1, help='This flag will cause the human data'
        ' to be included every modulo number of epochs.')
    parser.add_argument("-adversarial", action='store_true',
        help="failed exp., not used for AACL2022 (email authors if interested)")

    args = parser.parse_args()
    print(args.exp_name)

    use_dataparallel = args.dataparallel
    breaking = args.breaking
    use_aux_guesser_data = args.aux_data

    # ALPHA = 1. - args.beta
    # BETA = args.beta
    
    if use_aux_guesser_data:
        aux_loss_fn = aux_loss
    else:
        aux_loss_fn = None

    ensemble_args, dataset_args, optimizer_args, exp_config, oracle_args, word2i = preprocess_config(args)

    pad_token= word2i['<padding>']

    if args.seed is not None:
        exp_config['seed'] = args.seed

    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    if exp_config['logging']:
        log_dir = exp_config['logdir']+str(args.exp_name)+exp_config['ts']+'/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        copy2(args.config, log_dir)
        copy2(args.ens_config, log_dir)
        copy2(args.or_config, log_dir)
        with open(log_dir+'args.txt', 'w') as f:
            f.write(str(vars(args))) # converting args.namespace to dict

    if exp_config['save_models']:
        model_dir = exp_config['save_models_path'] + args.bin_name + exp_config['ts'] + '/'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        # This is again duplicate just for bookkeeping multiple times
        copy2(args.config, model_dir)
        copy2(args.ens_config, model_dir)
        copy2(args.or_config, model_dir)
        with open(model_dir+'args.txt', 'w') as f:
            f.write(str(vars(args))) # converting args.namespace to dict

    model = Ensemble(**ensemble_args)
    model = load_model(model, ensemble_args['bin_file'], use_dataparallel=use_dataparallel)
    if args.adversarial:
        aux_guesser = copy.deepcopy(model.guesser)
    else:
        aux_guesser = None

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

    oracle = load_model(oracle, oracle_args['bin_file'], use_dataparallel=use_dataparallel)
    oracle.eval()

    softmax = nn.Softmax(dim=-1)

    #For Guesser
    guesser_loss_function = nn.CrossEntropyLoss() #For Guesser
    # For QGen.
    _cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    #TODO: Decider

    if use_cuda:
        model = model.cuda()
        if aux_guesser is not None:
            aux_guesser = aux_guesser.cuda()

    if use_dataparallel:
        encoder_optim = optim.Adam(model.module.encoder.parameters(), optimizer_args['lr'])
        decider_optim = optim.Adam(model.module.decider.parameters(), optimizer_args['lr'])
        guesser_optim = optim.Adam(model.module.guesser.parameters(), optimizer_args['lr'])
        qgen_optim = optim.Adam(model.module.qgen.parameters(), optimizer_args['lr'])
        if aux_guesser is not None:
            raise NotImplementedError('Not implementing data parralell')
    else:
        encoder_optim = optim.Adam(model.encoder.parameters(), optimizer_args['lr'])
        decider_optim = optim.Adam(model.decider.parameters(), optimizer_args['lr'])
        guesser_optim = optim.Adam(model.guesser.parameters(), optimizer_args['lr'])
        qgen_optim = optim.Adam(model.qgen.parameters(), optimizer_args['lr'])
        if aux_guesser is not None:
            aux_guesser_optim = optim.Adam(aux_guesser.parameters(), optimizer_args['lr'])

    # Guesser dataset based on the Random Object selection
    dataset_guesser = RndObjSampDataset(split='train', **dataset_args)
    # QGen dataset using the GT data
    dataset_qgen = QGenDataset(split='train', **dataset_args)
    # Human game dataset
    dataset_aux = N2NDataset(split='train', full_dialogs_only=True,
        **dataset_args)

    # Validation data on the gameplay data
    dataset_val_gp = GamePlayDataset(split='val', **dataset_args)

    # TODO visualisation intit
    max_epochs = optimizer_args['no_epochs']
    for epoch in range(max_epochs):

        start = time()
        print('epoch', epoch)
        # Condition for guesser and QGen

        gu_dataloader = DataLoader(
        dataset= dataset_guesser,
        batch_size=optimizer_args['batch_size'],
        shuffle=True,
        num_workers= 1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count()//NUM_SCRIPTS,
        pin_memory= use_cuda,
        drop_last=False)

        qgen_dataloader = DataLoader(
        dataset= dataset_qgen,
        batch_size=optimizer_args['batch_size'],
        shuffle=True,
        num_workers= 1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count()//NUM_SCRIPTS,
        pin_memory= use_cuda,
        drop_last=False)

        aux_dataloader = DataLoader(
        dataset= dataset_aux,
        batch_size=optimizer_args['batch_size'],
        shuffle=True,
        num_workers= 1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count()//NUM_SCRIPTS,
        pin_memory= use_cuda,
        drop_last=False)

        aux_dataloader = iter(cycle(aux_dataloader))

        gp_dataloader = DataLoader(
        dataset=dataset_val_gp,
        batch_size=optimizer_args['batch_size'],
        shuffle=False, # If using this code for RL training make shuffle true
        num_workers= 1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count()//NUM_SCRIPTS,
        pin_memory= use_cuda,
        drop_last=False)

        modulo_value = (epoch % args.modulo == 0)
        aux_modulo_value = use_aux_guesser_data and (epoch % args.aux_modulo == 0)
        guesser_modulo_value = (epoch % args.guesser_modulo == 0)

        if modulo_value:
            train_dataloader = qgen_dataloader
        else:
            train_dataloader = gu_dataloader

        if args.eval_newobj:
            train_dataloader = gu_dataloader
            modulo_value = False
            if epoch > 4:
                break

        # cmd logging
        train_qgen_loss = []
        train_guesser_loss = []
        train_aux_loss = []
        training_guesser_accuracy = list()
        val_gameplay_accuray = list()

        for split, dataloader in zip(['train', 'val'], [train_dataloader, gp_dataloader]):

            if split == 'train':
                model.train()
            else:
                model.eval()

            if args.eval_newobj and split == 'val':
                break

            for i_batch, sample in enumerate(dataloader):

                if i_batch > 5 and breaking:
                    print('Breaking after processing 4 batch')
                    break

                for k, v in sample.items(): # redunant ? -> no implictly sends vars to cuda
                    if torch.is_tensor(v):
                        sample[k] = to_var(v)

                if modulo_value and split == 'train':

                    qgen_out = qgen_fwpass(q_model= model, inputs= sample, use_dataparallel= use_dataparallel)
                    word_logits_loss = _cross_entropy(qgen_out.view(-1, 4901), sample['target_q'].view(-1)) #TODO remove this hardcoded number

                    # Backprop
                    encoder_optim.zero_grad()
                    qgen_optim.zero_grad()
                    word_logits_loss.backward()
                    encoder_optim.step()
                    qgen_optim.step()
                    train_qgen_loss.append(word_logits_loss.data.cpu().item())
                else:
                    # Guesser GamePlay training
                    # TODO better this one
                    sample['pad_token'] = pad_token
                    exp_config['max_no_qs'] = dataset_args['max_no_qs']
                    exp_config['use_dataparallel'] = use_dataparallel

                    guesser_logits = gameplay_fwpass(q_model= model, o_model= oracle, inputs= sample, exp_config= exp_config, word2i= word2i)

                    guesser_loss = guesser_loss_function(guesser_logits*sample['objects_mask'].float(), sample['target_obj'])
                    guesser_accuracy = calculate_accuracy(softmax(guesser_logits)*sample['objects_mask'].float(), sample['target_obj'])

                    if split == 'train':

                        if not args.eval_newobj and guesser_modulo_value: # and ALPHA > 0:
                            encoder_optim.zero_grad()
                            guesser_optim.zero_grad()
                            guesser_loss.backward()
                            encoder_optim.step()
                            guesser_optim.step()

                        train_guesser_loss.append(guesser_loss.data.cpu().item())
                        training_guesser_accuracy.append(guesser_accuracy.data.cpu().item())

                        # do some fun aux training things
                        if not args.eval_newobj and aux_modulo_value: # and use_aux_guesser_data:
                            aux_sample = next(aux_dataloader)
                            for k, v in aux_sample.items(): # redunant ? -> no implictly sends vars to cuda
                                if torch.is_tensor(v):
                                    aux_sample[k] = to_var(v)
                            progress = epoch / max_epochs
                            if args.adversarial:
                                aloss = aux_loss_fn(model, aux_sample, adversarial=True,
                                    beta=args.adv_weight, progress=progress, aux_guesser=aux_guesser, oracle=oracle, 
                                    gen_sample=sample, exp_config=exp_config, word2i=word2i)
                            else:
                                aloss = aux_loss_fn(model, aux_sample, adversarial=False, 
                                    beta=1.0)
                            encoder_optim.zero_grad()
                            if aux_guesser is not None: aux_guesser_optim.zero_grad()
                            aloss.backward()
                            encoder_optim.step()
                            if aux_guesser is not None: aux_guesser_optim.step()
                            try:
                                train_aux_loss.append(aloss.data.cpu().item())
                            except:
                                train_aux_loss.append(aloss)
                    else:
                        val_gameplay_accuray.append(guesser_accuracy.data.cpu().item())

        if exp_config['save_models'] and not args.eval_newobj:
            model_file = os.path.join(model_dir, ''.join(['model_ensemble_addnTrain_', args.bin_name,'_E_', str(epoch)]))
            torch.save(model.state_dict(), model_file)

        print("Epoch %03d, Time taken %.3f"%(epoch, time()-start))
        if modulo_value:
            print("Training Loss:: QGen %.3f"%(np.mean(train_qgen_loss)))
        else:
            mean_aux_loss = np.mean(train_aux_loss) if len(train_aux_loss) > 0 else -1.
            print("Training Guesser:: Loss %.3f, Aux %.3f, Accuracy %.5f"%(np.mean(train_guesser_loss), 
                mean_aux_loss, np.mean(training_guesser_accuracy)))
        print("Validation GP Accuracy:: %.5f"%(np.mean(val_gameplay_accuray)))
        if exp_config['save_models'] and not args.eval_newobj:
            print("Saved model to %s" % (model_file))
        print('-----------------------------------------------------------------')
        # GamePlay validation score
