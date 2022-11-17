import numpy as np
import datetime
import json
# import progressbar
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

from train.SL.parser import preprocess_config
from train.SL.vis import Visualise
from utils.wrap_var import to_var
from utils.eval import calculate_accuracy

from utils.datasets.SL.N2NDataset import N2NDataset
from utils.datasets.SL.N2NResNetDataset import N2NResNetDataset
from models.Ensemble import Ensemble
from models.CNN import ResNet

# TODO Make this capitalised everywhere to inform it is a global variable
# use typically
use_cuda = torch.cuda.is_available()
# # use when testing
# use_cuda = False 

NUM_SCRIPTS = 4 # 2 times the actual number of scripts to be safe

# helper functions to fix bugs found in different pytorch version
def safe_mean(arr):
    # tensor on gpu safe
    x = sum(arr) / len(arr)
    try:
        return x.cpu().item()
    except AttributeError:
        return x

def safe_mask(tensor, mask):
    if mask.dim() == 0:
        return tensor[mask].unsqueeze(0)
    else:
        return tensor[mask]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/SL/config.json", help='Config file')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("-my_cpu", action='store_true', help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-breaking", action='store_true', help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-resnet", action='store_true', help='This flag will cause the program to use the image features from the ResNet forward pass instead of the precomputed ones.')
    parser.add_argument("-modulo", type=int, default=1, help='This flag will cause the guesser to be updated every modulo number of epochs')
    parser.add_argument("-no_decider", action='store_true', help='This flag will cause the decider to be turned off')

    args = parser.parse_args()
    print(args.exp_name)

    breaking = args.breaking
    # Load the Arguments and Hyperparamters
    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

    if exp_config['save_models']:
        model_dir = exp_config['save_models_path'] + args.bin_name + exp_config['ts'] + '/'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        # Copying config file for book keeping
        copy2(args.config, model_dir)
        with open(model_dir+'args.json', 'w') as f:
            json.dump(vars(args), f) # converting args.namespace to dict

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    # Init Model
    model = Ensemble(**ensemble_args)
    # TODO Checkpoint loading

    if use_cuda:
        # set cuda visible devices to specify device
        model.cuda()
        # model = DataParallel(model)
    print(model)

    if args.resnet:
        cnn = ResNet()

        if use_cuda:
            cnn.cuda()
            # cnn = DataParallel(cnn)

    softmax = nn.Softmax(dim=-1)

    # Loss Function and Optimizer

    #For Guesser
    guesser_loss_function = nn.CrossEntropyLoss()

    #For Decider
    decider_cross_entropy = nn.CrossEntropyLoss(size_average=False)

    # For QGen.
    _cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    # TODO Use different optimizers for different modules if required.
    optimizer = optim.Adam(model.parameters(), optimizer_args['lr'])

    if args.resnet:
        #This was for the new image case, we don't use it
        #Takes too much time.
        dataset_train = N2NResNetDataset(split='train', **dataset_args)
        dataset_val = N2NResNetDataset(split='val', **dataset_args)
    else:
        dataset_train = N2NDataset(split='train', **dataset_args)
        dataset_val = N2NDataset(split='val', **dataset_args)

    if exp_config['logging']:
        exp_config['model_name'] = 'ensemble'
        exp_config['model'] = str(model)
        exp_config['train_dataset_len'] = str(len(dataset_train))
        exp_config['valid_dataset_len'] = str(len(dataset_val))
        exp_config['modulo'] = True if args.modulo>1 else False
        visualise = Visualise(**exp_config)

    for epoch in range(optimizer_args['no_epochs']):

        start = time()
        print('epoch', epoch)

        #Logging
        train_decision_loss = float_tensor()
        val_decision_loss = float_tensor()
        train_qgen_loss = float_tensor()
        val_qgen_loss = float_tensor()
        train_guesser_loss = float_tensor()
        val_guesser_loss = float_tensor()
        train_total_loss = float_tensor()
        val_total_loss = float_tensor()

        training_guesser_accuracy = list()
        validation_guesser_accuracy = list()
        training_ask_accuracy = list()
        training_guess_accuracy = list()
        validation_ask_accuracy = list()
        validation_guess_accuracy = list()

        for split, dataset in zip(exp_config['splits'], [dataset_train, dataset_val]):

            dataloader = DataLoader(
            dataset=dataset,
            batch_size=optimizer_args['batch_size'],
            shuffle=True,
            num_workers=1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count() // NUM_SCRIPTS,
            pin_memory= use_cuda,
            drop_last=False)

            if split == 'train':
                model.train()
            else:
                model.eval()

            for i_batch, sample in enumerate(dataloader):

                sample['tgt_len'], ind = torch.sort(sample['tgt_len'], 0, descending=True)
                batch_size = ind.size(0)

                # Get Batch
                for k, v in sample.items():
                    if k == 'tgt_len':
                        sample[k] = to_var(v)
                    elif torch.is_tensor(v):
                        sample[k] = to_var(v[ind])

                if args.resnet:
                    # This is done so that during backprop the gradients dont flow through the ResNet
                    img_features, avg_img_features = cnn(to_var(sample['image'].data, True))
                    img_features, avg_img_features = to_var(img_features.data), to_var(avg_img_features.data)
                else:
                    avg_img_features = sample['image']

                # Masking w.r.t decider_tgt
                masks = list()
                mask1 = sample['decider_tgt'].data

                if torch.sum(mask1) >= 1:
                    masks.append(torch.nonzero(1-mask1))
                    masks.append(torch.nonzero(mask1))
                else:
                    masks.append(torch.nonzero(1-mask1))

                word_logits_loss = to_var(torch.zeros(1))
                guesser_loss = to_var(torch.zeros(1))
                decider_loss = to_var(torch.zeros(1))

                decider_accuracy = 0
                ask_accuracy = 0
                guess_accuracy = 0
                guesser_accuracy = 0

                for idx, mask in enumerate(masks):
                    # When all elements belongs to QGen or Guess only
                    if len(mask) <= 0:
                        continue
                    mask = mask.squeeze()

                    if idx == 0:
                        # decision, word_logits
                        decider_out, qgen_out = model(
                            history=safe_mask(sample['history'], mask),
                            history_len=safe_mask(sample['history_len'], mask),
                            src_q=safe_mask(sample['src_q'], mask),
                            tgt_len=safe_mask(sample['tgt_len'], mask),
                            visual_features=safe_mask(avg_img_features, mask),
                            spatials=safe_mask(sample['spatials'],mask),
                            objects=safe_mask(sample['objects'],mask),
                            mask_select=idx,
                            target_cat=safe_mask(sample['target_cat'], mask))

                        word_logits_loss += _cross_entropy(qgen_out.view(-1, 4901), 
                            sample['target_q'][mask].view(-1)) #TODO remove this hardcoded number

                        decider_loss +=  ensemble_args['decider']['ask_weight'] \
                            * decider_cross_entropy(decider_out.squeeze(1), safe_mask(sample['decider_tgt'],mask))
                        ask_accuracy = calculate_accuracy(decider_out.squeeze(1), 
                            safe_mask(sample['decider_tgt'], mask))

                    elif idx == 1:
                        if epoch%args.modulo != 0:
                            continue
                        else:
                            # decision, guesser_out
                            decider_out, guesser_out = model(
                                history=safe_mask(sample['history'], mask),
                                history_len=safe_mask(sample['history_len'], mask),
                                src_q=safe_mask(sample['src_q'], mask),
                                tgt_len=safe_mask(sample['tgt_len'], mask),
                                visual_features=safe_mask(avg_img_features, mask),
                                spatials=safe_mask(sample['spatials'], mask),
                                objects=safe_mask(sample['objects'], mask),
                                mask_select=idx,
                                target_cat=safe_mask(sample['target_cat'], mask))

                            decider_loss +=  ensemble_args['decider']['guess_weight'] \
                                * decider_cross_entropy(decider_out.squeeze(1), safe_mask(sample['decider_tgt'], mask))
                            guess_accuracy = calculate_accuracy(decider_out.squeeze(1), 
                                safe_mask(sample['decider_tgt'], mask))

                            guesser_loss += guesser_loss_function(
                                guesser_out * safe_mask(sample['objects_mask'], mask).float(), 
                                safe_mask(sample['target_obj'], mask))
                            guesser_accuracy = calculate_accuracy(softmax(guesser_out), 
                                safe_mask(sample['target_obj'], masks[1].squeeze())) # masks[1].squeeze() == mask

                if epoch%args.modulo != 0:
                    loss = word_logits_loss
                else:
                    if args.no_decider:
                        loss = guesser_loss + word_logits_loss
                    else:
                        loss = guesser_loss + word_logits_loss + decider_loss/batch_size

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Logging variables
                    if epoch%args.modulo != 0:
                        train_qgen_loss = torch.cat([train_qgen_loss, word_logits_loss.data])
                        train_decision_loss = torch.cat([train_decision_loss, decider_loss.data/batch_size])
                        training_guess_accuracy.append(guess_accuracy)
                        training_ask_accuracy.append(ask_accuracy)
                    else:
                        training_guesser_accuracy.append(guesser_accuracy)
                        training_ask_accuracy.append(ask_accuracy)
                        training_guess_accuracy.append(guess_accuracy)
                        train_decision_loss = torch.cat([train_decision_loss, decider_loss.data/batch_size])
                        train_qgen_loss = torch.cat([train_qgen_loss, word_logits_loss.data])
                        train_guesser_loss = torch.cat([train_guesser_loss, guesser_loss.data])

                    train_total_loss = torch.cat([train_total_loss, loss.data])

                    if exp_config['logging']:
                        visualise.iteration_update(
                            loss=loss.data[0],
                            qgen_loss=word_logits_loss.data[0],
                            guesser_loss=guesser_loss.data[0],
                            decider_loss=decider_loss.data[0]/batch_size,
                            ask_accuracy=ask_accuracy,
                            guess_accuracy=guess_accuracy,
                            guesser_accuracy=guesser_accuracy,
                            training=True,
                            modulo= args.modulo,
                            epoch= epoch
                        )
                elif split == 'val':
                    if epoch%args.modulo != 0:
                        val_qgen_loss = torch.cat([val_qgen_loss, word_logits_loss.data])
                        val_decision_loss = torch.cat([val_decision_loss, decider_loss.data/batch_size])
                        validation_ask_accuracy.append(ask_accuracy)
                        validation_guess_accuracy.append(guess_accuracy)
                    else:
                        validation_guesser_accuracy.append(guesser_accuracy)
                        validation_ask_accuracy.append(ask_accuracy)
                        validation_guess_accuracy.append(guess_accuracy)
                        val_decision_loss = torch.cat([val_decision_loss, decider_loss.data/batch_size])
                        val_qgen_loss = torch.cat([val_qgen_loss, word_logits_loss.data])
                        val_guesser_loss = torch.cat([val_guesser_loss, guesser_loss.data])

                    val_total_loss = torch.cat([val_total_loss, loss.data])

                    if exp_config['logging']:
                        visualise.iteration_update(
                            loss=loss.data[0],
                            qgen_loss=word_logits_loss.data[0],
                            guesser_loss=guesser_loss.data[0],
                            decider_loss=decider_loss.data[0]/batch_size,
                            ask_accuracy=ask_accuracy,
                            guess_accuracy=guess_accuracy,
                            guesser_accuracy=guesser_accuracy,
                            training=False,
                            modulo= args.modulo,
                            epoch= epoch
                        )

        if exp_config['save_models'] and (epoch%args.modulo == 0):
            model_file = os.path.join(model_dir, ''.join(['model_ensemble_', args.bin_name,'_E_', str(epoch)]))
            torch.save(model.state_dict(), model_file)

        if epoch%args.modulo != 0:
            print("Epoch %03d, Time taken %.3f, Total Training Loss %.4f, Total Validation Loss %.4f"
                %(epoch, time()-start, torch.mean(train_total_loss), torch.mean(val_total_loss)))
            print("Training Loss:: QGen %.3f, Decider %.3f"%(torch.mean(train_qgen_loss), torch.mean(train_decision_loss)))
            print("Validation Loss:: QGen %.3f, Decider %.3f"%(torch.mean(val_qgen_loss), torch.mean(val_decision_loss)))
        else:
            print("Epoch %03d, Time taken %.3f, Total Training Loss %.4f, Total Validation Loss %.4f"
                %(epoch, time()-start, torch.mean(train_total_loss), torch.mean(val_total_loss)))
            print("Training Loss:: QGen %.3f, Decider %.3f, Guesser %.3f"%(torch.mean(train_qgen_loss), torch.mean(train_decision_loss), torch.mean(train_guesser_loss)))
            print("Validation Loss:: QGen %.3f, Decider %.3f, Guesser %.3f"%(torch.mean(val_qgen_loss), torch.mean(val_decision_loss), torch.mean(val_guesser_loss)))
            print("Training Accuracy:: Ask %.3f, Guess  %.3f, Guesser %.3f"%(safe_mean(training_ask_accuracy), safe_mean(training_guess_accuracy), safe_mean(training_guesser_accuracy)))
            print("Validation Accuracy:: Ask %.3f, Guess  %.3f, Guesser %.3f"%(safe_mean(validation_ask_accuracy), safe_mean(validation_guess_accuracy), safe_mean(validation_guesser_accuracy)))

            if exp_config['save_models']:
                print("Saved model to %s" % (model_file))
        print('-----------------------------------------------------------------')
        if exp_config['logging']:
            visualise.epoch_update(
                train_loss=torch.mean(train_total_loss),
                train_qgen_loss=torch.mean(train_qgen_loss),
                train_guesser_loss=0 if (epoch%args.modulo != 0) else torch.mean(train_guesser_loss),
                train_decider_loss=torch.mean(train_decision_loss),
                train_ask_accuracy=safe_mean(training_ask_accuracy),
                train_guess_accuracy=safe_mean(training_guess_accuracy),
                train_guesser_accuracy=0 if (epoch%args.modulo != 0) else safe_mean(training_guesser_accuracy),
                valid_loss=torch.mean(val_total_loss),
                valid_qgen_loss=torch.mean(val_qgen_loss),
                valid_guesser_loss=0 if (epoch%args.modulo != 0) else torch.mean(val_guesser_loss),
                valid_decider_loss=torch.mean(val_decision_loss),
                valid_ask_accuracy=safe_mean(validation_ask_accuracy),
                valid_guess_accuracy=safe_mean(validation_guess_accuracy),
                valid_guesser_accuracy=0 if (epoch%args.modulo != 0) else safe_mean(validation_guesser_accuracy),
                epoch=epoch,
                modulo=args.modulo)
