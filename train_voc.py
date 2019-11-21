import torch
import torch.nn as nn
from torchsummary import summary

from classifiers.models import MobileNet
from models import SSD_MobileNet, SSD_VGG_16
from utils import datasets, loss_function, utils

import argparse
import numpy as np
import os
import yaml
import cv2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        dest="config_path",
                        required=True,
                        help="Configuration file with train hyperparameters")

    args = parser.parse_args()

    # ------------------------
    #    Load configuration 
    # ------------------------
    config_path = args.config_path
    if os.path.exists(config_path) == False:
        print("Error: Config file does not exist")
        exit(1)

    # Load hyperparameters from configuration file
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    config_model_name =         config['MODEL']['NAME']
    config_input_size =         config['MODEL']['IMAGE_SIZE']
    #config_num_classes =        config['MODEL']['NUM_CLASSES']
    config_base_model_path =    config['MODEL']['BASE_MODEL']['CHECKPOINT']
    config_alpha =              config['MODEL']['BASE_MODEL']['ALPHA']
    config_optimizer =          config['TRAIN']['OPTIMIZER']['OPTIMIZER']
    config_lr =                 config['TRAIN']['OPTIMIZER']['LEARNING_RATE']
    config_momentum =           config['TRAIN']['OPTIMIZER']['MOMENTUM']
    config_weight_decay =       config['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    config_lr_scheduler =       config['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER']
    config_sgdr_min_lr =        config['TRAIN']['LR_SCHEDULER']['MIN_LR']
    config_sgdr_max_lr =        config['TRAIN']['LR_SCHEDULER']['MAX_LR']
    config_sgdr_lr_decay =      config['TRAIN']['LR_SCHEDULER']['LR_DECAY']
    config_sgdr_cycle =         config['TRAIN']['LR_SCHEDULER']['CYCLE']
    config_sgdr_cycle_mult=     config['TRAIN']['LR_SCHEDULER']['CYCLE_MULT']
    config_workers =            config['TRAIN']['WORKERS']
    config_max_epochs =         config['TRAIN']['MAX_EPOCHS']
    config_train_batch =        config['TRAIN']['BATCH_SIZE']
    config_val_batch =          config['TEST']['BATCH_SIZE']
    config_print_freq =         config['TRAIN']['PRINT_FREQ']
    config_experiment_path =    config['EXP_DIR']
    config_checkpoint =         config['RESUME_CHECKPOINT']

    # ------------------------
    #       Dataloaders
    # ------------------------  
    data_folder = "/home/feaf-seat-1/Documents/nesvera/object_detection/a-PyTorch-Tutorial-to-Object-Detection"
    train_dataset = datasets.PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=True,
                                     dims=(config_input_size[0], config_input_size[1]))
    
    val_dataset = datasets.PascalVOCDataset(data_folder,
                                   split='test',
                                   keep_difficult=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config_train_batch,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=config_workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config_val_batch,
                                             shuffle=True,
                                             collate_fn=val_dataset.collate_fn,
                                             num_workers=config_workers,
                                             pin_memory=True)

    config_num_classes = len(train_dataset.label_map)

    # ------------------------
    #    Build/Load model
    # ------------------------
    # Keep track of losses
    train_loss_log = {}
    val_loss_log = {}
    top_5_log = {}
    top_1_log = {}
    lr_log = {}

    best_loss = 9000.
    start_epoch = 0

    checkpoint_path = config_experiment_path + "/" + config_model_name
    checkpoint_path += "/" + config_checkpoint

    # load checkpoint or create new model
    if os.path.exists(checkpoint_path) == False:
        config_checkpoint = ""
        print("Warning: Checkpoint was not found!")
    
        if config_model_name == "detector_mobilenet":
  
            # Initialize a new model
            if config_checkpoint == "":
                # load base model
                if os.path.exists(config_base_model_path) == False:
                    print("Error: base model file was not find!")
                    exit(1)

            print("Warning: Loading base weights")
            print(config_base_model_path)

            # load weights from trained classifier as a state_dict
            base_pretrained = torch.load(config_base_model_path, map_location=device)

            # build detector
            model = SSD_MobileNet.SSDMobileNet(base_pretrained, config_num_classes, alpha=config_alpha)

            '''        
            optimizer = torch.optim.Adam(model.parameters(),
                                            lr=config_lr,
                                            weight_decay=config_weight_decay)
            '''

            # Initialize the optimizer with twice the default learning rate for biases
            biases = list()
            not_biases = list()

            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)

            optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2*config_lr}, {'params': not_biases}],
                                        lr=config_lr, momentum=config_momentum, weight_decay=config_weight_decay)

        elif config_model_name == "detector_vgg_16":
        
            model = SSD_VGG_16.SSD300(n_classes=config_num_classes)

            # Initialize the optimizer with twice the default learning rate for biases
            biases = list()
            not_biases = list()

            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)

            optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2*config_lr}, {'params': not_biases}],
                                        lr=config_lr, momentum=config_momentum, weight_decay=config_weight_decay)

    else:
        print("Warning: Loading checkpoint!")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        start_epoch =       checkpoint['epoch'] + 1
        model =             checkpoint['model']
        optimizer =         checkpoint['optimizer']

        train_loss_log =    checkpoint['train_loss_log']
        val_loss_log =      checkpoint['val_loss_log']
        top_5_log = {}
        top_1_log = {}
        lr_log = {}

        best_loss = 9000
        for i in val_loss_log.keys():
            best_loss = min(best_loss, val_loss_log[i])
        
        print('best', best_loss)


    print("Press enter to continue")
    input()

    model = model.to(device)
    priors_boxes = model.create_prior()
    criterion = loss_function.MultiBoxLoss(priors_cxcy=priors_boxes)

    prior_b = priors_boxes
    print("prior", prior_b.size())

    # summarize the model
    print()
    print("----------------------------------------------------------------")
    print("--------------------- Model summary ----------------------------")
    print("----------------------------------------------------------------")
    summary(model, (config_input_size[2], config_input_size[0], config_input_size[1]))

    print("Press ENTER to continue")
    input()

    # Keep track of learning rate
    lr_schedule = utils.SGDR(min_lr=config_sgdr_min_lr,
                           max_lr=config_sgdr_max_lr,
                           lr_decay=config_sgdr_lr_decay,
                           epochs_per_cycle=config_sgdr_cycle,
                           mult_factor=config_sgdr_cycle_mult)

    # Keep track for improvement
    epochs_since_last_improvement = 0

    for epoch in range(start_epoch, config_max_epochs):

        # ------------------------
        #          Train
        # ------------------------   
        train_loss = train(model=model,
                           loader=train_loader,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           print_freq=config_print_freq)

        train_loss_log[epoch] = train_loss

        # ------------------------
        #        Validation 
        # ------------------------
        val_loss = validation(model=model,
                              loader=val_loader,
                              criterion=criterion,
                              optimizer=optimizer,
                              epoch=epoch,
                              print_freq=config_print_freq)
        
        val_loss_log[epoch] = val_loss

        # Check if the model improved
        is_best = 0
        epochs_since_last_improvement += 1

        if val_loss < best_loss:
            is_best = 1
            best_loss = val_loss
            epochs_since_last_improvement = 0
            print("Melhoroooooouuuuu\n")

        print('--------------------------------------------------------------------')
        print("Val loss: {0:.3f} - Best loss: {1:.3f} \n Epochs since last improvement: {2}\n"
              .format(val_loss, best_loss, epochs_since_last_improvement))
        print('--------------------------------------------------------------------')
        print()

        # ------------------------
        #        Save model 
        # ------------------------
        exp_folder = config_experiment_path + "/" + config_model_name

        # Create a folder for a new topology/experiment
        if os.path.isdir(exp_folder) == False:
            os.mkdir(exp_folder)

        state = {'model_name': config_model_name,
                 'epoch': epoch,
                 'loss': val_loss,
                 'train_loss_log': train_loss_log,
                 'val_loss_log': val_loss_log,
                 'top_5_log': top_5_log,
                 'top_1_log': top_1_log,
                 'lr_log': lr_log,
                 'model': model,
                 'optimizer': optimizer}

        exp_filename = exp_folder + '/' + config_model_name + '.pth.tar'
        torch.save(state, exp_filename)

        if is_best:
            exp_filename = exp_folder + '/BEST_' + config_model_name + '.pth.tar'
            torch.save(state, exp_filename)

        if epoch == (config_max_epochs-1):
            exp_filename = exp_folder + '/LAST_' + config_model_name + '.pth.tar'
            torch.save(state, exp_filename)

        # ------------------------
        #  Learning rate schedule 
        # ------------------------
        if config_lr_scheduler == True:
            config_lr = lr_schedule.update()

            for opt in optimizer.param_groups:
                opt['lr'] = config_lr

            print('LR Scheduler - Cycle: [{0}/{1}]'
                  .format(lr_schedule.epoch_since_restart, lr_schedule.epochs_per_cycle))
            print('LR: {0:.5f}\n'.format(config_lr))

        else:
            print("Sem lr scheduler\n")

def train(model, loader, criterion, optimizer, epoch, print_freq):
    
    model.train()

    # Data loading logging
    epoch_fetch_time = utils.Average()      
    batch_fetch_time = utils.Average()    

    epoch_train_time = utils.Average()      # forward prop. + backprop.
    partial_train_time = utils.Average()    # forward prop. + backprop., reset for each batch

    epoch_loss = utils.Average()        # loss average

    batch_start = time.time()
    for i, (images, boxes, labels, _) in enumerate(loader):

        epoch_fetch_time.add_value(time.time()-batch_start)
        batch_fetch_time.add_value(time.time()-batch_start)

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward prop.
        predited_locs, predited_scores = model(images)      
        # [N, n_priors, 4], [N, n_priors, n_classes]

        # Calculate loss
        loss = criterion(predited_locs, predited_scores, boxes, labels) 
        # scalar

        # Backprop
        loss.backward()

        # Update model
        optimizer.step()

        epoch_loss.add_value(loss.item()) 

        epoch_train_time.add_value(time.time()-batch_start)       # measure train time
        partial_train_time.add_value(time.time()-batch_start)

        batch_start = time.time()

        # print statistics
        if i % print_freq == 0:
            print('Epoch: [{0}] - Batch: [{1}/{2}]'.format(epoch, i, len(loader)))
            print('Batch fetch time - Average: {0:.4f} - Total: {1:.4f} (seconds/batch)'
                 .format(batch_fetch_time.get_average(), batch_fetch_time.get_sum()))
            print('Batch train time - Average: {0:.4f} - Total: {1:.4f} (seconds/batch)'
                 .format(partial_train_time.get_average(), partial_train_time.get_sum()))  
            print('Loss: {0:.5f}'.format(epoch_loss.get_average()))
            print()

            # Reset measurment for each
            batch_fetch_time = utils.Average()
            partial_train_time = utils.Average()

    print('--------------------------------------------------------------------')
    print('Training')    
    print('--------------------------------------------------------------------')
    print('Epoch [{0}] - Train time: {1:.4f} (seconds/epoch) - Loss: {2:.5f}'
          .format(epoch, epoch_train_time.get_sum(), epoch_loss.get_average()))
    print('--------------------------------------------------------------------')
    print()

    time.sleep(2)

    return epoch_loss.get_average()

def validation(model, loader, criterion, optimizer, epoch, print_freq):
    
    model.eval()

    epoch_eval_time = utils.Average()
    batch_eval_time = utils.Average()   

    epoch_loss = utils.Average()        # loss average

    batch_start = time.time()

    # Prohibit gradient computation explicity because problems with memory
    with torch.no_grad():
        for i, (images, boxes, labels, _) in enumerate(loader):

            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predited_locs, predited_scores = model(images)      
            # [N, n_priors, 4], [N, n_priors, n_classes]

            # Calculate loss
            eval_loss = criterion(predited_locs, predited_scores, boxes, labels) 

            epoch_loss.add_value(eval_loss.item())

            epoch_eval_time.add_value(time.time()-batch_start)
            batch_eval_time.add_value(time.time()-batch_start)

            batch_start = time.time()

            # print statistics
            if i % print_freq == 0:
                print('Validation - Epoch: [{0}] - Batch: [{1}/{2}]'.format(epoch, i, len(loader)))
                print('Evaluation time - Average: {0:.4f} - Total: {1:.4f} (seconds/batch)'
                     .format(batch_eval_time.get_average(), batch_eval_time.get_sum()))  
                print('Loss: {0:.5f} - Current loss: {1:.5f}'.format(epoch_loss.get_average(), eval_loss.item()))
                print()

                batch_eval_time = utils.Average()

    print('--------------------------------------------------------------------')
    print('Evaluation')    
    print('--------------------------------------------------------------------')
    print('Epoch [{0}] - Train time: {1:.4f} (seconds/epoch) - Loss: {2:.5f}'
          .format(epoch, epoch_eval_time.get_sum(), epoch_loss.get_average()))
    print('--------------------------------------------------------------------')
    print()


    return epoch_loss.get_average()

if __name__ == "__main__":
    main()
