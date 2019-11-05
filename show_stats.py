import torch

import os
import argparse
import matplotlib.pyplot as plt

from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        dest="model_path",
                        help="Path to the model",
                        required=True)
    
    args = parser.parse_args()
    model_path = args.model_path

    if os.path.exists(model_path) == False:
        print("ERROR: Model file was not found!")
        exit(1)
       
    # load model
    checkpoint = torch.load(model_path, map_location=device)

    #model_name =    checkpoint['model_name']
    cur_epoch =     checkpoint['epoch']
    model =         checkpoint['model']
    optimizer =     checkpoint['optimizer']
    train_loss_log =checkpoint['train_loss_log']
    val_loss_log =  checkpoint['val_loss_log']
    top_5_log =     checkpoint['top_5_log']
    top_1_log =     checkpoint['top_1_log']
    lr_log =        checkpoint['lr_log']

    # summarize the model
    print()
    print("----------------------------------------------------------------")
    print("--------------------- Model summary ----------------------------")
    print("----------------------------------------------------------------")
    summary(model, (3, 224, 224))
    
    # plot stats
    fig, sub_plot = plt.subplots(2, 1)

    # train and validation loss
    train_loss = sorted(train_loss_log.items())
    train_loss_x, train_loss_y = zip(*train_loss) # unzip a list

    val_loss = sorted(val_loss_log.items())
    val_loss_x, val_loss_y = zip(*val_loss)

    tl_graph, = sub_plot[0].plot(train_loss_x, train_loss_y, 'b')
    vl_graph, = sub_plot[0].plot(val_loss_x, val_loss_y, 'r')
    sub_plot[0].set_xlabel('Epoch')
    sub_plot[0].set_ylabel('Loss')
    sub_plot[0].set_xlim(0, max(train_loss_x[-1], val_loss_x[-1]))

    fig.legend((tl_graph, vl_graph), 
               ('Train loss', 'Val loss'), 
               'upper right')

    print("Best")
    print("Train loss: {0:.3f} - Val loss: {1:.3f}"
          .format(min(train_loss_y), min(val_loss_y)))

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
