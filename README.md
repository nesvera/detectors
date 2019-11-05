Train:
- It is recommended to have pretrained weights of the base model
- The trained weights file must contain the state_dict of the model
- You need to declare the model to use the trained weight, it is not possible to load model and weights from the checkpoint (keras>>pytorch in this case)

1. Convert the trained classifier. The script is placed inside the classifier repository.
python export_model.py --model /path/to/trained_model.pth.tar --output /path/to/save/model.pth.tar

2. Add the path to the trained model in the config file

