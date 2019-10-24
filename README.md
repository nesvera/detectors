* You must have the class definition of the base model inside the directory to load the trained model.

1. Convert the trained classifier
python export_model.py --model /path/to/trained_model.pth.tar --output /path/to/save/model.pth.tar

2. Use the converted model to train the detector
