# Category_classif_Images
Built a web app in Flask for generic category classification from images. 

## Assumptions
- Code is generic for n number of categories
- Assuming that classes are non-overlapping (Softmax assumption)
- Works really well for distinguishing different objects (like cat vs dog)

## Data
- Data has to organized in data/ folder
- training examples should be present in data/training/class1/ , data/training/class2/ etc
- validation examples should be present in data/validation/class1/, data/validation/class2 etc

## Model
- This works by doing Transfer learning on VGG16 model.
- step1: Firstly features are extracted from images using VGG16 model (by removing last layer) and stored in bottleneck_fc_model.h5
- step2: (Training) 2 fully connected layers are added on top of VGG16 and weights of VGG16 are frozen while training
- step3: (FineTuning) Last few layers of Vgg16 are fine tuned to give better accuracy. This step is optional

Code for step1 and step2 can be found in TransferLearning.py. Step3 - finetuning.py

## WebApp
- This model is hosted on WebApp whose backed is built on Flask (Python)
- Just Run app.py to start the server
