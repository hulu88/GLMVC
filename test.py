import torch
from network import Network
from metric import valid
from dataprocessing import load_data,load_test
"""
# This code is used to load different datasets and validate the performance of the trained model
# Main Steps:
# 1. Load the training and test parameters based on the selected dataset name
# 2. Configure device (CUDA or CPU)
# 3. Load the dataset and initialize the network model
# 4. Load model weights from saved checkpoints
# 5. Use the 'valid' function to validate the loaded model
"""

# MNIST_USPS
# msrcv1
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
dataname="MNIST_USPS"
params = load_test(dataname)
args=params
Dataname = args["dataname"]

dataset = Dataname

feature_dim = args["feature_dim"]
high_feature_dim = args["high_feature_dim"]
mid_dim = args["mid_dim"]
layers1 = args["layers1"]
layers2 = args["layers2"]
chance = args["chance"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(Dataname)
model = Network(view, dims, feature_dim, high_feature_dim, mid_dim,layers1,layers2, chance,class_num, device)
model = model.to(device)
checkpoint = torch.load('models/' + Dataname + '.pth', map_location=lambda storage, loc: storage.cuda(0))
model.load_state_dict(checkpoint)

print("Dataset:{}".format(Dataname))
print("Datasize:" + str(data_size))
#print("Loading models...")
valid(model, device, dataset, view, data_size, class_num, eval_h=False)
