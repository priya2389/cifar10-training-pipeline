import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 

#model is used on a series of input images and outputs and results in 10 labels (0-9) defined by features
#  each corresponding to a particular output. Used to classift new input images 

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      #1 input (image) --> (kernel size 3, stride 1) --> outputs 32 convolutional features
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      #32 input layers --> (kernel size 3, stride 1) --> outputs 64 convolutional features
      self.conv2 = nn.Conv2d(32, 64, 3, 1)

      #dropout layers used for regularization during training and preserving output magnitude to reduce overfitting
      #zeros out 25% of channels from input tensor 
      self.dropout1 = nn.Dropout2d(0.25)
      #zeros out 50% of channels from input tensor 
      self.dropout2 = nn.Dropout2d(0.5)

      #fully connected layer
      self.fc1 = nn.Linear(9216, 128)
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      #conv1 first then reLu to make it non-linear
      x = self.conv1(x)
      x = F.relu(x)
      #conv2 next then reLu again to make non-linear
      x = self.conv2(x)
      x = F.relu(x)

      #max pooling with kernel size 2 to reduce spatial dimensions
      #sides a 2x2 kernel over input and takes only the max values 
      x = F.max_pool2d(x, 2)

      #reduces overfitting 
      x = self.dropout1(x)
      #3D vector --> 1D vector
      x = torch.flatten(x, 1)

      #picks 128 meaningful features/weights from the 9216 input features
      x = self.fc1(x)

       #non-linearity and reduces overfitting
      x = F.relu(x)
      x = self.dropout2(x)

      #picks the 10 most important features outputing 10 labels
      x = self.fc2(x)

      #softmax converts scores into probabilities 
      output = F.log_softmax(x, dim=1)
      return output
