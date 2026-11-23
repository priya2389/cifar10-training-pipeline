# CIFAR-10 Training Pipeline (PyTorch)

A small convolutional neural network (CNN) trained to classify
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) images into 10 classes:
plane, car, bird, cat, deer, dog, frog, horse, ship, truck.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python training.py        # 12 epochs (default)
python training.py 2      # or choose the number of epochs
```

On the first run, CIFAR-10 (~170 MB compressed) downloads into `./data/`.
The trained weights are saved to `./models/epoch<N>_model.pt`, and the script
then prints overall and per-class accuracy on the 10,000 test images.

One epoch takes roughly 20–30 seconds on an Apple Silicon CPU and gets about 43% test accuracy.
More epochs improve accuracy up to a point.

## What's in here

| File | Purpose |
|---|---|
| `training.py` | The full pipeline: load data → train → save weights → evaluate |
| `training.ipynb` | The same pipeline as a step-by-step notebook, with sample image plots |
| `net1.py` | An alternative, deeper CNN with dropout (for 1-channel 28×28 images, MNIST-style) |
| `net2.ipynb` | A walkthrough of PyTorch basics: forward pass, loss, backprop, weight updates |
| `models/` | Previously trained weights (2, 12 and 25 epochs) |
| `cifar_net.pt` | Weights saved from the notebook run |

## How it works

### 1. The model (`Net`)

The network follows the LeNet-5 design:

```
input  3×32×32  (RGB image)
conv1  5×5, 6 filters   → 6×28×28  → ReLU → maxpool 2×2 → 6×14×14
conv2  5×5, 16 filters  → 16×10×10 → ReLU → maxpool 2×2 → 16×5×5
flatten                 → 400
fc1                     → 120 → ReLU
fc2                     → 84  → ReLU
fc3                     → 10  (one score per class)
```

### 2. Data

Images are converted to float tensors in `[0, 1]`, then normalized to `[-1, 1]`
with mean 0.5 and std 0.5 per channel. `DataLoader` serves shuffled mini-batches of 4.

### 3. Training

For each epoch, each mini-batch goes through these steps:
1. `optimizer.zero_grad()` clears the old gradients
2. `outputs = net(inputs)` runs the forward pass
3. `loss = criterion(outputs, labels)` computes cross-entropy loss
4. `loss.backward()` backpropagates to compute gradients
5. `optimizer.step()` runs an SGD update (lr 0.001, momentum 0.9)

The average loss is printed every 2,000 mini-batches.

### 4. Evaluation

The script turns off gradients (`torch.no_grad()`), runs the test set through
the network, takes the highest-scoring class as the prediction, and reports
overall accuracy and accuracy for each class.

## Loading saved weights

```python
import torch
from training import Net

net = Net()
net.load_state_dict(torch.load("models/epoch12_model.pt"))
net.eval()
```
