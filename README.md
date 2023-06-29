# SWP-2023-Automatic-Detection-of-Vibratory-Behavior-of-Honeybees

Model for classifying bee behavior based on a stack of video frames.

The evaluations of different model architectures are saved into the `stats_*` directories.
The contents of the `stats_*` directory are generated after training and show the test/training accuracies at each epoch,
mean loss at each epoch and confusion matrix from the trained model.

## Results

<table>
      <tr><td><b>Model</b></td><td>CNN (4 x Conv2d + BN + ReLU + MaxPool) + LSTM (1 x 128)</td></tr>
      <tr></tr>
      <tr><td><b>Optimizer</b></td><td>Adam, LR=1e-5</td></tr>
      <tr></tr>
      <tr><td><b>Criterion</b></td><td>Cross Entropy</td></tr>
      <tr></tr>
      <tr><td><b>Batch Size</b></td><td>16</td></tr>
      <tr></tr>
      <tr><td><b>Batch Sampler</b></td><td>None</td></tr>
      <tr></tr>
      <tr><td><b>Epochs</b></td><td>64</td></tr>
      <tr></tr>
      <tr><td><b>Image Augmentation</b></td><td>Random Roation / Flip for each image in video</td></tr>
      <tr></tr>
      <tr><td><b>Image Size</b></td><td>110 x 110</td></tr>
      <tr></tr>
      <tr><td><b>Image Normalization</b></td><td>Mean=0.5, Std=0.5 (Range=[-1,1])</td></tr>
      <tr></tr>
      <tr><td><b>Input Format</b></td><td>16 x L x 1 x 110 x 110 (No Padding / Trimming)</td></tr>
      <tr></tr>
      <tr><td><b>Output Format</b></td><td>16 x 4</td></tr>
</table>

![Accuracy](stats_20230602T2153/accuracy.jpg)
![Confusion Matrix](stats_20230602T2153/confusion.jpg)

<table>
      <tr><td><b>Model</b></td><td>Pretrained Resnet18 CNN + LSTM (1 x 128)</td></tr>
      <tr></tr>
      <tr><td><b>Optimizer</b></td><td>Adam, LR=1e-5</td></tr>
      <tr></tr>
      <tr><td><b>Criterion</b></td><td>Cross Entropy</td></tr>
      <tr></tr>
      <tr><td><b>Batch Size</b></td><td>8</td></tr>
      <tr></tr>
      <tr><td><b>Batch Sampler</b></td><td>Oversampling of minority classes for equal class distribution</td></tr>
      <tr></tr>
      <tr><td><b>Epochs</b></td><td>128</td></tr>
      <tr></tr>
      <tr><td><b>Image Augmentation</b></td><td>None</td></tr>
      <tr></tr>
      <tr><td><b>Image Size</b></td><td>224 x 224</td></tr>
      <tr></tr>
      <tr><td><b>Image Normalization</b></td><td>Mean=(0.485 + 0.456 + 0.406)/3, Std=(0.229 + 0.224 + 0.225)/3
      <tr></tr>
      <tr><td><b>Input Format</b></td><td>8 x L x 1 x 224 x 224 (No Padding / Trimming)</td></tr>
      <tr></tr>
      <tr><td><b>Output Format</b></td><td>8 x 4</td></tr>
</table>

![Accuracy](stats_20230628T2053/accuracy.jpg)
![Confusion Matrix](stats_20230628T2053/confusion.jpg)

## TODO

-   [x] image augmenation (blur, contrast, crop, translation, stretching, padding)
-   [ ] pretrain CNN to produce good image embeddings
-   [x] write a custom batch sampler that samples training examples such that class labels are equally distributed ("other", "waggle", "ventilating", "activating" classes should have an equal chance of being sampled in a batch)
-   [x] save the state of the model and the state of the optimizer during training. See [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for how to do this in pytorch
