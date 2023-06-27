# SWP-2023-Automatic-Detection-of-Vibratory-Behavior-of-Honeybees

Model for classifying bee behavior based on a stack of video frames.

The evaluations of different model architectures are saved into the `stats_*` directories.
The contents of the `stats_*` directory are generated after training and show the test/training accuracies at each epoch,
mean loss at each epoch and confusion matrix from the trained model.

## TODO

-   [x] image augmenation (blur, contrast, crop, translation, stretching, padding)
-   [ ] pretrain CNN to produce good image embeddings
-   [ ] write a custom batch sampler that samples training examples such that class labels are equally distributed ("other", "waggle", "ventilating", "activating" classes should have an equal chance of being sampled in a batch)
-   [ ] save the state of the model and the state of the optimizer during training. See [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for how to do this in pytorch
