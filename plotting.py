import matplotlib.pyplot as plt
import numpy as np

def playback(images):
    for i, img in enumerate(images):
        plt.figure(1)
        plt.clf()
        plt.imshow(img, cmap="gray")
        plt.title(f"Image {i+1}")
        plt.pause(0.1)

def plot_accuracy(train_acc_list, test_acc_list, save_path):
    x = np.arange(1, len(train_acc_list) + 1)
    fig, ax = plt.subplots()
    ax.plot(x, train_acc_list, label="Training")
    ax.plot(x, test_acc_list, label="Testing")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig(save_path)

def plot_loss(loss_mean_list, loss_std_list, save_path):
    x = np.arange(1, len(loss_mean_list) + 1)
    fig, ax = plt.subplots()
    ax.errorbar(x, loss_mean_list, loss_std_list)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Loss")
    fig.savefig(save_path)

def plot_confusion_matrix(cm, labels, save_path):
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Actuals")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center')
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    fig.savefig(save_path)
