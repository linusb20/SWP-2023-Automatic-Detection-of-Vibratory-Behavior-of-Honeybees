import os
import json
import numpy as np
import torch
from plotting import plot_accuracy, plot_loss, plot_confusion_matrix, playback

class Callback:
    def on_train_begin(self): pass
    def on_train_end(self): pass
    def on_epoch_begin(self): pass
    def on_epoch_end(self): pass
    def on_batch_begin(self): pass
    def on_batch_end(self): pass

class AugmentScheduler(Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, epoch):
        self.dataset.augment_p = min(0.5 + np.log1p(epoch / 50), 1)

class StatsCallback(Callback):
    def __init__(self, log_interval, path_stats, all_labels, stats=None):
        self.log_interval = log_interval
        self.path_stats = path_stats
        self.all_labels = all_labels
        self.stats = stats or {
            "train_acc_list": [],
            "test_acc_list": [],
            "train_loss_mean_list": [],
            "train_loss_std_list": [],
            "test_loss_mean_list": [],
            "test_loss_std_list": [],
        }
        os.makedirs(path_stats, exist_ok=True)

    def on_batch_end(self, batch_idx, batch_total, epoch, loss):
        if (batch_idx + 1) % self.log_interval == 0:
            print(f"[Epoch {epoch+1:3}] Batch {batch_idx+1:3} of {batch_total:3}")
            print(f"Loss: {loss:.4f}")

    def on_epoch_end(
        self,
        train_loss_mean, 
        train_loss_std, 
        train_acc, 
        test_loss_mean,
        test_loss_std,
        test_acc,
    ):
        self.stats["train_acc_list"].append(round(train_acc, 3))
        self.stats["test_acc_list"].append(round(test_acc, 3))
        self.stats["train_loss_mean_list"].append(round(train_loss_mean, 3))
        self.stats["train_loss_std_list"].append(round(train_loss_std, 3))
        self.stats["test_loss_mean_list"].append(round(test_loss_mean, 3))
        self.stats["test_loss_std_list"].append(round(test_loss_std, 3))

        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Testing Accuracy: {test_acc:.2f}%")
        print(f"Training Loss Mean: {train_loss_mean:.2f}")
        print(f"Training Loss Std: {train_loss_std:.2f}")
        print(f"Testing Loss Mean: {test_loss_mean:.2f}")
        print(f"Testing Loss Std: {test_loss_std:.2f}")

    def on_train_end(self, metrics, model_str):
        self.stats.update({k: metrics[k].tolist() for k in ["precision", "recall", "fscore", "roc_auc"]})
        other_precision, waggle_precision, ventilating_precision, activating_precision = metrics["precision"]
        other_recall, waggle_recall, ventilating_recall, activating_recall = metrics["recall"]
        other_fscore, waggle_fscore, ventilating_fscore, activating_fscore = metrics["fscore"]
        print(f"waggle_precision: {waggle_precision:.3f}, waggle_recall: {waggle_recall:.3f}, waggle_fscore: {waggle_fscore:.3f}")
        print(f"ventilating_precision: {ventilating_precision:.3f}, ventilating_recall: {ventilating_recall:.3f}, ventilating_fscore: {ventilating_fscore:.3f}")
        print(f"activating_precision: {activating_precision:.3f}, activating_recall: {activating_recall:.3f}, activating_fscore: {activating_fscore:.3f}")
        print(f"roc_auc:", metrics["roc_auc"])

        plot_accuracy(self.stats["train_acc_list"], self.stats["test_acc_list"], os.path.join(self.path_stats, "accuracy.pdf"))
        plot_loss(self.stats["train_loss_mean_list"], self.stats["train_loss_std_list"], os.path.join(self.path_stats, "train_loss.pdf"))
        plot_loss(self.stats["test_loss_mean_list"], self.stats["test_loss_std_list"], os.path.join(self.path_stats, "test_loss.pdf"))
        plot_confusion_matrix(metrics["confusion_matrix"], self.all_labels, os.path.join(self.path_stats, "confusion.pdf"))
        with open(os.path.join(self.path_stats, "stats.json"), "w") as f:
            json.dump(self.stats, f)
        with open(os.path.join(self.path_stats, "model.txt"), "w") as f:
            print(model_str, file=f)


class SaveModelCallback(Callback):
    def __init__(self, path_checkpoints, save_interval, accuracy_threshold):
        self.save_interval = save_interval
        self.accuracy_threshold = accuracy_threshold
        self.path_checkpoints = path_checkpoints
        os.makedirs(path_checkpoints, exist_ok=True)

    def on_epoch_end(self, model, optimizer, epoch, stats, testing_accuracy):
        if testing_accuracy > self.accuracy_threshold:
            save_path = os.path.join(path_checkpoints, "save_best.pth")
            SaveModelCallback.save_model(model, optimizer, epoch, stats, save_path)
        elif (epoch + 1) % self.save_interval == 0:
            save_path = os.path.join(path_checkpoints, "save_last.pth")
            SaveModelCallback.save_model(model, optimizer, epoch, stats, save_path)

    @staticmethod
    def save_model(model, optimizer, epoch, stats, save_path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": stats,
        }, save_path)
        print(f"Saved model in epoch {epoch+1} to {save_path}")
