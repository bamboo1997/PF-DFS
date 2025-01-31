import torch
from tqdm import tqdm
import datetime
import os
import csv
import time


def calc_accuracy(output, labels):
    pred = output.data.max(1, keepdim=True)[1]
    acc = pred.eq(labels.data.view_as(pred)).sum()
    return acc


class Classifier(object):
    def __init__(self, cfg, model, device, optimizer, scheduler, loss_func) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.scaler = torch.cuda.amp.GradScaler(init_scale=2**12)

    def train_step(self, train_loader):
        self.model.train()

        train_loss_sum = 0
        train_acc_sum = 0
        n_trains = 0

        bar = (
            tqdm(total=len(train_loader), leave=False) if self.cfg.default.bar else None
        )

        start_time = time.time()
        for i_batch, sample_batched in enumerate(train_loader):
            images = sample_batched["image"].to(self.device, non_blocking=True)
            labels = sample_batched["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=True):
                pred = self.model(images)
                train_loss = self.loss_func(pred, labels)

            train_acc = calc_accuracy(pred, labels)

            self.scaler.scale(train_loss).backward()
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            # update model parameter
            self.scaler.step(self.optimizer)
            # update scaler paramter
            self.scaler.update()

            train_loss_sum += train_loss * labels.size(0)
            train_acc_sum += train_acc
            n_trains += labels.size(0)

            bar.update(1) if self.cfg.default.bar else None

        self.training_time = time.time() - start_time
        bar.close() if self.cfg.default.bar else None

        self.train_loss = float(train_loss_sum / n_trains)
        self.train_acc = float(train_acc_sum / n_trains)

    def val_step(self, val_loader):
        self.model.eval()

        val_loss_sum = 0
        val_acc_sum = 0
        n_vals = 0

        start_time = time.time()
        with torch.no_grad():
            for i, sample_batched in enumerate(val_loader):
                images = sample_batched["image"].to(self.device, non_blocking=True)
                labels = sample_batched["label"].to(self.device, non_blocking=True)

                preds = self.model(images)
                val_loss = self.loss_func(preds, labels)
                val_acc = calc_accuracy(preds, labels)

                val_loss_sum += val_loss * labels.size(0)
                val_acc_sum += val_acc
                n_vals += labels.size(0)

        self.val_time = time.time() - start_time
        self.val_loss = float(val_loss_sum / n_vals)
        self.val_acc = float(val_acc_sum / n_vals)

    def update_scheduler(self):
        "update the learning rate scheduler"
        self.scheduler.step()

    def show_result(self, current_epoch):
        self.now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        print(
            f"Epoch: [{current_epoch:03}/{self.cfg.default.epochs:03}] \t"
            + f"train loss: {self.train_loss:.6f} \t"
            + f"train acc: {self.train_acc:.6f} \t"
            + f"val loss: {self.val_loss:.6f} \t"
            + f"val acc: {self.val_acc:.6f} \t"
            + f"Now {self.now_time}"
        )

    def save_output(self, current_epoch):
        csv_file_path = self.cfg.out_dir + "/output.csv"
        if not os.path.exists(csv_file_path):
            f = open(csv_file_path, "w")
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epochs",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "objective",
                    "traiing_time",
                    "validation_time",
                    "run_time",
                    "lr",
                ]
            )
        else:
            f = open(csv_file_path, "a")
            writer = csv.writer(f)

        writer.writerow(
            [
                current_epoch,
                float(self.train_loss),
                float(self.train_acc),
                float(self.val_loss),
                float(self.val_acc),
                float(1 - self.val_acc),
                self.training_time,
                self.val_time,
                self.training_time + self.val_time,
                self.optimizer.param_groups[-1]["lr"],
            ]
        )
        f.close()
