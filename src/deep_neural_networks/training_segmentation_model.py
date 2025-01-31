import numpy as np
import torch
from tqdm import tqdm
import datetime
import os
import csv
import time


class Evaluator(object):
    def __init__(self, num_class):

        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

        self.scaler = torch.cuda.amp.GradScaler(init_scale=2**12)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Segmenter(object):
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

        evaluator = Evaluator(self.cfg.dataset.numof_classes)
        evaluator.reset()

        start_time = time.time()
        bar = (
            tqdm(total=len(train_loader), leave=False) if self.cfg.default.bar else None
        )
        for i_batch, sample_batched in enumerate(train_loader):
            images = sample_batched["image"].to(self.device, non_blocking=True)
            labels = sample_batched["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=True):
                pred = self.model(images)
                train_loss = self.loss_func(pred, labels.long())

            self.scaler.scale(train_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            # update model parameter
            self.scaler.step(self.optimizer)
            # update scaler paramter
            self.scaler.update()

            pred = torch.argmax(pred, axis=1).data.cpu().numpy()
            evaluator.add_batch(labels.cpu().numpy(), pred)
            train_loss_sum += train_loss
            bar.update(1) if self.cfg.default.bar else None

        self.training_time = time.time() - start_time
        self.train_loss = float(train_loss_sum / len(train_loader))
        self.train_mIoU = float(evaluator.Mean_Intersection_over_Union())
        self.train_pixel_acc = float(evaluator.Pixel_Accuracy())
        bar.close() if self.cfg.default.bar else None

    def val_step(self, val_loader):
        self.model.eval()

        val_loss_sum = 0

        evaluator = Evaluator(self.cfg.dataset.numof_classes)
        evaluator.reset()

        start_time = time.time()
        with torch.no_grad():
            for i, sample_batched in enumerate(val_loader):
                images = sample_batched["image"].to(self.device, non_blocking=True)
                labels = sample_batched["label"].to(self.device, non_blocking=True)

                outputs = self.model(images)
                val_loss = self.loss_func(outputs, labels.long())

                outputs = torch.argmax(outputs, axis=1).data.cpu().numpy()
                evaluator.add_batch(labels.cpu().numpy(), outputs)

                val_loss_sum += val_loss
        self.val_time = time.time() - start_time
        self.val_loss = val_loss_sum / len(val_loader)
        self.val_mIoU = float(evaluator.Mean_Intersection_over_Union())
        self.val_pixel_acc = float(evaluator.Pixel_Accuracy())

    def update_scheduler(self):
        "update the learning rate scheduler"
        self.scheduler.step()

    def show_result(self, current_epoch):
        self.now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        print(
            f"Epoch: [{current_epoch:03}/{self.cfg.default.epochs:03}] \t"
            + f"train loss: {self.train_loss:.6f} \t"
            + f"train mIoU: {self.train_mIoU:.6f} \t"
            + f"train p-acc: {self.train_pixel_acc:.6f} \t"
            + f"val loss: {self.val_loss:.6f} \t"
            + f"val mIoU: {self.val_mIoU:.6f} \t"
            + f"val p-acc: {self.val_pixel_acc:.6f} \t"
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
                    "train_mIoU",
                    "train_pixel_acc",
                    "val_loss",
                    "val_mIoU",
                    "val_pixel_acc",
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
                float(self.train_mIoU),
                float(self.train_pixel_acc),
                float(self.val_loss),
                float(self.val_mIoU),
                float(self.val_pixel_acc),
                float(1 - self.val_mIoU),
                self.training_time,
                self.val_time,
                self.training_time + self.val_time,
                self.optimizer.param_groups[-1]["lr"],
            ]
        )
        f.close()
