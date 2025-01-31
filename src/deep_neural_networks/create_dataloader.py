import torch
from torch.utils.data import DataLoader


def suggest_data_loader(cfg):
    if cfg.execute_config_name == "classification":
        from deep_neural_networks.load_dataset_classification import suggest_dataset
    elif cfg.execute_config_name == "segmentation":
        from deep_neural_networks.load_dataset_segmentation import suggest_dataset

    train_dataset, val_dataset, test_dataset = suggest_dataset(cfg)
    collate_fn = None
    g = torch.Generator()
    # g.manual_seed(cfg.custom_seed.value)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.optimizer.hp.batch_size,
        num_workers=cfg.default.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
        # generator=g,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.optimizer.hp.batch_size,
        num_workers=cfg.default.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_fn,
        generator=g,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.optimizer.hp.batch_size,
        num_workers=cfg.default.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_fn,
        generator=g,
    )

    return train_dataloader, val_dataloader, test_dataloader
