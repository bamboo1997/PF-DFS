import os

from omegaconf import DictConfig, OmegaConf

from common import setup_config
from deep_neural_networks.create_dataloader import suggest_data_loader
from deep_neural_networks.utils import (
    fixed_r_seed,
    save_learner,
    setup_device,
    setup_learner,
    suggest_loss_func,
)


def main(cfg: DictConfig) -> None:
    # Random seed fixed
    fixed_r_seed(cfg)

    # Check device (GPU or CPU)
    device = setup_device(cfg)

    # Create data loader
    train_loader, val_loader, test_loader = suggest_data_loader(cfg)

    # Load the state of the previous epoch
    model_states_path = cfg.out_dir + "weights/latest_epochs.pth"
    cfg, model, optimizer, scheduler, previous_learned_epoch = setup_learner(
        cfg, model_states_path, device
    )

    # Setup loss function
    loss_func = suggest_loss_func(cfg, device)

    # Add epochs
    current_epoch = previous_learned_epoch

    # Train and val
    if cfg.execute_config_name == "classification":
        from deep_neural_networks.training_classification_model import (
            Classifier as Learner,
        )
    elif cfg.execute_config_name == "segmentation":
        from deep_neural_networks.training_segmentation_model import (
            Segmenter as Learner,
        )

    learner = Learner(cfg, model, device, optimizer, scheduler, loss_func)

    # Training
    learner.train_step(train_loader)
    # Validation
    learner.val_step(val_loader)
    # Update learning rate
    learner.update_scheduler()
    # Show training results
    learner.show_result(current_epoch)

    # Save incumbent state and performance
    save_learner(
        learner.cfg,
        learner.model,
        learner.optimizer,
        learner.scheduler,
        current_epoch + 1,
    )

    output_dir_path = cfg.out_dir + "/config_per_epoch"
    os.makedirs(output_dir_path, exist_ok=True)
    with open(output_dir_path + f"/epoch_{current_epoch:03}.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Save training results
    learner.save_output(current_epoch)


if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)
