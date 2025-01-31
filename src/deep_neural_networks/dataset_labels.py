def suggest_sementation_classes(cfg):
    if cfg.dataset.name == "CamVid":
        color2index = [
            (0, 0, 0),
            (128, 128, 128),
            (0, 128, 64),
            (192, 192, 128),
            (128, 64, 128),
            (0, 0, 192),
            (128, 128, 0),
            (192, 128, 128),
            (64, 64, 128),
            (64, 0, 128),
            (64, 64, 0),
            (0, 128, 192),
        ]
    return color2index
