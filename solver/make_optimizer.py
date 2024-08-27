import torch

def make_optimizer(cfg, model, center_criterion):
    """
    Creates and configures the optimizer for the model and center criterion based on the configuration.

    This function iterates over the model's parameters to set their learning rate and weight decay based on the configuration. It then creates the optimizer for the model and center criterion using the specified optimizer name and parameters.

    Args:
        cfg (dict): Configuration dictionary containing model and training parameters.
        model (nn.Module): The neural network model to optimize.
        center_criterion (nn.Module): The center criterion module to optimize.

    Returns:
        Tuple: A tuple containing the optimizer for the model and the optimizer for the center criterion.
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key or 'binary' in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_name = cfg.SOLVER.OPTIMIZER_NAME
    if optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, optimizer_name)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
