from torch import optim


def select_lr_scheduler(params: dict, lr_scheduler: str, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    """Select and return the appropriate learning rate scheduler based on the provided parameters.

    Args:
        params (dict): parameters for the learning rate scheduler.
        lr_scheduler (str): the type of learning rate scheduler to use, e.g., "step", "lambda", or "none".
        optimizer (optim.Optimizer): the optimizer for which the scheduler is being created.

    Raises:
        ValueError: if an unknown learning rate scheduler type is specified.

    Returns:
        optim.lr_scheduler._LRScheduler: the selected learning rate scheduler instance.
    """

    scheduler = None
    if lr_scheduler == "step":
        step_size = int(params["lr_schedulers"][lr_scheduler]["step_size"])
        gamma = float(params["lr_schedulers"][lr_scheduler]["gamma"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if lr_scheduler == "lambda":
        factor = float(params["lr_schedulers"][lr_scheduler]["factor"])
        step_size = int(params["lr_schedulers"][lr_scheduler]["step_size"])
        lr_lambda = lambda step: factor ** -(step // step_size)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif lr_scheduler != "none":
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")
    
    return scheduler