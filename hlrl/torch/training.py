def polyak_average(source, target, polyak):
    """
    Copies parameters from the source to the target using the formula:
    t * (1 - polyak) + s * polyak.
    """
    for param, t_param in zip(source.parameters(), target.parameters()):
        t_param.data.copy_(
            t_param.data * (1 - polyak) + param.data * polyak
        )