def polyak_average(source, target, polyak):
    for param, t_param in zip(source.parameters(), target.parameters()):
        t_param.data.copy_(
            t_param.data * polyak + param.data * (1 - polyak)
        )