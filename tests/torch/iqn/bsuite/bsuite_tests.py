

def setup_experiment(args, bsuite_id, results_dir):
    # Initialize the environment
    bsuite_env = load_and_record_to_csv(
        bsuite_id, results_dir=results_dir, overwrite=True
    )
    gym_env = gym_wrapper.GymFromDMEnv(bsuite_env)
    env = GymEnv(gym_env)

    # The logger
    logger = args.logs_path
    logger = None if logger is None else TensorboardLogger(logger)
    
    # Initialize IQN
    activation_fn = nn.ReLU
    optim = partial(torch.optim.Adam, lr=args.lr)

    # Setup networks
    if args.num_layers > 1:
        qfunc_num_layers = 1
        autoencoder_out_n = args.hidden_size
    else:
        qfunc_num_layers = 0
        autoencoder_out_n = env.action_space[0]

    qfunc = LinearPolicy(
        args.hidden_size, env.action_space[0], args.hidden_size,
        qfunc_num_layers, activation_fn
    )

    if args.recurrent:
        num_lin_before = 1 if args.num_layers > 2 else 0

        autoencoder = LSTMPolicy(
            env.state_space[0], autoencoder_out_n, args.hidden_size,
            num_lin_before, args.hidden_size, max(args.num_layers - 2, 1),
            0, 0, activation_fn
        )
        # TODO CREATE RAINBOW IQN INSTANCE HERE ONCE IMPLEMENTED
    else:
        autoencoder = LinearPolicy(
            env.state_space[0], autoencoder_out_n, args.hidden_size,
            min(args.num_layers - 1, 1), activation_fn
        )

        if args.num_layers > 1:
            autoencoder = nn.Sequential(autoencoder, activation_fn())

        algo = RainbowIQN(
            args.hidden_size, autoencoder, qfunc, args.discount,
            args.polyak, args.n_quantiles, args.embedding_dim,
            args.huber_threshold, args.target_update_interval, optim,
            optim, logger
        )

    algo = algo.to(torch.device(args.device))

    if args.load_path is not None:
        algo.load(args.load_path)

    # Create agent class
    agent = OffPolicyAgent(
        env, algo, args.render, silent=True, logger=logger
    )

    if args.recurrent:
        agent = SequenceInputAgent(agent, device=args.device)
        agent = TorchRecurrentAgent(agent)
    else:
        agent = TorchRLAgent(agent, batch_state=False, device=args.device)

    algo.create_optimizers()

    algo.train()
    algo.share_memory()
    # Experience replay
    if args.recurrent:
        experience_replay = TorchR2D2(
            args.er_capacity, args.er_alpha, args.er_beta,
            args.er_beta_increment, args.er_epsilon,
            args.max_factor
        )
        agent = ExperienceSequenceAgent(
            agent, args.burn_in_length + args.sequence_length,
            args.burn_in_length
        )
    else:
        experience_replay = TorchPER(
            args.er_capacity, args.er_alpha, args.er_beta,
            args.er_beta_increment, args.er_epsilon
        )

    agents = [agent]

    agent_pool = AgentPool(agents)

    return algo, agent_pool, env, experience_replay

def run_experiment(args, algo, agent_pool, env, experience_replay,
    experience_queue, mp_event):
    agent_procs = agent_pool.train_process(
        1000, args.decay, args.n_steps, experience_queue,
        mp_event
    )

    # Start the worker for the model
    worker = Worker(algo, experience_replay, experience_queue)
    worker.train(
        agent_procs, mp_event, args.batch_size, args.start_size,
        args.save_path, args.save_interval
    )

if __name__ == "__main__":
    from bsuite import sweep, load_and_record_to_csv
    from bsuite.utils import gym_wrapper
    from bsuite.logging import csv_load
    from bsuite.experiments import summary_analysis
    from bsuite.experiments.bandit import analysis as bandit_analysis
    from matplotlib.pyplot import savefig as save_plt_fig
    from pathlib import Path

    mp.set_sharing_strategy('file_system')

    results_dir = "./bsuite_results/"

    csv_dir = results_dir + "csv/"
    plot_dir = results_dir + "plots/"

    plot_format = "png"
    plot_ext = "." + plot_format

    #envs = sweep.BANDIT
    envs = ["bandit/0"]

    for bsuite_id in envs:
        b_env = 'bandit'
        env_plot_path = Path(plot_dir + bsuite_id.replace("/", "-") + "/")
        env_plot_path.mkdir(parents=True, exist_ok=True)
        env_plot_path = str(env_plot_path.resolve())

        args = get_args()

        # Initialize the environment
        bsuite_env = load_and_record_to_csv(
            bsuite_id, results_dir=results_dir, overwrite=True
        )
        gym_env = gym_wrapper.GymFromDMEnv(bsuite_env)
        env = GymEnv(gym_env)

    

        algo, agent_pool, env, experience_replay = setup_experiment(
            args, bsuite_id, csv_dir
        )
        experience_queue = mp.Queue()
        mp_event = mp.Event()

        run_experiment(
            args, algo, agent_pool, env, experience_replay, experience_queue,
            mp_event
        )

        # Analyze performance
        df, sweep_vars = csv_load.load_bsuite(csv_dir)

        bandit_df = df[df.bsuite_env == b_env].copy()

        bsuite_score = summary_analysis.bsuite_score(df, sweep_vars)
        bsuite_summary = summary_analysis.ave_score_by_tag(
            bsuite_score, sweep_vars
        )
        print(bsuite_score)

        radar_plot = summary_analysis.bsuite_radar_plot(
            bsuite_summary, sweep_vars
        )
        save_plt_fig(
            env_plot_path + "/radar" + plot_ext, format=plot_format
        )
    
        score_plot = summary_analysis.plot_single_experiment(
            bsuite_score, b_env, sweep_vars
        )
        score_plot.save(
            env_plot_path + "/bsuite-score" + plot_ext, format=plot_format
        )

        regret_plot = bandit_analysis.plot_learning(bandit_df, sweep_vars)
        regret_plot.save(
            env_plot_path + "/average-regret" + plot_ext, format=plot_format
        )