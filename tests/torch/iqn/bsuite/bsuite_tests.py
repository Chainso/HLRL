from hlrl.core.distributed import ApexWorker

def run_experiment(args, algo, agent_pool, env, experience_replay,
    experience_queue, done_event):
    agent_procs = agent_pool.train_process(
        args.episodes, args.decay, args.n_steps, experience_queue,
        done_event
    )

    # Start the worker for the model
    worker = ApexWorker(algo, experience_replay, experience_queue)
    worker.train(
        agent_procs, done_event, args.batch_size, args.start_size,
        args.save_path, args.save_interval
    )

if __name__ == "__main__":
    import torch.multiprocessing as mp
    import gym
    
    from bsuite import sweep, load_and_record_to_csv
    from bsuite.utils import gym_wrapper
    from bsuite.logging import csv_load
    from bsuite.experiments import summary_analysis
    from bsuite.experiments.bandit import analysis as bandit_analysis
    from matplotlib.pyplot import savefig as save_plt_fig
    from pathlib import Path

    from hlrl.core.envs.gym import GymEnv
    from tests.torch.iqn.get_args import get_args
    from tests.torch.iqn.setup_test import setup_test

    mp.set_sharing_strategy('file_system')

    results_dir = "./results/"

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
            bsuite_id, results_dir=csv_dir, overwrite=True
        )
        gym_env = gym_wrapper.GymFromDMEnv(bsuite_env)
        env = GymEnv(gym_env)

        algo, agent_pool, env, experience_replay = setup_test(args, env)

        experience_queue = mp.Queue()
        done_event = mp.Event()

        run_experiment(
            args, algo, agent_pool, env, experience_replay, experience_queue,
            done_event
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