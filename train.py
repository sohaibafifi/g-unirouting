import argparse
import os
import sys
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mavrp'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model',
                                     prog="train.py",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=int, default=0, help='The configuration to use')
    parser.add_argument('--print', action='store_true', help='Print the configuration')
    parser.add_argument('--tune', action='store_true', help='Tune the configuration')
    parser.add_help = True

    args = parser.parse_args()
    if args.print:
        from mavrp.configs.config import Config
        Config.print_table()
        exit(0)


    config_id = args.config

    from mavrp.configs.config import Config
    config = Config.all()[config_id]
    if config.deterministic_train :
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if args.tune and config.problem == 'MTVRP':
        print("You can't tune the MTVRP problem")
        exit(0)

    import lightning
    import torch
    from mavrp.env.train.trainers import ReinforceTrainer, ActorCriticTrainer
    print('Using PyTorch version : ', torch.__version__)
    print('Using Lightning version : ', lightning.__version__)
    print('Using CUDA : ', torch.cuda.is_available())
    print('Using accelerator : ', config.device)
    print('Number of threads : ', torch.get_num_threads())
    print('Host : ', os.uname().nodename)
    print('Working directory : ', config.working_dir)
    print('Seed : ', config.seed)

    torch.set_float32_matmul_precision('medium')

    model = ReinforceTrainer(config)
    if args.tune or config.tune:
        loaded_config = deepcopy(config)
        loaded_config.problem = 'MTVRP'
        loaded_config.tune = None
        folder = os.path.join(config.working_dir, 'MTVRP',
                              str(config.graph_size),
                              repr(loaded_config)
                              )
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist")
            exit(0)
        model_path = os.path.join(folder, 'checkpoint.ckpt')
        if not os.path.exists(model_path):
            print(f"Model {model_path} does not exist")
            exit(0)
        print(f"Tuning model from {model_path}")
        model.model.load_from_ckpt(model_path, baseline=True)
        model.baseline.update(model.model)

    torch.compile(model, fullgraph=True, dynamic=True).fit()
