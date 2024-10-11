import sys
sys.path.append('../') # ensure correct path dependency

import argparse
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import wandb
from src.schema import Schema
from trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dirpath", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=False, default=1)
    parser.add_argument("--subset_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    cs = ConfigStore.instance()
    cs.store(name="schema", node=Schema)
    initialize(version_base=None, config_path=args.config_dirpath)
    dict_cfg: DictConfig = compose(config_name=args.config_name)

    cfg: Schema = OmegaConf.to_object(dict_cfg)
    cfg.subset_size = args.subset_size
    cfg.seed = args.seed

    if cfg.wandb:
        wandb.login(key="") # use your own WanbB key
        #cfg.__dict__['num_params'] = sum(param.numel() for param in self.model.parameters())
        wandb.init(project="Pearl-Reddit-Binary", name=cfg.wandb_run_name, config=cfg.__dict__)

    for i in [5, 6, 7, 8, 9]:
        trainer = Trainer(cfg, args.gpu_id, args.dataset)
        print(i)
        loss = trainer.train(i)
        print(i, loss)


if __name__ == "__main__":
    main()
