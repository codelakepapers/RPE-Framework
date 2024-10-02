import sys
sys.path.append('../') # ensure correct path dependency

import argparse
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.schema import Schema
from trainer import Trainer
import wandb
from root import root

# Runner for k,d,#MLP hyperparam search
def main() -> None:
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dirpath", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=False, default=0)
    parser.add_argument("--subset_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    cs = ConfigStore.instance()
    cs.store(name="schema", node=Schema)
    initialize(version_base=None, config_path=args.config_dirpath)
    dict_cfg: DictConfig = compose(config_name=args.config_name)

    cfg: Schema = OmegaConf.to_object(dict_cfg)
    cfg.subset_size = args.subset_size

    cfg.dataset = args.dataset

    # wandb should be false!
    # k: RAND_k
    # D_pe: pe_dims
    # D_node: node_emb_dims
    # M: RAND_mlp_nlayers: 1
    # H: RAND_mlp_hid: 16   
    # O: RAND_mlp_out: 16
    # K, M, H, O, D_pe, D_node
    hyperparams = [  [14, 1, 32, 32, 32, 64], # BASIS IS TRU
                        [14, 2, 32, 32, 32, 64]]
                        #[14, 1, 37, 37, 37, 128]]
    '''hyperparams = [#[12, 1, 32, 32, 32, 64],  #BASIS IS TRUE
                   [12, 2, 32, 32, 32, 64],
                   [16, 1, 37, 37, 37, 64]]'''
    '''hyperparams = [[16, 1, 37, 37, 37, 64],   # BASIS FALSE
                   #[16, 1, 37, 37, 37, 128],
                   [16, 2, 32, 32, 32, 64]]'''
    '''hyperparams = [[16, 2, 37, 37, 37, 64],
                   #[16, 2, 37, 37, 37, 128],
                   [14, 2, 32, 32, 32, 64]]
                   #[14, 2, 37, 37, 37, 128]]'''
    
    Final_test = 100
    best_hyperparam = []
    cfg.seed = args.seed
    for hyperparam in hyperparams:
        cfg.RAND_k = hyperparam[0]
        cfg.RAND_mlp_nlayers = hyperparam[1]
        cfg.RAND_mlp_hid = hyperparam[2]
        cfg.RAND_mlp_out = hyperparam[3]
        cfg.pe_dims = hyperparam[4]
        cfg.node_emb_dims = hyperparam[5]
        if cfg.wandb:
            wandb.login(key="") # use your own WanbB key
            #cfg.__dict__['num_params'] = sum(param.numel() for param in self.model.parameters())
            name = 'K='+str(hyperparam[0])+'M='+str(hyperparam[1])+'H='+str(hyperparam[2])+'O='+str(hyperparam[3])+'D_pe='+str(hyperparam[4])+'D_node='+str(hyperparam[5])
            wandb.init(dir=root("."), project="SPE-KD-Assay", name=name, config=cfg.__dict__)
        trainer = Trainer(cfg, args.gpu_id)
        test_loss_best = trainer.train()
        print("Hyperparam ", hyperparam, " | loss: ", test_loss_best)
        if test_loss_best < Final_test:
            Final_test = test_loss_best
            best_hyperparam = hyperparam
    print("Best hyperparam (K, M, H, O, D_pe, D_node) is: ", best_hyperparam, "| loss: ", Final_test)

if __name__ == "__main__":
    main()