import torch

from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

from fignet.logger import Logger
from fignet.simulator import LearnedSimulator
from fignet.trainer import Trainer

if __name__ == '__main__':

    mujoco_model_path = "test_models/mujoco_scene.xml"

    ref_sim = MjSim.from_xml_file(mujoco_model_path)
    ref_sim.add_render_context(MjRenderContextOffscreen(ref_sim, 0))

    latent_dim = 128

    config = {
        "data_path": "datasets/three_bodies_1m",
        "logging_folder": "log",
        "log_level": "debug",
        "lr_init": 1e-3,
        "lr_decay_rate": 0.1,
        "lr_decay_steps": 1e5,
        "loss_report_step": 10,
        "save_model_step": 1000,
        "eval_step": 1000,
        # "training_steps": 50,
        "clip_norm": 1e-2,
        "rollout_steps": 50,
        "run_validate": True,
        "num_eval_rollout": 10,
        "save_video": True,
    }
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using cuda')
    else:
        device = torch.device('cpu')
        print('using cpu')
    sim = LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=latent_dim,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        noise_std=1e-4,
        device=device,
    )
    logger = Logger(config)
    trainer = Trainer(sim=sim, ref_sim=ref_sim, logger=logger, config=config)
    trainer.train()
    