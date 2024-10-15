"""
Config for CQN.
"""

from robomimic.config.base_config import BaseConfig


class CQNConfig(BaseConfig):
    ALGO_NAME = "cqn"

    def experiment_config(self):
        super(CQNConfig, self).experiment_config()

    def train_config(self):
        super(CQNConfig, self).train_config()
        self.train.output_dir = f"../../{self.algo_name}_trained_models"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        self.algo.discount = 1.0   # discount factor
        self.algo.n_step = 3       # for using n-step returns in TD-updates

        # optimizer parameters
        self.algo.optim_params.critic.optimizer_type = "adamw"
        self.algo.optim_params.critic.learning_rate.initial = 3e-4
        self.algo.optim_params.critic.regularization.L2 = 0.0
        self.algo.optim_params.bc_loss_weight = 1.0
        self.algo.optim_params.rl_loss_weight = 0.1
        # target network parameters
        self.algo.target_tau = 0.01

        ##################### Critic Network Config #####################
        self.algo.critic.layer_dims = (256, 256)
        self.algo.critic.value_bounds = (-1, 1)

        # C2F parameters
        self.algo.critic.input_min = -1    # action lower bound
        self.algo.critic.input_max = 1     # action upper bound
        self.algo.critic.levels = 3        # number of levels in the C2F hierarchy
        self.algo.critic.bins = 5          # number of bins in each level

        # C51 (distributional) parameters
        self.algo.critic.distributional.enabled = False
        self.algo.critic.distributional.num_atoms = 51

    def observation_config(self):
        super(CQNConfig, self).observation_config()

        # observation modalities
        self.observation.modalities.obs.low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ]
        self.observation.modalities.obs.rgb = [
            # "agentview_image",
            # "robot0_eye_in_hand_image"
        ]
    
    def meta_config(self):
        super(CQNConfig, self).meta_config()