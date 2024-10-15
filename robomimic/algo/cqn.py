"""
CQN: Coarse-to-Fine Q-Network

CQN implementation is based on the following paper: 
Younggyo Seo, Jafar Uruc, Stephen James - "Continuous Control with 
Coarse-to-Fine Reinforcement Learning" (CORL 2024)
Paper link: https://arxiv.org/abs/2407.07787
Original source code: https://github.com/younggyoseo/CQN
"""

from collections import OrderedDict

import torch
import torch.nn as nn

import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo
from robomimic.config import Config


@register_algo_factory_func("cqn")
def algo_config_to_class(algo_config: Config):
    """
    Maps algo config to the CQN algo class to instantiate, along with any 
    additional algo kwargs.

    Args:
        algo_config (OrderedDict): config for CQN algorithm

    Returns:
        CQN: CQN class
    """
    return CQN, {}


class CQN(PolicyAlgo, ValueAlgo):
    """
    Default CQN algorithm implementation.
    """

    def _create_networks(self):
        """
        Create networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        critic_class = ValueNets.DistributionalActionValueNetwork
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            value_bounds=self.algo_config.critic.value_bounds,
            num_atoms=self.algo_config.critic.distributional.num_atoms,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets["critic"] = critic_class(**critic_args)
        self.nets["critic_target"] = critic_class(**critic_args)

        # sync target networks at start of training
        with torch.no_grad():
            TorchUtils.hard_update(source=self.nets["critic"], target=self.nets["critic_target"])
    
    def get_action(self, obs_dict: dict, goal_dict: dict | None = None) -> torch.Tensor:
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        # TODO
        assert not self.nets.training
        pass

    def get_state_value(self, obs_dict: dict, goal_dict: dict | None = None) -> torch.Tensor:
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        assert not self.nets.training
        actions = self.get_action(obs_dict=obs_dict, goal_dict=goal_dict)
        return self.get_state_action_value(obs_dict=obs_dict, actions=actions, goal_dict=goal_dict)
    
    def get_state_action_value(
        self, obs_dict: dict, actions: torch.Tensor, goal_dict: dict | None = None
    ) -> torch.Tensor:
        """
        Get state-action value outputs.

        Args:
            obs_dict (dict): current observation
            actions (torch.Tensor): action
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        assert not self.nets.training
        return self.nets["critic"](obs_dict=obs_dict, goal_dict=goal_dict, actions=actions)
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out relevant 
        information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled from a data 
                loader

        Returns:
            input_batch (dict): processed and filtered batch that will be used 
                for training 
        """
        # TODO
        return batch
    
    def train_on_batch(self, batch: dict, epoch: int, validate: bool) -> dict:
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        # TODO
        assert validate or self.nets.training
        return OrderedDict()

    def log_info(self, info: dict) -> dict:
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss log (dict): name -> summary statistic
        """
        # TODO
        log = OrderedDict()

        # record current optimizer learning rates
        for k in self.optimizers:
            for i, param_group in enumerate(self.optimizers[k].param_groups):
                log["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]

        return log

    def on_epoch_end(self, epoch: int):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): current epoch
        """
        # TODO
        pass

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.nets.train()
        # target networks always in eval
        self.nets["critic_target"].eval()