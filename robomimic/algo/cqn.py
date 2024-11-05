"""
CQN: Coarse-to-Fine Q-Network

CQN implementation is based on the following paper: 
Younggyo Seo, Jafar Uruc, Stephen James - "Continuous Control with 
Coarse-to-Fine Reinforcement Learning" (CORL 2024)
Paper link: https://arxiv.org/abs/2407.07787
Original source code: https://github.com/younggyoseo/CQN

To test:
# download lift datasets
cd <robomimic_root>
python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph --hdf5_types raw
python robomimic/scripts/dataset_states_to_obs.py --dataset datasets/lift/ph/demo_v141.hdf5 --output_name low_dim_v141.hdf5 --done_mode 2
python robomimic/scripts/dataset_states_to_obs.py --dataset datasets/lift/ph/demo_v141.hdf5 --output_name image_v141.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# generate config files
python robomimic/scripts/generate_config_templates.py

# train
python robomimic/scripts/train.py --config robomimic/exps/templates/cqn.json --dataset datasets/lift/ph/low_dim_v141.hdf5
python robomimic/scripts/train.py --config robomimic/exps/templates/cqn.json --dataset datasets/lift/ph/image_v141.hdf5
"""

from collections import OrderedDict

import torch
import torch.nn as nn

import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo
from robomimic.config import Config
import robomimic.utils.tensor_utils as TensorUtils


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
        critic_class = ValueNets.C2FNetwork
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            levels=self.algo_config.critic.levels,
            bins=self.algo_config.critic.bins,
            value_bounds=self.algo_config.critic.value_bounds,
            input_bounds=(self.algo_config.critic.input_min, self.algo_config.critic.input_max),
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            device=self.device,
        )
        self.nets["critic"] = critic_class(**critic_args)
        self.nets["critic_target"] = critic_class(**critic_args)
        self.nets = self.nets.to(self.device).float()

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
        # TODO
    
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
        # TODO
    
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
        # TODO: check if this is correct
        input_batch = dict()

        # n-step returns
        n_step = self.algo_config.n_step
        assert batch["actions"].shape[1] >= n_step

        # remove temporal batches for all
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["next_obs"] = {k: batch["next_obs"][k][:, n_step - 1, :] for k in batch["next_obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]

        # note: ensure scalar signals (rewards, done) retain last dimension of 1 to be compatible with model outputs

        # single timestep reward is discounted sum of intermediate rewards in sequence
        reward_seq = batch["rewards"][:, :n_step]
        discounts = torch.pow(self.algo_config.discount, torch.arange(n_step).float()).unsqueeze(0)
        input_batch["rewards"] = (reward_seq * discounts).sum(dim=1).unsqueeze(1)

        # discount rate will be gamma^N for computing n-step returns
        new_discount = (self.algo_config.discount ** n_step)
        self.set_discount(new_discount)

        # consider this n-step seqeunce done if any intermediate dones are present
        done_seq = batch["dones"][:, :n_step]
        input_batch["dones"] = (done_seq.sum(dim=1) > 0).float().unsqueeze(1)

        if self.algo_config.infinite_horizon:
            # scale terminal rewards by 1 / (1 - gamma) for infinite horizon MDPs
            done_inds = input_batch["dones"].round().long().nonzero(as_tuple=False)[:, 0]
            if done_inds.shape[0] > 0:
                input_batch["rewards"][done_inds] = input_batch["rewards"][done_inds] * (1. / (1. - self.discount))
        
        # print('\n\n\n\n')
        # print(f'{input_batch["obs"]["object"].shape=}')
        # print(f'{input_batch["obs"]["agentview_image"].shape=}')
        # print(f'{input_batch["obs"]["robot0_eye_in_hand_image"].shape=}')
        # print(f'{input_batch["obs"]["robot0_eef_pos"].shape=}')
        # print(f'{input_batch["obs"]["robot0_eef_quat"].shape=}')
        # print(f'{input_batch["obs"]["robot0_gripper_qpos"].shape=}')
        # print(f'{input_batch["actions"].shape=}')
        # print(f'{input_batch["rewards"].shape=}')
        # print(f'{input_batch["dones"].shape=}')
        # print('\n\n\n\n')

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))
    
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
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate)
            
            # update critic:
            critic_info = self._train_critic_on_batch(batch, epoch, validate)
            info.update(critic_info)

            # update critic target
            with torch.no_grad():
                TorchUtils.soft_update(
                    source=self.nets["critic"],
                    target=self.nets["critic_target"],
                    tau=self.algo_config.target_tau
                )
            
        return info
    
    def _train_critic_on_batch(self, batch: dict, epoch: int, validate: bool) -> dict:
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
        info = OrderedDict()

        # batch variables
        s_batch = batch["obs"]
        a_batch = batch["actions"]
        r_batch = batch["rewards"]
        ns_batch = batch["next_obs"]
        goal_s_batch = batch["goal_obs"]

        # 1 if not done, 0 otherwise
        done_mask_batch = 1. - batch["dones"]
        info["done_masks"] = done_mask_batch

        # Bellman backup for Q-targets
        q_targets = self._get_target_values(
            next_states=ns_batch, 
            goal_states=goal_s_batch, 
            rewards=r_batch, 
            dones=done_mask_batch,
        )
        info["critic/q_targets"] = q_targets

        # Train critics using this set of targets for regression
        critic_loss = self._compute_critic_loss(
            critic=self.nets["critic"],
            states=s_batch,
            actions=a_batch,
            goal_states=goal_s_batch,
            q_targets=q_targets,
        )
        info["critic/critic_loss"] = critic_loss

        if not validate:
            critic_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets["critic"],
                optim=self.optimizers["critic"],
                loss=critic_loss, 
                max_grad_norm=self.algo_config.critic.max_gradient_norm,
            )
            info["critic/critic_grad_norms"] = critic_grad_norms

        return info
    
    def _get_target_values(
        self,
        next_states: dict,
        goal_states: dict | None,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute target Q-values for the critic.

        Args:
            next_states (dict): next states
            goal_states (dict): goal states
            rewards (torch.Tensor): rewards
            dones (torch.Tensor): dones

        Returns:
            q_targets (torch.Tensor): target Q-values
        """
        # TODO
        with torch.no_grad():
            target_dict = self.nets["critic_target"](next_states, goal_states)
            exit()
        pass
    
    def _compute_critic_loss(
        self,
        critic: nn.Module,
        states: dict,
        actions: torch.Tensor,
        goal_states: dict | None,
        q_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute critic loss.

        Args:
            critic (nn.Module): critic network

            states (dict): current states

            actions (torch.Tensor): actions

            goal_states (dict): goal states

            q_targets (torch.Tensor): target Q-values

        Returns:
            critic_loss (torch.Tensor): critic loss
        """
        
        q_values = 0. # TODO
        critic_loss = nn.MSELoss()(q_values, q_targets) # TODO: maybe use Huber?
        return critic_loss

    def log_info(self, info: dict) -> dict:
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            log (dict): name -> summary statistic
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

    def set_discount(self, discount: float):
        """
        Useful function to modify discount factor if necessary (e.g. for n-step returns).
        """
        self.discount = discount