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
import torch.nn.functional as F

from robomimic.models.value_nets import C2FNetwork
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
        critic_args = dict(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            levels=self.algo_config.critic.levels,
            bins=self.algo_config.critic.bins,
            value_bounds=self.algo_config.critic.value_bounds,
            input_bounds=(
                self.algo_config.critic.input_min, self.algo_config.critic.input_max
            ),
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            device=self.device,
        )
        self.nets["critic"] = C2FNetwork(**critic_args)
        self.nets["critic_target"] = C2FNetwork(**critic_args)
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
        assert not self.nets.training
        return self.nets["critic"](obs_dict, goal_dict)["action"]

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
        return self.nets["critic"](obs_dict, goal_dict, actions)["q_values"]
    
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
        info = OrderedDict()

        # batch variables
        s_batch = batch["obs"]
        a_batch = batch["actions"]
        r_batch = batch["rewards"]
        ns_batch = batch["next_obs"]
        dones = batch["dones"]
        goal_s_batch = batch["goal_obs"]

        # compute critic loss (RL + BC loss)
        loss_info = self._compute_critic_loss(
            states=s_batch,
            actions=a_batch,
            goal_states=goal_s_batch,
            rewards=r_batch,
            dones=dones,
            next_states=ns_batch,
        )
        info["critic/critic_loss"] = loss_info['loss'].detach().item()
        info["critic/rl_loss"] = loss_info['rl_loss'].detach().item()
        info["critic/bc_loss"] = loss_info['bc_loss'].detach().item()

        if not validate:
            critic_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets["critic"],
                optim=self.optimizers["critic"],
                loss=loss_info['loss'], 
                max_grad_norm=self.algo_config.critic.max_gradient_norm,
            )
            info["critic/critic_grad_norms"] = critic_grad_norms

        return info
    
    def _compute_critic_loss(
        self,
        states: dict,
        actions: torch.Tensor,
        goal_states: dict | None,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: dict
    ) -> dict:
        batch_size = actions.shape[0]

        # TODO: redo this but iterate through layer by layer and be super 
        # explicit about loss calculations

        # RL loss

        q_targets = self._get_target_values(next_states, goal_states, rewards, dones)
        rl_loss = self._compute_rl_loss(states, goal_states, q_targets)

        # BC loss

        low = torch.tensor(
            [self.algo_config.critic.input_min] * self.ac_dim
        ).to(self.device).float().repeat(batch_size, 1).to(self.device)
        high = torch.tensor(
            [self.algo_config.critic.input_max] * self.ac_dim
        ).to(self.device).float().repeat(batch_size, 1).to(self.device)

        # compute q values for expert action
        expert_dict = self.nets["critic"](states, goal_states, actions)
        q_expert = expert_dict['q_values']
        a_expert_enc = expert_dict['encoded_action']

        # compute q values for c2f network with margin
        margin = (1 - F.one_hot(
            a_expert_enc, num_classes=self.algo_config.critic.bins
        ).to(self.device)) * self.algo_config.optim_params.bc_margin
        q_values = torch.zeros(
            batch_size, self.ac_dim, self.algo_config.critic.levels
        ).to(self.device)

        # to only use margin when expert and c2f are looking at same bin:
        prev_wrong_bin = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

        for level in range(self.algo_config.critic.levels):
            prev_action = (low + high) / 2.
            bin_values = self.nets["critic"].forward_level(
                states, goal_states, level, prev_action
            )['bin_values']
            bin_values += margin[:, :, level] * ~prev_wrong_bin[:, None, None]
            bin_selection = bin_values.argmax(dim=-1)
            prev_wrong_bin += (bin_selection != a_expert_enc[:, :, level]).any(1)

            q_values[:, :, level] = torch.gather(
                bin_values, 2, bin_selection.unsqueeze(-1)
            ).squeeze(-1)

            low, high = C2FNetwork.zoom_in(
                low, high, bin_selection, self.algo_config.critic.bins
            )
        
        bc_loss = (q_values - q_expert).sum()

        rl_loss /= (batch_size * self.ac_dim * self.algo_config.critic.levels)
        bc_loss /= (batch_size * self.ac_dim * self.algo_config.critic.levels)
        loss = (
            self.algo_config.optim_params.bc_loss_weight * bc_loss +
            self.algo_config.optim_params.rl_loss_weight * rl_loss
        )

        return dict(loss=loss, rl_loss=rl_loss, bc_loss=bc_loss)
    
    def _compute_rl_loss(
        self,
        states: dict,
        goal_states: dict | None,
        q_targets: torch.Tensor,
    ):
        q_values = self.nets['critic'](states, goal_states)['q_values']
        return nn.MSELoss(reduction='sum')(q_values, q_targets)

    def _get_target_values(
        self,
        next_states: dict,
        goal_states: dict | None,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            q_targets = self.nets["critic_target"](next_states, goal_states)['q_values']
            rewards = rewards[:, None].expand(-1, *q_targets.shape[1:])
            dones = dones[:, None].expand(-1, *q_targets.shape[1:])
            return rewards + self.discount * (1. - dones) * q_targets

    def log_info(self, info: dict) -> dict:
        log = OrderedDict()

        # record current optimizer learning rates
        for k in self.optimizers:
            for i, param_group in enumerate(self.optimizers[k].param_groups):
                log["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]

        log.update(info)
        return log

    def on_epoch_end(self, epoch: int):
        pass

    def set_train(self):
        self.nets.train()
        # target networks always in eval
        self.nets["critic_target"].eval()

    def set_discount(self, discount: float):
        self.discount = discount