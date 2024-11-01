"""
Contains torch Modules for value networks. These networks take an 
observation dictionary as input (and possibly additional conditioning, 
such as subgoal or goal dictionaries) and produce value or 
action-value estimates or distributions.
"""
import numpy as np
from collections import OrderedDict
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.obs_nets import MIMO_MLP
from robomimic.models.distributions import DiscreteValueDistribution


class ValueNetwork(MIMO_MLP):
    """
    A basic value network that predicts values from observations.
    Can optionally be goal conditioned on future observations.
    """
    def __init__(
        self,
        obs_shapes,
        mlp_layer_dims,
        value_bounds=None,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes. 

            value_bounds (tuple): a 2-tuple corresponding to the lowest and highest possible return
                that the network should be possible of generating. The network will rescale outputs
                using a tanh layer to lie within these bounds. If None, no tanh re-scaling is done.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.value_bounds = value_bounds
        if self.value_bounds is not None:
            # convert [lb, ub] to a scale and offset for the tanh output, which is in [-1, 1]
            self._value_scale = (float(self.value_bounds[1]) - float(self.value_bounds[0])) / 2.
            self._value_offset = (float(self.value_bounds[1]) + float(self.value_bounds[0])) / 2.

        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(ValueNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict values, but may instead predict the parameters
        of a value distribution.
        """
        return OrderedDict(value=(1,))

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [1]

    def forward(self, obs_dict, goal_dict=None):
        """
        Forward through value network, and then optionally use tanh scaling.
        """
        values = super(ValueNetwork, self).forward(obs=obs_dict, goal=goal_dict)["value"]
        if self.value_bounds is not None:
            values = self._value_offset + self._value_scale * torch.tanh(values)
        return values

    def _to_string(self):
        return "value_bounds={}".format(self.value_bounds)


class ActionValueNetwork(ValueNetwork):
    """
    A basic Q (action-value) network that predicts values from observations
    and actions. Can optionally be goal conditioned on future observations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        value_bounds=None,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes. 

            value_bounds (tuple): a 2-tuple corresponding to the lowest and highest possible return
                that the network should be possible of generating. The network will rescale outputs
                using a tanh layer to lie within these bounds. If None, no tanh re-scaling is done.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # add in action as a modality
        new_obs_shapes = OrderedDict(obs_shapes)
        new_obs_shapes["action"] = (ac_dim,)
        self.ac_dim = ac_dim

        # pass to super class to instantiate network
        super(ActionValueNetwork, self).__init__(
            obs_shapes=new_obs_shapes,
            mlp_layer_dims=mlp_layer_dims,
            value_bounds=value_bounds,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def forward(self, obs_dict, acts, goal_dict=None):
        """
        Modify forward from super class to include actions in inputs.
        """
        inputs = dict(obs_dict)
        inputs["action"] = acts
        return super(ActionValueNetwork, self).forward(inputs, goal_dict)

    def _to_string(self):
        return "action_dim={}\nvalue_bounds={}".format(self.ac_dim, self.value_bounds)


class DistributionalActionValueNetwork(ActionValueNetwork):
    """
    Distributional Q (action-value) network that outputs a categorical distribution over
    a discrete grid of value atoms. See https://arxiv.org/pdf/1707.06887.pdf for 
    more details.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        value_bounds,
        num_atoms,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes. 

            value_bounds (tuple): a 2-tuple corresponding to the lowest and highest possible return
                that the network should be possible of generating. This defines the support
                of the value distribution.

            num_atoms (int): number of value atoms to use for the categorical distribution - which
                is the representation of the value distribution.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to DistributionalActionValueNetwork
        self.num_atoms = num_atoms
        self._atoms = np.linspace(value_bounds[0], value_bounds[1], num_atoms)

        # pass to super class to instantiate network
        super(DistributionalActionValueNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            value_bounds=value_bounds,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Network outputs log probabilities for categorical distribution over discrete value grid.
        """
        return OrderedDict(log_probs=(self.num_atoms,))

    def forward_train(self, obs_dict, acts, goal_dict=None):
        """
        Return full critic categorical distribution.

        Args:
            obs_dict (dict): batch of observations
            acts (torch.Tensor): batch of actions
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            value_distribution (DiscreteValueDistribution instance)
        """

        # add in actions
        inputs = dict(obs_dict)
        inputs["action"] = acts

        # network returns unnormalized log probabilities (logits) for each of the value atoms
        logits = MIMO_MLP.forward(self, obs=inputs, goal=goal_dict)["log_probs"]

        # turn these logits into a categorical distribution over the value atoms.
        # (unsqueeze to make sure atoms are compatible with batch operations)
        value_atoms = torch.Tensor(self._atoms).unsqueeze(0).to(logits.device)
        return DiscreteValueDistribution(values=value_atoms, logits=logits)

    def forward(self, obs_dict, acts, goal_dict=None):
        """
        Return mean of critic categorical distribution. Useful for obtaining
        point estimates of critic values.

        Args:
            obs_dict (dict): batch of observations
            acts (torch.Tensor): batch of actions
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            mean_value (torch.Tensor): expectation of value distribution
        """
        vd = self.forward_train(obs_dict=obs_dict, acts=acts, goal_dict=goal_dict)
        return vd.mean()

    def _to_string(self):
        return "action_dim={}\nvalue_bounds={}\nnum_atoms={}".format(self.ac_dim, self.value_bounds, self.num_atoms)


class C2FNetwork(MIMO_MLP):
    """
    A coarse-to-fine Q-network. For a C2F Q-network with L layers and B bins 
    per layer, the output of each layer is a vector of length B indicating the 
    value of the Q-function at that bin. This network implements the entire 
    hierarchy.
    """

    def __init__(
        self,
        obs_shapes: OrderedDict,
        ac_dim: int,
        mlp_layer_dims: list[int],
        levels: int,
        bins: int,
        input_bounds: tuple[int, int],
        value_bounds: tuple[int, int] | None = None,
        goal_shapes: OrderedDict | None = None,
        encoder_kwargs: dict | None = None,
    ):
        self.levels = levels
        self.bins = bins
        self.ac_dim = ac_dim
        self.input_min = input_bounds[0]
        self.input_max = input_bounds[1]

        self.value_bounds = value_bounds
        if self.value_bounds is not None:
            # convert [lb, ub] to a scale and offset for the tanh output, which is in [-1, 1]
            self._value_scale = (float(self.value_bounds[1]) - float(self.value_bounds[0])) / 2.
            self._value_offset = (float(self.value_bounds[1]) + float(self.value_bounds[0])) / 2.

        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        observation_group_shapes["obs"]["prev_action"] = (ac_dim,)  # previous action
        observation_group_shapes["obs"]["level"] = (levels,)  # one-hot encoding of level

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()
        
        output_shapes = self._get_layer_output_shapes()
        super(C2FNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_layer_output_shapes(self) -> OrderedDict:
        return OrderedDict(bin_values=(self.levels, self.bins))

    def output_shape(self, input_shape : Iterable[int] | None = None) -> list[int]:
        return dict(layer_values=(self.levels, self.ac_dim, self.bins), action=(self.ac_dim,))
    
    def forward(
        self, 
        obs_dict: OrderedDict, 
        goal_dict: OrderedDict | None = None,
        action: torch.Tensor | None = None
    ) -> dict:
        # TODO
        if action is not None:
            # TODO: encode the action
            pass

        # low and high initialized to bounds of input
        low = torch.tensor([self.input_min] * self.ac_dim).float()
        high = torch.tensor([self.input_max] * self.ac_dim).float()

        # iterate through levels
        for level in range(self.levels):
            # get Q-value for current level
            prev_action = (low + high) / 2.
            obs_dict["prev_action"] = prev_action
            obs_dict["level"] = F.one_hot(torch.tensor(level), self.levels).float()
            bin_values = super(C2FNetwork, self).forward(obs_dict=obs_dict, goal_dict=goal_dict)
            exit()
        
    @staticmethod
    def encode_action(
        continuous_action: torch.Tensor,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        levels: int,
        bins: int
    ) -> torch.Tensor:
        """
        Encode continuous action into discrete action (bin selections).

        Args:
            continuous_action (torch.Tensor): shape (batch_size, ac_dim)
            action_min (torch.Tensor): shape (ac_dim)
            action_max (torch.Tensor): shape (ac_dim)
            levels (int): number of levels in the C2F hierarchy
            bins (int): number of bins in each level
        
        Returns:
            discrete_action (torch.Tensor): shape (batch_size, ac_dim, levels)
        """
        low = action_min.clone()
        high = action_max.clone()

        discrete_action = torch.zeros(
            continuous_action.shape[0], continuous_action.shape[1], levels
        ).to(continuous_action.device)
        for l in range(levels):
            # put continuous values into bins
            slice_range = (high - low) / bins
            idx = torch.floor((continuous_action - low) / slice_range).to(torch.int)
            idx = torch.clamp(idx, 0, bins - 1)
            discrete_action[:, :, l] = idx

            # compute new low and high for each bin (zoom in)
            low = low + idx * slice_range
            high = low + slice_range
        return discrete_action
    
    @staticmethod
    def decode_action(
        discrete_action: torch.Tensor,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        levels: int,
        bins: int
    ):
        """
        Decode discrete action (bin selections) into continuous action.

        Args:
            discrete_action (torch.Tensor): shape (batch_size, ac_dim, levels)
            action_min (torch.Tensor): shape (ac_dim)
            action_max (torch.Tensor): shape (ac_dim)
            levels (int): number of levels in the C2F hierarchy
            bins (int): number of bins in each level
        """
        low = action_min.clone()
        high = action_max.clone()
        for l in range(levels):
            slice_range = (high - low) / bins
            low = low + discrete_action[:, :, l] * slice_range
            high = low + slice_range
        return (low + high) / 2.

    def _to_string(self) -> str:
        msg = f"levels={self.levels}"
        msg += f"\nbins={self.bins}"
        msg += f"\naction_dim={self.ac_dim}"
        msg += f"\nvalue_bounds={self.value_bounds}"
        return msg