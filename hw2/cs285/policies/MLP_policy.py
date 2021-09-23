import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure.utils import normalize


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
<<<<<<< HEAD
        # TODO: get this from HW1
=======
>>>>>>> others
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
<<<<<<< HEAD

        action_distribution = self(ptu.from_numpy(observation))
        action = action_distribution.sample()
        return  ptu.to_numpy(action)
=======
        
        observation = ptu.from_numpy(observation)
        action_distribution = self(observation)
        action = action_distribution.sample() # don't bother with rsample
        return ptu.to_numpy(action)

>>>>>>> others

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from hw1
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution



#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss

<<<<<<< HEAD
        log_pi = self.forward(observations).log_prob(actions)
        loss = -(log_pi * advantages).sum()

=======
        pi_st = super().forward (observations)
        logpi = pi_st.log_prob (actions)
        # print ('logpi shape: ',logpi.shape) # 1023
        # print ('observations shape: ',observations.shape) # 1023,4
        # print ('actions shape: ',actions.shape) #1023
        # print ('logpi type: ', type (logpi)) # torch.Tensor
        # print ('q_val type: ', type (q_values)) # ndarray

        # print ('baseline: ', self.baseline) # None
        # qvals and adv are vectors of 0. I dunno what to do

        # x - Q_t - b_t
        # min of negative is max
        # baseline?
        loss = -1 * torch.mean (logpi * advantages)
        # print ('logpi: ', logpi)
        # print ('q_val: ', q_values)
        # print ('adv: ', advantages)
        # print (loss)
        # raise NotImplementedError

        # TODO: optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
>>>>>>> others
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
<<<<<<< HEAD
            ## TODO: update the neural network baseline using the q_values as
            ## targets. The q_values should first be normalized to have a mean
            ## of zero and a standard deviation of one.
            targets = normalize(q_values, np.mean(q_values), np.std(q_values))
            targets = ptu.from_numpy(targets)

            ## HINT1: use self.baseline_optimizer to optimize the loss used for
                ## updating the baseline. Remember to 'zero_grad' first
            ## HINT2: You will need to convert the targets into a tensor using
                ## ptu.from_numpy before using it in the loss
            baseline_predictions = torch.squeeze(self.baseline(observations))
            assert baseline_predictions.shape == targets.shape

            baseline_loss = self.baseline_loss(baseline_predictions, targets)
            
=======
            ## TODO: normalize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            targets = utils.normalize(q_values, np.mean (q_values), np.std (q_values))
            targets = ptu.from_numpy(targets)

            ## TODO: use the `forward` method of `self.baseline` to get baseline predictions
            baseline_predictions = torch.squeeze(self.baseline (observations))
            
            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            # print (baseline_predictions.shape)
            # print (targets.shape)
            assert baseline_predictions.shape == targets.shape
            
            # TODO: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            # HINT: use `F.mse_loss`
            baseline_loss = self.baseline_loss (baseline_predictions, targets)

            # TODO: optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
>>>>>>> others
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }
        return train_log

<<<<<<< HEAD
    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())
=======
    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]
>>>>>>> others

