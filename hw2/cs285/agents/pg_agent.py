import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import normalize


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # step 1: calculate q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values = self.calculate_q_vals(rewards_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advantages = self.estimate_advantage(observations, q_values)

        # TODO: step 3: use all datapoints (s_t, a_t, q_t, adv_t) to update the PG actor/policy
        ## HINT: `train_log` should be returned by your actor update method
        train_log = self.actor.update (observations, actions, advantages, q_values=q_values)

        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)
        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.

        if not self.reward_to_go:
            q_values = np.concatenate([self._discounted_return(rewards) for rewards in rewards_list])

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            q_values = np.concatenate([self._discounted_cumsum(rewards) for rewards in rewards_list])

        return q_values

    def estimate_advantage(self, obs, q_values):

        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the baseline
        if self.nn_baseline:
            baselines_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the baseline and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero
            advantages = normalize(advantages, np.mean(advantages), np.std(advantages))

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def __sumt (self, rewards, t):
        T = len (rewards)
        sum = 0
        for t_p in range (t, T):
            n = self.gamma**(t_p-t) * rewards[t_p]
            sum += 0
        return sum

    def _discounted_return(self, rewards):
        """
            Helper function
            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T
            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # TODO: create list_of_discounted_returns
        # Hint: note that all entries of this output are equivalent
            # because each sum is from 0 to T (and doesnt involve t)
        
        # 0 to T
        idxs = np.arange (len (rewards))
    
        # discount gammas
        discounts = np.power (self.gamma, idxs)

        discounted_rewards = discounts * rewards

        list_of_discounted_returns = [np.sum (discounted_rewards)] * len (rewards)

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `list_of_discounted_returns`
        # HINT1: note that each entry of the output should now be unique,
            # because the summation happens over [t, T] instead of [0, T]
        # HINT2: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine

        list_of_discounted_cumsums = []

        for t in range (len (rewards)):
            # t' = t to T
            idxs = np.arange (t, len (rewards))

            # power = t' - t
            power_idxs = idxs - t
        
            # discount gammas
            discounts = np.power (self.gamma, power_idxs)

            # discounts * rewards from t to T
            discounted_rewards = discounts * rewards[idxs]

            sum_discounted_rewards = np.sum (discounted_rewards)

            list_of_discounted_cumsums.append (sum_discounted_rewards)

        return list_of_discounted_cumsums


