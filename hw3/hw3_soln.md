# HW3 solutions
These solutions only show the relevant functions and the ones that were specific to this assignment.

## `agents/dqn_agent.py`
```
def step_env(self):
    """
    Step the env and store the transition
    At the end of this block of code, the simulator should have been
    advanced one step, and the replay buffer should contain one more transition.
    Note that self.last_obs must always point to the new latest observation.
    """

    # TODO store the latest observation into the replay buffer
    # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
        # in dqn_utils.py
    self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

    eps = self.exploration.value(self.t)

    # TODO use epsilon greedy exploration when selecting action
    # HINT: take random action
        # with probability eps (see np.random.random())
        # OR if your current step number (see self.t) is less that self.learning_starts
    perform_random_action = np.random.random() < eps or self.t < self.learning_starts

    if perform_random_action:
        action = self.env.action_space.sample()
    else:
        # HINT: Your actor will take in multiple previous observations ("frames") in order
            # to deal with the partial observability of the environment. Get the most recent # `frame_history_len` observations using functionality from the replay buffer,
            # and then use those observations as input to your actor.
        processed = self.replay_buffer.encode_recent_observation()
        action = self.actor.get_action(processed)

    # TODO take a step in the environment using the action from the policy
    # HINT1: remember that self.last_obs must always point to the newest/latest observation
    # HINT2: remember the following useful function that you've seen before:
        #obs, reward, done, info = env.step(action)
    next_obs, reward, done, info = self.env.step(action)
    self.last_obs = next_obs.copy()

    # TODO store the result of taking this action into the replay buffer
    # HINT1: see replay buffer's store_effect function
    # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
    self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

    # TODO if taking this step resulted in done, reset the env (and the latest observation)
    if done:
        self.last_obs = self.env.reset()

def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    log = {}
    if (self.t > self.learning_starts
        and self.t % self.learning_freq == 0
        and self.replay_buffer.can_sample(self.batch_size)
    ):
    
        # TODO fill in the call to the update function using the appropriate tensors
        log = self.critic.update(
            ob_no, ac_na, next_ob_no, re_n, terminal_n,
        )

        # TODO update the target network periodically 
        # HINT: your critic already has this functionality implemented
        if self.num_param_updates % self.target_update_freq == 0:
            self.critic.update_target_network()

        self.num_param_updates += 1

    self.t += 1
    return log
```

## `critics/dqn_critic.py`
```
def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    """
        Update the parameters of the critic.
        let sum_of_path_lengths be the sum of the lengths of the paths sampled from
            Agent.sample_trajectories
        let num_paths be the number of paths sampled from Agent.sample_trajectories
        arguments:
            ob_no: shape: (sum_of_path_lengths, ob_dim)
            ac_na: length: sum_of_path_lengths. The action taken at the current step.
            next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
            reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                the reward for each timestep
            terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                at that timestep of 0 if the episode did not end
        returns:
            nothing
    """
    ob_no = ptu.from_numpy(ob_no)
    ac_na = ptu.from_numpy(ac_na).to(torch.long)
    next_ob_no = ptu.from_numpy(next_ob_no)
    reward_n = ptu.from_numpy(reward_n)
    terminal_n = ptu.from_numpy(terminal_n)

    qa_t_values = self.q_net(ob_no)
    q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        
    # TODO compute the Q-values from the target network 
    qa_tp1_values = self.q_net_target(next_ob_no)


    if self.double_q:
        # You must fill this part for Q2 of the Q-learning portion of the homework.
        # In double Q-learning, the best action is selected using the Q-network that
        # is being updated, but the Q-value for this action is obtained from the
        # target Q-network. Please review Lecture 8 for more details,
        # and page 4 of https://arxiv.org/pdf/1509.06461.pdf is also a good reference.
        next_actions = self.q_net(next_ob_no).argmax(dim=1)
        q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)

    else:
        q_tp1, _ = qa_tp1_values.max(dim=1)

    # TODO compute targets for minimizing Bellman error
    # HINT: as you saw in lecture, this would be:
        #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
    target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
    target = target.detach()
    loss = self.loss(q_t_values, target)

    self.optimizer.zero_grad()
    loss.backward()
    utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
    self.optimizer.step()
    self.learning_rate_scheduler.step()
    return {
        'Training Loss': ptu.to_numpy(loss),
    }
```

## `policies/argmax_policy.py`
```
def get_action(self, obs):
    if len(obs.shape) > 3:
        observation = obs
    else:
        observation = obs[None]

    ## TODO return the action that maxinmizes the Q-value 
    # at the current observation as the output
    q_values = self.critic.qa_values(observation)
    action = q_values.argmax(-1)

    return action[0]
```

## `policies/MLP_policy.py`
```
class MLPPolicyAC(MLPPolicy):
    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        adv_n = ptu.from_numpy(adv_n)

        action_distribution = self(observations)
        loss = - action_distribution.log_prob(actions) * adv_n
        loss = loss.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

## `agents/ac_agent.py`
```
def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    # TODO Implement the following pseudocode:
    # for agent_params['num_critic_updates_per_agent_update'] steps,
    #     update the critic

    # advantage = estimate_advantage(...)

    # for agent_params['num_actor_updates_per_agent_update'] steps,
    #     update the actor

    for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
        critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

    advantages = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

    for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
        actor_loss = self.actor.update(ob_no, ac_na, adv_n=advantages)

    loss = OrderedDict()
    loss['Critic_Loss'] = critic_loss
    loss['Actor_Loss'] = actor_loss

    return loss

def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
    # TODO Implement the following pseudocode:
    # 1) query the critic with ob_no, to get V(s)
    # 2) query the critic with next_ob_no, to get V(s')
    # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
    # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
    # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
    v_n = self.critic.forward_np(ob_no)
    next_v_n = self.critic.forward_np(next_ob_no)

    assert v_n.shape == next_v_n.shape == re_n.shape == terminal_n.shape

    q_n = re_n + self.gamma * next_v_n * (1 - terminal_n)
    adv_n = q_n - v_n

    if self.standardize_advantages:
        adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
    return adv_n
```

## `critics/bootstrapped_continuous_critic.py`
```
def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    """
        Update the parameters of the critic.
        let sum_of_path_lengths be the sum of the lengths of the paths sampled from
            Agent.sample_trajectories
        let num_paths be the number of paths sampled from Agent.sample_trajectories
        arguments:
            ob_no: shape: (sum_of_path_lengths, ob_dim)
            ac_na: length: sum_of_path_lengths. The action taken at the current step.
            next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
            reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                the reward for each timestep
            terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                at that timestep of 0 if the episode did not end
        returns:
            training loss
    """
    # TODO: Implement the pseudocode below: do the following (
    # self.num_grad_steps_per_target_update * self.num_target_updates)
    # times:
    # every self.num_grad_steps_per_target_update steps (which includes the
    # first step), recompute the target values by
    #     a) calculating V(s') by querying the critic with next_ob_no
    #     b) and computing the target values as r(s, a) + gamma * V(s')
    # every time, update this critic using the observations and targets
    #
    # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
    #       to 0) when a terminal state is reached
    # HINT: make sure to squeeze the output of the critic_network to ensure
    #       that its dimensions match the reward

    ob_no = ptu.from_numpy(ob_no)
    ac_na = ptu.from_numpy(ac_na)
    next_ob_no = ptu.from_numpy(next_ob_no)
    reward_n = ptu.from_numpy(reward_n)
    terminal_n = ptu.from_numpy(terminal_n)

    for i in range(self.num_grad_steps_per_target_update):
        next_v = self(next_ob_no)
        target = reward_n + self.gamma * next_v * (1 - terminal_n)

        for j in range(self.num_target_updates):
            pred = self(ob_no)

            assert pred.shape == target.shape
            loss = self.loss(pred, target.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    return loss.item()
```