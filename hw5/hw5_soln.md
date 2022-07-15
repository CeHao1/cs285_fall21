# HW5 solutions
These solutions only show the relevant functions and the ones that were specific to this assignment.

## `agents/awac_agent.py`
```
def estimate_advantage(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, n_actions=10):
    # Calculate the advantage (n sample estimate) 
    ob_no = ptu.from_numpy(ob_no)
    ac_na = ptu.from_numpy(ac_na)
    re_n = ptu.from_numpy(re_n)
    next_ob_no = ptu.from_numpy(next_ob_no)
    terminal_n = ptu.from_numpy(terminal_n)

    vals = []
    dist = self.awac_actor(ob_no)
    if self.agent_params['discrete']:
        for i in range(self.agent_params['ac_dim']):
            ac_pi = ptu.ones((ob_no.shape[0],)) * i
            v1_pi = torch.exp(dist.log_prob(ac_pi)) * self.get_qvals(self.exploitation_critic, ob_no, ac_pi)
            vals.append(v1_pi)
    else:
        for _ in range(n_actions):
            ac_pi = dist.sample()
            v1_pi = self.get_qvals(self.exploitation_critic, ob_no, ac_pi)
            vals.append(v1_pi)
    v_pi = torch.cat(vals, 1).mean(dim=1)

    return self.get_qvals(self.exploitation_critic, ob_no, ac_na) - v_pi

def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    log = {}

    if self.t > self.num_exploration_steps:
        # <DONE>: After exploration is over, set the actor to optimize the extrinsic critic
        #HINT: Look at method ArgMaxPolicy.set_critic
        self.actor.set_critic(self.exploitation_critic)
        self.actor.use_boltzmann = False

    if (self.t > self.learning_starts
            and self.t % self.learning_freq == 0
            and self.replay_buffer.can_sample(self.batch_size)
    ):

        # Get Reward Weights
        # <DONE>: Get the current explore reward weight and exploit reward weight
        #       using the schedule's passed in (see __init__)
        # COMMENT: Until part 3, explore_weight = 1, and exploit_weight = 0
        explore_weight = self.explore_weight_schedule.value(self.t)
        exploit_weight = self.exploit_weight_schedule.value(self.t)

        # Run Exploration Model #
        # <DONE>: Evaluate the exploration model on s' to get the exploration bonus
        # HINT: Normalize the exploration bonus, as RND values vary highly in magnitudeelse:
        expl_bonus = self.exploration_model.forward_np(ob_no)
        if self.normalize_rnd:
            expl_bonus = normalize(expl_bonus, 0, self.running_rnd_rew_std)
            self.running_rnd_rew_std = (self.rnd_gamma * self.running_rnd_rew_std 
                + (1 - self.rnd_gamma) * expl_bonus.std())

        # Reward Calculations #
        # <DONE>: Calculate mixed rewards, which will be passed into the exploration critic
        # HINT: See doc for definition of mixed_reward
        mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n

        # <DONE>: Calculate the environment reward
        # HINT: For part 1, env_reward is just 're_n'
        #       After this, env_reward is 're_n' shifted by self.exploit_rew_shift,
        #       and scaled by self.exploit_rew_scale
        env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale

        # Update Critics And Exploration Model #

        # <DONE> 1): Update the exploration model (based off s')
        # <DONE> 2): Update the exploration critic (based off mixed_reward)
        # <DONE> 3): Update the exploitation critic (based off env_reward)
        expl_model_loss = self.exploration_model.update(next_ob_no)
        exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
        exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)


        #update actor
        # <DONE> 1): Estimate the advantage
        # <DONE> 2): Calculate the awac actor loss
        advantage = self.estimate_advantage(ob_no, ac_na, re_n, next_ob_no, terminal_n, self.agent_params['n_actions'])
        actor_loss = self.awac_actor.update(ob_no, ac_na, advantage)

        # Target Networks #
        if self.num_param_updates % self.target_update_freq == 0:
            # <DONE>: Update the exploitation and exploration target networks
            self.exploitation_critic.update_target_network()
            self.exploration_critic.update_target_network()

        # Logging #
        log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
        log['Exploitation Critic Loss'] = exploitation_critic_loss['Training Loss']
        log['Exploration Model Loss'] = expl_model_loss

        # <DONE>: Uncomment these lines after completing awac
        log['Actor Loss'] = actor_loss

        self.num_param_updates += 1

    self.t += 1
    return log
```

## `agents/explore_or_exploit_agent.py`
```
def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    log = {}

    if self.t > self.num_exploration_steps:
        # <DONE>: After exploration is over, set the actor to optimize the extrinsic critic
        #HINT: Look at method ArgMaxPolicy.set_critic
        self.actor.set_critic(self.exploitation_critic)
        self.actor.use_boltzmann = False

    if (self.t > self.learning_starts
            and self.t % self.learning_freq == 0
            and self.replay_buffer.can_sample(self.batch_size)
    ):

        # Get Reward Weights
        # <DONE>: Get the current explore reward weight and exploit reward weight
        #       using the schedule's passed in (see __init__)
        # COMMENT: Until part 3, explore_weight = 1, and exploit_weight = 0
        explore_weight = self.explore_weight_schedule.value(self.t)
        exploit_weight = self.exploit_weight_schedule.value(self.t)

        # Run Exploration Model #
        # <DONE>: Evaluate the exploration model on s' to get the exploration bonus
        # HINT: Normalize the exploration bonus, as RND values vary highly in magnitudeelse:
        expl_bonus = self.exploration_model.forward_np(ob_no)
        if self.normalize_rnd:
            expl_bonus = normalize(expl_bonus, 0, self.running_rnd_rew_std)
            self.running_rnd_rew_std = (self.rnd_gamma * self.running_rnd_rew_std 
                + (1 - self.rnd_gamma) * expl_bonus.std())

        # Reward Calculations #
        # <DONE>: Calculate mixed rewards, which will be passed into the exploration critic
        # HINT: See doc for definition of mixed_reward
        mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n

        # <DONE>: Calculate the environment reward
        # HINT: For part 1, env_reward is just 're_n'
        #       After this, env_reward is 're_n' shifted by self.exploit_rew_shift,
        #       and scaled by self.exploit_rew_scale
        env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale

        # Update Critics And Exploration Model #

        # <DONE> 1): Update the exploration model (based off s')
        # <DONE> 2): Update the exploration critic (based off mixed_reward)
        # <DONE> 3): Update the exploitation critic (based off env_reward)
        expl_model_loss = self.exploration_model.update(next_ob_no)
        exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
        exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)

        # Target Networks #
        if self.num_param_updates % self.target_update_freq == 0:
            # <DONE>: Update the exploitation and exploration target networks
            self.exploitation_critic.update_target_network()
            self.exploration_critic.update_target_network()

        # Logging #
        log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
        log['Exploitation Critic Loss'] = exploitation_critic_loss['Training Loss']
        log['Exploration Model Loss'] = expl_model_loss

        # <DONE>: Uncomment these lines after completing cql_critic.py
        log['Exploitation Data q-values'] = exploitation_critic_loss['Data q-values']
        log['Exploitation OOD q-values'] = exploitation_critic_loss['OOD q-values']
        log['Exploitation CQL Loss'] = exploitation_critic_loss['CQL Loss']

        self.num_param_updates += 1

    self.t += 1
    return log
```

## `critics/cql_critic.py`
```
def dqn_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    qa_t_values = self.q_net(ob_no)
    q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
    qa_tp1_values = self.q_net_target(next_ob_no)

    next_actions = self.q_net(next_ob_no).argmax(dim=1)
    q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)

    target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
    target = target.detach()
    loss = self.loss(q_t_values, target)

    return loss, qa_t_values, q_t_values


def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    """
        Update the parameters of the critic.
        let sum_of_path_lengths be the sum of the lengths of the paths sampled from
            Agent.sample_trajectories
        let num_paths be the number of paths sampled from Agent.sample_trajectories
        arguments:
            ob_no: shape: (sum_of_path_lengths, ob_dim)
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

    loss, qa_t_values, q_t_values = self.dqn_loss(
        ob_no, ac_na, next_ob_no, reward_n, terminal_n
        )
    
    # CQL Implementation
    # <DONE>: Implement CQL as described in the pdf and paper
    # Hint: After calculating cql_loss, augment the loss appropriately
    q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)
    cql_loss = torch.mean(q_t_logsumexp - q_t_values)
    loss = self.cql_alpha * cql_loss + loss

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    info = {'Training Loss': ptu.to_numpy(loss)}

    # <DONE>: Uncomment these lines after implementing CQL
    info['CQL Loss'] = ptu.to_numpy(cql_loss)
    info['Data q-values'] = ptu.to_numpy(q_t_values).mean()
    info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()

    return info
```

## `policies/MLP_policy.py`
```
def update(self, observations, actions, adv_n=None):
    if adv_n is None:
        assert False
    if isinstance(observations, np.ndarray):
        observations = ptu.from_numpy(observations)
    if isinstance(actions, np.ndarray):
        actions = ptu.from_numpy(actions)
    if isinstance(adv_n, np.ndarray):
        adv_n = ptu.from_numpy(adv_n)

    dist = self(observations)
    log_prob_n = dist.log_prob(actions)
    actor_loss = -log_prob_n * torch.exp(adv_n/self.lambda_awac)
    actor_loss = actor_loss.mean()
    
    self.optimizer.zero_grad()
    actor_loss.backward()
    self.optimizer.step()
    
    return actor_loss.item()
```