# HW4 solutions
These solutions only show the relevant functions and the ones that were specific to this assignment.

## `agents/mb_agent.py`
```
def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    # training a MB agent refers to updating the predictive model using observed state transitions
    # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
    losses = []
    num_data = ob_no.shape[0]
    num_data_per_ens = int(num_data / self.ensemble_size)

    for i in range(self.ensemble_size):

        # select which datapoints to use for this model of the ensemble
        start_index = i * num_data_per_ens
        end_index = (i + 1) * num_data_per_ens

        # get datapoints (s,a,s') for model training
        observations = ob_no[start_index:end_index]
        actions = ac_na[start_index:end_index]
        next_observations = next_ob_no[start_index:end_index]

        # use datapoints to update the model
        model = self.dyn_models[i]
        log = model.update(observations, actions, next_observations,
                            self.data_statistics)
        loss = log['Training Loss']
        losses.append(loss)

    avg_loss = np.mean(losses)
    return {
        'Training Loss': avg_loss,
    }
```

## `policies/MPC_policy.py`
```
def sample_action_sequences(self, num_sequences, horizon, obs=None):
    if self.sample_strategy == 'random' \
        or (self.sample_strategy == 'cem' and obs is None):
        return self.get_random_actions(num_sequences, horizon)
    elif self.sample_strategy == 'cem':
        actions = self.get_random_actions(num_sequences, horizon)
        elite_mean, elite_std = np.zeros(actions.shape[1:]), np.zeros(actions.shape[1:])
        for i in range(self.cem_iterations):
            if i > 0:
                actions = np.random.normal(elite_mean, elite_std, size=(num_sequences, *elite_mean.shape))
            rewards = self.evaluate_candidate_sequences(actions, obs)
            sorted_idxs = sorted(range(len(actions)), key=lambda i: rewards[i])
            elites = actions[sorted_idxs][-self.cem_num_elites:]
            if i == 0:
                elite_mean, elite_std = np.mean(elites, axis=0), np.std(elites, axis=0)
            else:
                elite_mean = self.cem_alpha * np.mean(elites, axis=0) + (1 - self.cem_alpha) * elite_mean
                elite_std = self.cem_alpha * np.std(elites, axis=0) + (1 - self.cem_alpha) * elite_std
        return elite_mean[None] # Only a single candidate action sequence for CEM
    else:
        raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")
```

```
def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
    # For each candidate action sequence, predict a sequence of
    # states for each dynamics model in your ensemble.
    # Once you have a sequence of predicted states from each model in
    # your ensemble, calculate the sum of rewards for each sequence
    # using `self.env.get_reward(predicted_obs)`
    # You should sum across `self.horizon` time step.
    # Hint: this should be an array of shape [N]
    # Hint: you should use model.get_prediction and you shouldn't need
    #       to import pytorch in this file.
    observations_per_timestep = []
    sum_of_rewards = np.zeros((self.N,))

    # N copies of obs
    # (each copy will undergo its own sequence of horizon actions)
    obs_pred = np.tile(obs, (self.N, 1))
    observations_per_timestep.append(obs_pred)

    # pass sampled candidate action sequences through model & get reward predictions
    for t in range(self.horizon):

        # select N actions to try at this timestep
        actions = candidate_action_sequences[:, t, :]  # [N, ac]

        # calculate predicted reward of current timestep
        r, _ = self.env.get_reward(obs_pred, actions)  # [N,]
        sum_of_rewards += r

        # predict result of executing the actions
        next_obs_prediction = model.get_prediction(
            obs_pred, actions, self.data_statistics)

        # bookkeeping
        obs_pred = next_obs_prediction
        observations_per_timestep.append(obs_pred)
    return sum_of_rewards
```

```
def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
    # for each model in ensemble:
    predicted_sum_of_rewards_per_model = []
    for model in self.dyn_models:
        sum_of_rewards = self.calculate_sum_of_rewards(
            obs, candidate_action_sequences, model)
        predicted_sum_of_rewards_per_model.append(sum_of_rewards)

    # calculate mean_across_ensembles(predicted rewards)
    return np.mean(
        predicted_sum_of_rewards_per_model, axis=0)  # [ens,N] --> N
```

## ``models/ff_model.py`
```
def forward(
            self,
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
    # normalize input data to mean 0, std 1
    obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std)
    acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std)

    # predicted change in obs
    concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)
    delta_pred_normalized = self.delta_network(concatenated_input)
    delta_pred_unnormalized = unnormalize(
        delta_pred_normalized, delta_mean, delta_std)
    next_obs_pred = obs_unnormalized + delta_pred_unnormalized
    return next_obs_pred, delta_pred_normalized

def get_prediction(self, obs, acs, data_statistics):
    obs = ptu.from_numpy(obs)
    acs = ptu.from_numpy(acs)
    new_data_statistics = {
        k: ptu.from_numpy(v) for k, v in data_statistics.items()
    }
    prediction, _ = self(obs, acs, **new_data_statistics)
    return ptu.to_numpy(prediction)

def update(self, observations, actions, next_observations, data_statistics):
    target = normalize(
        next_observations - observations,
        data_statistics['delta_mean'],
        data_statistics['delta_std'],
    )

    obs = ptu.from_numpy(observations)
    acs = ptu.from_numpy(actions)
    torch_data_statistics = {
        k: ptu.from_numpy(v) for k, v in data_statistics.items()
    }
    _, delta_pred_normalized = self(obs, acs, **torch_data_statistics)

    target = ptu.from_numpy(target)
    loss = self.loss(delta_pred_normalized, target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {
        'Training Loss': ptu.to_numpy(loss),
    }
```