import torch
import numpy as np

from initial_buffer.algorithms.projection_buffer import ProjectionBuffer


def evaluate_value_funtion(obs):
    return np.random.random([obs.shape[0]])


def main():
    obs_dim = 18
    state_dim = 52
    ppo_gamma = 0.99
    gae_lambda = 0.95
    nr_timesteps = 250
    nr_envs = 8
    sampling_strategy = 'network'  # ['network', 'observations', 'random']
    device = 'cuda:0'

    projection_buffer = ProjectionBuffer(
    device=device,
    nr_clusters=64,
    cluster_algo='kmeans',
    obs_dim=obs_dim,
    advantage_gamma=ppo_gamma,
    gae_lambda=gae_lambda,
    sampling_strategy=sampling_strategy,
    min_timesteps=8,
    )
    visited_state_buffer_obs = None
    for i_epoch in range(10):
        # ========================== Roll Out Phase ==========================
        # Select states to initialize robot in environment based on the states and observations stored in the visited state buffer
        if visited_state_buffer_obs is not None:
            selected_idx = projection_buffer.create_initial_state_buffer(
                                                            torch.from_numpy(visited_state_buffer_obs.reshape([nr_timesteps*nr_envs, obs_dim])).to(device),
                                                            buffer_length=256,
                                                            )
            initialization_states = visited_state_buffer_states.reshape([nr_timesteps*nr_envs, state_dim])[selected_idx]

        # Collect rollout data
        # Add experiences to visited state buffer / Here, the prefiltering to exclude failing states can be added
        visited_state_buffer_obs = np.random.random([nr_timesteps, nr_envs, obs_dim]).astype(dtype=np.float32)
        visited_state_buffer_rewards = np.random.random([nr_timesteps, nr_envs])
        visited_state_buffer_dones = np.random.random([nr_timesteps, nr_envs])
        visited_state_buffer_states = np.random.random([nr_timesteps, nr_envs, state_dim])


        # ======================== Policy Update Phase ========================
        projection_buffer.create_train_data(
                        rollout_rewards=visited_state_buffer_rewards,
                        rollout_observations=visited_state_buffer_obs,
                        rollout_episode_starts=visited_state_buffer_dones,
        )

        mean_cluster_loss = 0
        for i_policy_update in range(10):
            ## For each gradient update
            obs = projection_buffer.get_train_sample_observs()
            # The current value of the observations needs to be sampled at this points, e.g, actor_critic.evaluate(obs).squeeze(-1)
            # The following function is just a placeholder
            values = evaluate_value_funtion(obs)
            mean_cluster_loss += projection_buffer.train_step(values)


if __name__ == '__main__':
    main()
