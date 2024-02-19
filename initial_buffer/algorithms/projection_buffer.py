from typing import Union

import torch
import numpy as np
from fast_pytorch_kmeans import KMeans as GPU_KMeans

from initial_buffer.utils.loss import SoftNearestNeighborLoss


class ProjectionBuffer:
    def __init__(
        self,
        device: str,
        obs_dim: int,
        advantage_gamma: float,
        gae_lambda: float,
        nr_clusters: int,
        sampling_strategy: str = 'network',
        cluster_algo: str = 'kmeans',
        lr_rate: float = 1e-4,
        min_timesteps: int = 16,
        n_train_data: int = 256,
        cluster_embedding_dim: int = 64,
        n_cluster_states: int = 256,
        nr_mining_samples: int = 10
    ):
        self.sampling_strategy = sampling_strategy

        self.advantage_gamma = advantage_gamma
        self.gae_lambda = gae_lambda
        self.obs_dim = obs_dim
        self.device = device
        self.lr_rate = lr_rate
        self.min_timesteps = min_timesteps
        self.n_train_data = n_train_data
        self.cluster_embedding_dim = cluster_embedding_dim
        self.n_cluster_states = n_cluster_states
        self.nr_mining_samples = nr_mining_samples
        self.nr_clusters = nr_clusters
        self.cluster_algo = cluster_algo

        self.train_data = None
        self.prev_improvement = None

        if self.sampling_strategy == 'network':
            self.setup_projection_network()

    def setup_projection_network(self) -> None:
        cluster_net_arch = [self.obs_dim, 64, 64]
        activation_fn = torch.nn.Tanh
        cluster_net = []
        last_layer_dim = cluster_net_arch[0]
        for layer_dim_out in cluster_net_arch[1:]:
            cluster_net.append(torch.nn.Linear(last_layer_dim, layer_dim_out))
            cluster_net.append(activation_fn())
            last_layer_dim = layer_dim_out
        cluster_net.append(torch.nn.Linear(last_layer_dim, self.cluster_embedding_dim))

        self.state_cluster_net = torch.nn.Sequential(*cluster_net).to(self.device)
        self.cluster_optimizer = torch.optim.Adam(self.state_cluster_net.parameters(), lr=self.lr_rate)
        self.cluster_contrastive_loss = SoftNearestNeighborLoss()

    def create_train_data(self,
                          rollout_rewards: np.ndarray,
                          rollout_observations: Union[torch.Tensor, np.ndarray],
                          rollout_episode_starts: np.ndarray,
                          ) -> None:
        """

        :param rollout_rewards: shape of [nr_timesteps, nr_envs]
        :param rollout_observations: shape of [nr_timesteps, nr_envs, observ_dim]
        :param rollout_states: shape of [nr_timesteps, nr_envs, state_dim]
        :param rollout_episode_starts: shape of [nr_timesteps, nr_envs]
        :return:
        """
        assert rollout_rewards.ndim == 2
        assert rollout_observations.ndim == 3
        assert rollout_episode_starts.ndim == 2
        assert self.sampling_strategy == 'network'

        nr_timesteps, nr_envs = rollout_rewards.shape
        observ_dim = rollout_observations.shape[-1]
        data_nr_timesteps = nr_timesteps - self.min_timesteps

        if type(rollout_observations) == np.ndarray:
            rollout_observations = torch.from_numpy(rollout_observations).to(self.device)

        # Reset previous improvement
        self.prev_improvement = None

        # Create arrays to store information
        cluster_rewards = np.zeros([self.n_train_data, data_nr_timesteps])
        cluster_observs = torch.zeros([self.n_train_data, data_nr_timesteps, observ_dim], device=self.device)
        cluster_mask = np.zeros([self.n_train_data, data_nr_timesteps], dtype=bool)

        # Fill arrays
        index_idx = 0
        search_states_ids = np.random.permutation((nr_timesteps - self.min_timesteps) * nr_envs)
        for i_trajectory in range(self.n_train_data):
            # Find a subsequence with a minimum length of at least min_timesteps
            for _ in range(50):
                random_idx = search_states_ids[index_idx]
                idx_timestep = random_idx // nr_envs
                idx_envs = random_idx % nr_envs
                steps_next_start = rollout_episode_starts[idx_timestep:, idx_envs].argmax()

                index_idx = (index_idx + 1) % search_states_ids.shape[0]
                if steps_next_start > self.min_timesteps:
                    break

            steps_next_start = min(steps_next_start, data_nr_timesteps)
            # If no environment restart, set next start to max possible index
            if steps_next_start == 0:
                steps_next_start = data_nr_timesteps - idx_timestep

            idx_trajectory_end = idx_timestep + steps_next_start
            cluster_rewards[i_trajectory, :steps_next_start] = rollout_rewards[idx_timestep:idx_trajectory_end, idx_envs]
            cluster_observs[i_trajectory, :steps_next_start, :] = rollout_observations[idx_timestep:idx_trajectory_end,
                                                                 idx_envs, :]
            cluster_mask[i_trajectory, :steps_next_start] = True

        # Crop the buffer
        max_timesteps = cluster_mask.sum(-1).max()
        self.train_data = {
                'rewards': cluster_rewards[:, :max_timesteps],
                'observs': cluster_observs[:, :max_timesteps, :],
                'traj_mask': cluster_mask[:, :max_timesteps],
        }

    def get_train_sample_observs(self) -> np.ndarray:
        assert self.train_data is not None, 'Train data for initial state buffer has not been created before'
        return self.train_data['observs'][self.train_data['traj_mask']]

    def train_step(self, train_data_value: np.ndarray) -> float:
        assert train_data_value.ndim == 1
        assert self.sampling_strategy == 'network'

        improvement_begin_to_step = self.policy_improvement(train_data_value)
        if self.prev_improvement is None:
            self.prev_improvement = np.zeros_like(improvement_begin_to_step)

        gradient_improvement = improvement_begin_to_step - self.prev_improvement
        self.prev_improvement = improvement_begin_to_step

        pos_ids = np.argpartition(gradient_improvement, self.nr_mining_samples)[-self.nr_mining_samples:]
        neg_ids = np.argpartition(-gradient_improvement, self.nr_mining_samples)[-self.nr_mining_samples:]

        projection_loss = self.cluster_train_step(self.train_data['observs'][pos_ids, 0, :],
                                                  self.train_data['observs'][neg_ids, 0, :])

        return projection_loss

    def cluster_train_step(self, pos_observs: Union[torch.Tensor, np.ndarray], neg_observs: Union[torch.Tensor, np.ndarray]):
        net_input = torch.cat([pos_observs, neg_observs], dim=0)

        embs = self.project_obs(net_input)
        pos_embs, neg_embs = torch.split(embs, [pos_observs.shape[0], neg_observs.shape[0]])

        loss = self.cluster_contrastive_loss(pos_embs[0, None, :], pos_embs[1:], neg_embs)

        self.cluster_optimizer.zero_grad()
        loss.backward()
        self.cluster_optimizer.step()

        return loss

    def policy_improvement(self, train_data_value: np.ndarray) -> np.ndarray:
        assert train_data_value.ndim == 1

        cluster_values = np.zeros_like(self.train_data['rewards'])
        cluster_values[self.train_data['traj_mask']] = train_data_value

        advantages = self.lambda_GAE_estimator(self.train_data['rewards'],
                                               cluster_values,
                                               np.logical_not(self.train_data['traj_mask']),
                                               self.train_data['traj_mask'],
                                               gamma=self.advantage_gamma,
                                               gae_lambda=self.gae_lambda)

        nr_timesteps = cluster_values.shape[1]
        gamma_t = self.advantage_gamma * np.ones(nr_timesteps)
        gamma_t[0] = 1
        gamma_t = np.cumprod(gamma_t)
        improvement = (advantages*gamma_t).sum(1)

        return improvement

    def create_initial_state_buffer(
                self,
                observations: Union[torch.Tensor, np.ndarray],
                buffer_length: int = 40,
                ) -> np.ndarray:
        """

        :param observations: shape of [nr_samples, overv_dim]
        :return:
        """
        if self.sampling_strategy == 'random':
            nr_samples = observations.shape[0]
            return np.random.permutation(nr_samples)[:buffer_length]

        assert observations.ndim == 2

        embs = self.project_obs(observations)
        cluster_distances, cluster_ids = self.cluster_embeddings(embs,
                                                                 nr_clusters=self.nr_clusters,
                                                                 cluster_algo=self.cluster_algo)
        k = buffer_length // self.nr_clusters
        k_smallest_distances_idx = np.argpartition(-cluster_distances, -k, axis=0)[-k:]
        k_smallest_distances = np.take_along_axis(cluster_distances, k_smallest_distances_idx, axis=0)
        sorted_idx = np.argsort(k_smallest_distances, axis=0)
        k_smallest_ordered_idx = np.take_along_axis(k_smallest_distances_idx, sorted_idx, axis=0)

        return k_smallest_ordered_idx.flatten()  # Important: ordering

    def cluster_embeddings(self, embs, nr_clusters, cluster_algo='kmeans'):
        if cluster_algo == 'kmeans':
            kmeans = GPU_KMeans(n_clusters=nr_clusters, mode='cosine', verbose=1)
            _ = kmeans.fit_predict(embs)
            cluster_distances = 1 - torch.matmul(embs, (torch.nn.functional.normalize(kmeans.centroids, p=2)).T).detach().cpu().numpy()
            cluster_id = np.argmin(cluster_distances, axis=1)

        else:
            raise ValueError('Specified clustering algorithm is not implemented')

        return cluster_distances, cluster_id

    def project_obs(self, observs):
        if self.sampling_strategy == 'network':
            embs = self.state_cluster_net(observs)
        elif self.sampling_strategy == 'observations':
            embs = observs
        else:
            raise ValueError("Specified observation projection is not implemented")

        embs = torch.nn.functional.normalize(embs, dim=-1, p=2)

        return embs

    def lambda_GAE_estimator(self,
                             rewards: np.ndarray,
                             values: np.ndarray,
                             episode_starts: np.ndarray,
                             traj_mask: np.ndarray,
                             gamma: float,
                             gae_lambda: float) -> np.ndarray:
        """Adapted from stable_baselines3/common/buffers.py """
        last_gae_lam = 0
        nr_cluster_states, buffer_size = rewards.shape[:2]
        advantages = np.zeros((nr_cluster_states, buffer_size), dtype=np.float32)

        for step in reversed(range(buffer_size - 1)):
            next_non_terminal = 1.0 - episode_starts[:, step + 1]
            next_values = values[:, step + 1]
            delta = rewards[:, step] + gamma * next_values * next_non_terminal - values[:, step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[:, step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA

        advantages = advantages * traj_mask

        return advantages
