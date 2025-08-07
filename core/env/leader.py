from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp
import jax.random as random

class PlanarMultiAgentEnv:
    def __init__(self, config: dict = None, state: Optional[jnp.ndarray] = None):
        if config is None:
            self.num_agents = 2
        else:
            self.num_agents = config["simulator"]["n_agents"]

        self.default_state = jnp.zeros((self.num_agents, 6))
        self.state = self.default_state if state is None else state
    def sample_random_action(self):
        self.key, subkey = random.split(self.key)
        # Random action trong [-1, 1]
        return random.uniform(
            subkey,
            shape=(self.num_agents, 3),
            minval=-1.0,
            maxval=1.0,
        )
    def reset(self):
        self.state = self.default_state
        return self.state

    @partial(jax.jit, static_argnums=0)
    def step(self, control: jnp.ndarray, dt: float = 0.05):
        """
        control: [u_x, u_y, u_theta] per agent
        state:   [x, x_dot, y, y_dot, theta, theta_dot] per agent
        """
        x, x_dot, y, y_dot, theta, theta_dot = (
            self.state[:, 0],
            self.state[:, 1],
            self.state[:, 2],
            self.state[:, 3],
            self.state[:, 4],
            self.state[:, 5],
        )

        u_x, u_y, u_theta = control[:, 0], control[:, 1], control[:, 2]

        new_x_dot = u_x
        new_y_dot = u_y
        new_theta_dot = u_theta

        new_x = x + new_x_dot * dt
        new_y = y + new_y_dot * dt
        new_theta = theta + new_theta_dot * dt

        new_state = jnp.stack(
            [new_x, new_x_dot, new_y, new_y_dot, new_theta, new_theta_dot], axis=-1
        )
        self.state = new_state
        return new_state

class PlanarMultiAgentEnvWrapper:
    def __init__(self, config):
        self.env = PlanarMultiAgentEnv(config)
        self.state_dim = 6 * config["simulator"]["n_agents"]
        self.action_dim = 3 * config["simulator"]["n_agents"]

    def reset(self):
        obs = self.env.reset().reshape(-1)
        obstacle = self.get_obstacle_encoding() 
        return jnp.concatenate([obs, obstacle])

    def step(self, action):
        action = action.reshape((self.env.num_agents, 3))
        next_state = self.env.step(action).reshape(-1)
        obstacle = self.get_obstacle_encoding()
        return jnp.concatenate([next_state, obstacle])

    def get_obstacle_encoding(self):
        # Tạm thời không có chướng ngại vật
        return jnp.zeros(6)
    
    def sample_random_action(self):
        return self.env.sample_random_action().reshape(-1)



