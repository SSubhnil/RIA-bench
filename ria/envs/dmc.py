from dm_control import suite
from dm_env import specs
import numpy as np
import collections
from types import SimpleNamespace

class DMC_Custom:
    def __init__(self, domain_name, task_name, seed=None):
        self._env = suite.load(domain_name=domain_name, task_name=task_name)
        self.proc_observation_space_dims = self._get_proc_observation_space_dims()
        self._random_state = np.random.RandomState()
        self._seed = seed
        self.dt = 0.025
        if self._seed is not None:
            self.seed(self._seed)

    def _get_proc_observation_space_dims(self):
      # Assuming the processing involves flattening the observations
      time_step = self._env.reset()
      obs = self._get_obs(time_step)
      proc_obs = self.obs_preproc(obs)
      return proc_obs.shape[0]

    def reset(self):
        time_step = self._env.reset()
        return self._get_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._get_obs(time_step)
        reward = time_step.reward if time_step.reward is not None else 0.0
        done = time_step.last()
        return obs, reward, done, {}

    def _get_obs(self, time_step):
        obs = np.concatenate([np.atleast_1d(v) for v in time_step.observation.values()])
        return obs

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reward(self, obs, act, next_obs):
        # Define your custom reward calculation here
        reward_ctrl = 0.0
        vel = (next_obs[..., -3] - obs[..., -3]) / self.dt
        reward_run = vel
        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive
        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            reward_ctrl = 0.0
            vel = (next_obs[..., -3] - obs[..., -3]) / self.dt
            reward_run = vel
            reward_contact = 0.0
            reward_survive = 0.05
            reward = reward_run + reward_ctrl + reward_contact + reward_survive
            return reward
        return _thunk

    def get_sim_parameters(self):
        # Assuming no additional simulation parameters are used
        return np.array([0.0])

    def num_modifiable_parameters(self):
        # Assuming no additional parameters can be modified
        return 0

    def log_diagnostics(self, paths, prefix):
        pass

    @property
    def observation_spec(self):
        return self._env.observation_spec()

    @property
    def action_spec(self):
        return self._env.action_spec()
    
    @property
    def observation_space(self):
        obs_spec = self._env.observation_spec()
        flattened_dim = sum(np.prod(spec.shape) for spec in obs_spec.values())
        return SimpleNamespace(shape=(int(flattened_dim),))
    
    @property
    def action_space(self):
        action_spec = self._env.action_spec()
        if isinstance(action_spec, collections.OrderedDict):
            flattened_dim = sum(np.prod(spec.shape) for spec in action_spec.values())
        else:
            flattened_dim = np.prod(action_spec.shape)
        
        # Addthe sample method to action_space
        return SimpleNamespace(shape=(int(flattened_dim),),
        sample=lambda: np.random.uniform(action_spec.minimum, action_spec.maximum))


    def render(self, mode='human'):
        # Render the environment if needed
        pass
    
    def seed(self, seed=None):
        self._seed = seed
        self._random_state.seed(seed)
    
    def get_labels(self):
        # Implement a method to return the labels.
        # As an example, if there are no specific labels, return an empty list or any required labels.
        return []

    def _sample_action(self):
        return np.random.uniform(self.actions_spec.minimum, self.action_spec.maximum)
    