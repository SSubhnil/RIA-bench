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
        
        default_params = {
            'cripple_part': None, # 'right_hip/_knee/_ankle' or 'left_hip/_knee/_ankle'
            'force_type': 'step', # can be None, swelling, step
            'timing': 'random',
            'body_part': 'torso',
            'random_chance': 0.8,  # Chance to apply random force
            'force_range': (90, 170),
            'interval_mean': 90,  # Mean for sampling interval 90, 180
            'interval_std': 10,  # Standard deviation for sampling interval
            'duration_min': 5,  # Minimum duration for swelling force
            'duration_max': 20  # Maximum duration for the swelling force
            }
        self.confounder_params = default_params

        # Initialize attributes based on confounder_params
        self.cripple_part = self.confounder_params['cripple_part']
        self.force_type = self.confounder_params['force_type']
        self.timing = self.confounder_params['timing']
        self.body_part = self.confounder_params['body_part']
        self.random_chance = self.confounder_params['random_chance']
        self.force_range = self.confounder_params['force_range']
        self.interval_mean = self.confounder_params['interval_mean']
        self.interval_std = self.confounder_params['interval_std']
        self.duration_min = self.confounder_params['duration_min']
        self.duration_max = self.confounder_params['duration_max']
        self.time_since_last_force = 0

        # Applying action masking for crippling of the legs
        self.action_mask = self._action_mask(self.cripple_part)

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
        if self.force_type is not None:
            self.apply_force()
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

    def apply_force(self):
        if self.timing == 'random':
            self.interval = max(30, int(np.random.normal(self.interval_mean,
                                                         self.interval_std)))
            if np.random.uniform() > self.random_chance:
                return

        # Update the timing
        self.time_since_last_force += 1
        if self.time_since_last_force < self.interval:
            return

        # Reset timing for next force application
        self.time_since_last_force = 0

        # Sample the force magnitude fom a normal distribution within the range
        force_magnitude = np.clip(np.random.normal((self.force_range[0] + self.force_range[1]) / 2,
                                                   (self.force_range[1] - self.force_range[0]) / 6),
                                  self.force_range[0], self.force_range[1])

        # Calculate the duration for the force application if 'swelling'
        duration = np.random.randint(self.duration_min, self.duration_max + 1)

        # FLipping the direction for additional challenge
        direction = np.random.choice([-1, 1])

        # Apply swelling or other dynamics based on force type
        # Construct the force vector
        if self.force_type == 'step':
            force = np.array([direction * force_magnitude, 0, 0, 0, 0, 0])
        elif self.force_type == 'swelling':
            # Calculate the time step where the force magnitude is at its peak
            peak_time = duration / 2
            # Calculate the standard deviation to control thh width of the bell curve
            sigma = duration / 6  # Adjust as needed for the desired width
            # Calculate the force magnitude at the current time step using a Gaussian function
            time_step_normalized = (self.time_since_last_force - peak_time) / sigma
            magnitude = force_magnitude * np.exp(-0.5 * (time_step_normalized ** 2))
            force = np.array([direction * magnitude, 0, 0, 0, 0, 0])

        body_id = self._env.physics.model.name2id(self.body_part, 'body')
        # Apply the force
        self._env.physics.data.xfrc_applied[body_id] = force

    def _action_mask(self, name):
        mask_vec = None
        if name == 'right_hip':
            mask_vec = [0, 1, 1, 1, 1, 1]
        elif name == 'right_knee':
            mask_vec = [1, 0, 1, 1, 1, 1]
        elif name == 'right_ankle':
            mask_vec = [1, 1, 0, 1, 1, 1]
        elif name == 'left_hip':
            mask_vec = [1, 1, 1, 0, 1, 1]
        elif name == 'left_knee':
            mask_vec = [1, 1, 1, 1, 0, 1]
        elif name == 'left_ankle':
            mask_vec = [1, 1, 1, 1, 1, 0]
        return mask_vec
    