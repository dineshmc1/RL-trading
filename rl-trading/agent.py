from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import Logger

class RLAgent:
    def __init__(self, env, model_path=None):
        """
        PPO is suitable for financial time series because:
        1. It's on-policy, ensuring more stable updates in noisy environments.
        2. It uses a clipped objective to prevent large, destabilizing policy updates.
        3. Efficient and easy to tune compared to DQN for continuous or high-dimensional states.
        """
        self.env = env
        if model_path:
            self.model = PPO.load(model_path, env=env)
        else:
            self.model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
                device="auto"
            )

    def train(self, total_timesteps=100000, callback=None):
        Logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        self.model.save(path)

    def predict(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action
