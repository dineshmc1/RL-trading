from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import Logger

class RLAgent:
    def __init__(self, env, model_path=None):
        """
        RecurrentPPO (LSTM) is suitable for financial time series because:
        1. It captures temporal dependencies and hidden states (market regimes).
        2. It's on-policy, ensuring more stable updates in noisy environments.
        """
        self.env = env
        if model_path:
            self.model = RecurrentPPO.load(model_path, env=env)
        else:
            self.model = RecurrentPPO(
                "MlpLstmPolicy", 
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
                policy_kwargs=dict(net_arch=[256, 256], enable_critic_lstm=False), # Common optimized config
                device="auto"
            )

    def train(self, total_timesteps=100000, callback=None):
        Logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        self.model.save(path)

    def predict(self, obs, state=None, episode_start=None):
        # RecurrentPPO requires state management for correct inference
        action, state = self.model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        return action, state
