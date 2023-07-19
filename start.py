import gymnasium as gym
import mobile_env
from stable_baselines3 import A2C, PPO


env = gym.make(
    "mobile-verylarge-ma-v0", render_mode="rgb_array", disable_env_checker=False
)

model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=None,
    stats_window_size=100,
    tensorboard_log="./runs/",
    policy_kwargs=None,
    verbose=0,
    seed=None,
    device="auto",
    _init_setup_model=True,
)
model.learn(total_timesteps=25000)
env.close()
