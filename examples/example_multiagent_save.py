from gym_battlesnake.gymbattlesnake import BattlesnakeEnv
from gym_battlesnake.custompolicy import CustomPolicy
from stable_baselines import PPO2

num_threads = 12
num_agents = 4
num_envs = 64
timesteps_per_generation=100000

# placeholder_env necessary for model to recognize,
# the observation and action space, and the vectorized environment
placeholder_env = BattlesnakeEnv(n_threads=num_threads, n_envs=num_envs)
models = [PPO2(CustomPolicy, placeholder_env, verbose=1, learning_rate=1e-3, tensorboard_log="./tensorboard") for _ in range(num_agents)]
# Close environment to free allocated resources
placeholder_env.close()

for generation in range(10):
    for (agent, model) in enumerate(models, start=1):
        log_name = "gen{}_agent{}".format(generation, agent)
        model_name = 'ppo2_agent{}'.format(agent)
        env = BattlesnakeEnv(n_threads=num_threads, n_envs=num_envs, opponents=[ m for m in models if m is not model])
        model.set_env(env)
        model.learn(total_timesteps=timesteps_per_generation, tb_log_name=log_name)
        model.save(model_name)
        env.close()
