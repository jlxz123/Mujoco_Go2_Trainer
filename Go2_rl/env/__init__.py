from gymnasium.envs.registration import register

register(
    id="Unitree-Go2",
    entry_point="env.go2:Go2Env",
)