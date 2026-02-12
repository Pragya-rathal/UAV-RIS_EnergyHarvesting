try:
    from gymnasium.envs.registration import register
except ImportError:  # pragma: no cover - fallback for gym-only installs
    from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_foo.envs:FooEnv',
)
