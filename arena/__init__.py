from gymnasium.envs.registration import register
from arena.envs.AICRL_arena import ArenaEnv

register(
    id="arena",
    entry_point="arena.envs.AICRL_arena:ArenaEnv",
)
