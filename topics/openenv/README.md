# OpenEnv — Environments for RL Training

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is a framework from Meta for creating, deploying, and using execution environments for reinforcement learning. Think of it as **Gymnasium, but production-ready** — with a client-server architecture, container isolation, and remote deployment built in.

## Why OpenEnv?

If you've used Gymnasium (OpenAI Gym), you know the API: `reset()`, `step()`, `state()`. OpenEnv keeps that same simple interface but solves the problems you hit when moving from a notebook to production:

| Problem | Gymnasium | OpenEnv |
|---|---|---|
| **Isolation** | Runs in your process | Each env runs as its own server (even in a container) |
| **Remote access** | Local only | Client connects over WebSocket — env can run anywhere |
| **Scaling** | One env per process | Concurrent sessions, deploy to clusters |
| **Type safety** | Dict-based | Pydantic-typed actions, observations, and state |
| **Agent-agnostic** | Mostly classic RL | Works with DQN, PPO, LLMs (GRPO/RLHF), anything |

The key insight: **the environment shouldn't care what's training on it.** OpenEnv separates the environment (server) from the agent (client), so the same maze environment works with a tiny DQN neural net or a 135M-parameter LLM — no code changes.

## Core API

```python
# Connect to an environment server
client = MyEnvClient(base_url="http://localhost:8000")
client.connect()

# Gymnasium-style loop
result = client.reset()           # start a new episode
while not result.done:
    action = agent.act(result.observation)
    result = client.step(action)  # take an action, get next observation + reward

state = client.state()            # query episode metadata
client.close()
```

## Building an Environment

Three things to define:

**1. Models** — what actions look like, what the agent sees, and episode metadata:
```python
class MazeAction(Action):
    direction: str = "RIGHT"

class MazeObservation(Observation):
    grid: list[list[str]]
    agent_pos: tuple[int, int]
    exit_pos: tuple[int, int]

class MazeState(State):
    maze_seed: int
    optimal_path_length: int
```

**2. Environment** — the game logic:
```python
class MazeEnvironment(Environment[MazeAction, MazeObservation, MazeState]):
    def reset(self, seed=None, **kwargs) -> MazeObservation:
        # generate maze, place agent
        ...

    def step(self, action: MazeAction, **kwargs) -> MazeObservation:
        # move agent, compute reward, check if done
        ...

    @property
    def state(self) -> MazeState:
        return MazeState(maze_seed=self._seed, ...)
```

**3. Server** — one line to serve it:
```python
from openenv.core.env_server import create_app

app = create_app(MazeEnvironment, MazeAction, MazeObservation, env_name="maze")
# Run with: uvicorn app:app
```

That's it. Your environment is now accessible over HTTP/WebSocket.

## Demo: Maze RL

See [`maze-rl/`](maze-rl/) for a complete example:

- **Same maze environment** trained with two different approaches
- **DQN** (2-layer MLP) — learns to solve the maze in minutes
- **LLM GRPO** (SmolLM2-135M) — same env, different agent
- Both run as **Flyte pipelines** with visual reports and interactive maze replays

```bash
cd maze-rl
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# DQN — fast learner
flyte run --local maze_rl_dqn.py pipeline --training_steps 20

# LLM — same environment, different agent
flyte run --local maze_rl_llm.py pipeline --training_steps 10
```

## Links

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [Documentation](https://meta-pytorch.org/OpenEnv/)
- [Example environments](https://github.com/meta-pytorch/OpenEnv/tree/main/examples) — Chess, Connect4, Snake, Atari, Coding, and more
