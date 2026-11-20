# lunar-lander

A Deep Q-Learning (DQN) agent that learns to play Gymnasium's **LunarLander-v3**
using only PyTorch and Gymnasium — no pretrained weights. The program trains the
agent from scratch, saves the learned network to a PyTorch checkpoint, and can
then load that checkpoint to watch the agent fly.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
sync the project from the repository root:

```bash
uv sync
```

uv reads `pyproject.toml` and the committed `uv.lock`, creates a `.venv`, and
installs PyTorch, Gymnasium with Box2D, and NumPy. It will also download a
compatible Python interpreter (3.10–3.13) if needed; those versions have
prebuilt Box2D wheels. PyTorch is pinned to the small CPU-only builds in
`pyproject.toml` — see the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) if you
want to switch to CUDA builds. No manual virtualenv or pip commands are needed.

## Train

```bash
uv run main.py train                 # 1000 episodes, seed 42 (defaults)
uv run main.py train --episodes 2000 --log-every 25
```

Training prints the last score, the mean score over the last 100 episodes, the
current exploration rate, and the average loss. A checkpoint is written to
`models/dqn_lunar_lander.pt` whenever the 100-episode average reaches a new
best; that file therefore always contains the latest *saved* (best-so-far)
policy. Training stops early when the 100-episode average reaches 200, the
"solved" threshold for Lunar Lander.

Training can be continued from the checkpoint, and hyperparameters can be
changed from the command line:

```bash
uv run main.py train --resume --episodes 500
uv run main.py train --learning-rate 5e-4 --batch-size 128 --epsilon-decay 0.997
uv run main.py train --device cpu --model models/my_model.pt
```

Run `uv run main.py train --help` for the full list of options.

## Watch the trained agent

```bash
uv run main.py watch                 # 5 episodes with human rendering
uv run main.py watch --episodes 10 --model models/dqn_lunar_lander.pt
```

This greedily loads the saved network (no exploration, no training) and renders
the game. On a machine without a display, run it under a virtual framebuffer,
for example `xvfb-run -a uv run main.py watch`, or use
`--render-mode rgb_array`.

## Algorithm

Lunar Lander exposes 8 observation values and 4 discrete actions. The agent
uses the classic DQN recipe:

- A fully connected Q-network `8 -> 128 -> 128 -> 4` predicts the return for
  each action in the current state.
- Transitions `(s, a, r, s', done)` are stored in a 50,000-step replay buffer;
  after 1,000 warm-up steps, batches of 64 are sampled uniformly for training.
- The loss is the Huber (smooth L1) distance between `Q(s, a)` and the Bellman
  target `r + gamma * max_a' Q_target(s', a')` (`gamma = 0.99`).
- A separate target network is updated slowly toward the policy network
  (`tau = 0.005` per gradient step) to stabilize learning.
- Actions are chosen epsilon-greedily, with epsilon decaying from 1.0 to 0.05
  by a factor of 0.995 each episode. Optimization uses Adam (`lr = 1e-4`) with
  gradient clipping.

## Files

- `main.py` — command line interface for training and watching.
- `dqn.py` — Q-network, replay buffer, DQN agent, checkpoint save/load.
- `pyproject.toml` — project metadata and dependencies (managed by uv).
- `uv.lock` — locked dependency versions for reproducible installs.
- `models/` — created on training; holds the saved `.pt` checkpoint.

## Useful references

- Gym documentation: https://gymnasium.farama.org/
- Working with gym in PyTorch: https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/GYM.html
- Deep Q Learning paper: https://arxiv.org/abs/1312.5602
- HuggingFace on Reinforcement Learning: https://huggingface.co/learn/deep-rl-course/unit3/introduction
- Lunar Lander the game: https://gymnasium.farama.org/environments/box2d/lunar_lander/
