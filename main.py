"""Train or watch a DQN agent playing Gymnasium's Lunar Lander.

Examples:
    python main.py train --episodes 1000
    python main.py watch --episodes 5
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dqn import DQN, DQNConfig

DEFAULT_ENV_ID = "LunarLander-v3"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "dqn_lunar_lander.pt"
DEFAULT_GIF_PATH = Path(__file__).resolve().parent / "assets" / "landing.gif"


def make_env(env_id: str, render_mode: str | None = None) -> gym.Env:
    """Create the environment, falling back to v2 on older Gymnasium installs."""
    try:
        return gym.make(env_id, render_mode=render_mode)
    except gym.error.Error:
        if env_id == DEFAULT_ENV_ID:
            print(f"{env_id} is unavailable; trying LunarLander-v2 instead.")
            return gym.make("LunarLander-v2", render_mode=render_mode)
        raise


def set_seed(seed: int) -> None:
    """Seed all random number generators used by training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_env_compat(agent: DQN, env: gym.Env) -> None:
    """Fail early if a saved model does not match the environment."""
    if not hasattr(env.action_space, "n"):
        raise ValueError("this DQN implementation requires a discrete action space")
    if env.observation_space.shape[0] != agent.n_observations:
        raise ValueError(
            f"model expects {agent.n_observations} observations, "
            f"environment has {env.observation_space.shape[0]}"
        )
    if env.action_space.n != agent.n_actions:
        raise ValueError(
            f"model expects {agent.n_actions} actions, "
            f"environment has {env.action_space.n}"
        )


def train(args: argparse.Namespace) -> None:
    """Train an agent from scratch (or from a checkpoint with --resume)."""
    set_seed(args.seed)

    agent: DQN | None = None
    checkpoint: dict | None = None
    env_id = args.env_id
    if args.resume:
        agent, checkpoint = DQN.load(args.model)
        env_id = checkpoint.get("env_id") or env_id
        print(
            f"Resuming from {args.model} "
            f"(episode {checkpoint.get('episode', '?')}, "
            f"average reward {checkpoint.get('mean_reward', '?')})."
        )

    env = make_env(env_id)
    try:
        if agent is not None:
            check_env_compat(agent, env)
        else:
            config = DQNConfig(
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                min_buffer_size=args.min_buffer_size,
                batch_size=args.batch_size,
                tau=args.tau,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay=args.epsilon_decay,
                hidden_size=args.hidden_size,
            )
            agent = DQN(
                n_observations=env.observation_space.shape[0],
                n_actions=env.action_space.n,
                config=config,
                device=args.device,
                seed=args.seed,
            )
    except ValueError:
        env.close()
        if args.resume:
            raise ValueError(
                "The saved model does not match the requested environment. "
                "Use the --env-id stored in the checkpoint or train from scratch."
            ) from None
        raise

    print(
        f"Training DQN on {env_id} for {args.episodes} episodes "
        f"(device: {agent.device}, seed: {args.seed})."
    )
    previous_episodes = (checkpoint or {}).get("episode") or 0
    previous_mean_reward = (checkpoint or {}).get("mean_reward")
    scores: deque[float] = deque(maxlen=100)
    best_mean_reward = (
        float(previous_mean_reward)
        if previous_mean_reward is not None
        else float("-inf")
    )
    saved_path: Path | None = None
    total_steps = 0

    try:
        for episode in range(1, args.episodes + 1):
            reset_kwargs = {"seed": args.seed} if episode == 1 else {}
            state, _ = env.reset(**reset_kwargs)
            epsilon = agent.epsilon_for_episode(previous_episodes + episode - 1)
            episode_reward = 0.0
            episode_losses: list[float] = []

            while True:
                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)

                agent.remember(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                episode_reward += reward
                total_steps += 1
                if done:
                    break

            scores.append(episode_reward)
            mean_reward = float(np.mean(scores))

            # Keep overwriting a single checkpoint: it always holds the best
            # policy seen so far, and is the one `watch` loads by default.
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                saved_path = agent.save(
                    args.model,
                    episode=episode,
                    mean_reward=mean_reward,
                    env_id=env_id,
                )

            if episode % args.log_every == 0 or episode == 1:
                mean_loss = (
                    float(np.mean(episode_losses)) if episode_losses else float("nan")
                )
                print(
                    f"episode {episode:5d}/{args.episodes} | "
                    f"score {episode_reward:8.1f} | "
                    f"avg(100) {mean_reward:8.1f} | "
                    f"epsilon {epsilon:.3f} | "
                    f"loss {mean_loss:.4f} | "
                    f"steps {total_steps}"
                )

            if len(scores) == scores.maxlen and mean_reward >= args.solved_score:
                print(
                    f"Solved! Average score over the last 100 episodes: "
                    f"{mean_reward:.1f} >= {args.solved_score:.1f}"
                )
                break
    finally:
        env.close()

    if saved_path is None:
        if args.resume:
            # Keep the better checkpoint we loaded instead of overwriting it.
            print(f"Training complete. No improvement; keeping {args.model}.")
            return
        saved_path = agent.save(args.model, episode=args.episodes, env_id=env_id)
    print(f"Training complete. Model saved to {saved_path}")


def watch(args: argparse.Namespace) -> None:
    """Load the latest saved model and watch it play without training."""
    agent, checkpoint = DQN.load(args.model)
    env_id = args.env_id or checkpoint.get("env_id") or DEFAULT_ENV_ID
    agent.policy_net.eval()

    env = make_env(env_id, render_mode=args.render_mode)
    try:
        check_env_compat(agent, env)
    except ValueError:
        env.close()
        raise

    print(
        f"Watching {env_id} with the model from {args.model} "
        f"(trained for {checkpoint.get('episode', '?')} episodes, "
        f"average reward {checkpoint.get('mean_reward', '?')})."
    )
    try:
        for episode in range(1, args.episodes + 1):
            seed = args.seed + episode - 1 if args.seed is not None else None
            state, _ = env.reset(seed=seed)
            episode_reward = 0.0

            while True:
                action = agent.select_action(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break

            print(f"episode {episode}: score {episode_reward:.1f}")
    finally:
        env.close()


def save_gif(frames: list, path: Path, duration: int, image_module) -> None:
    """Write rendered frames as an animated, palette-reduced GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    palette_frames = [
        frame.quantize(colors=128, method=image_module.Quantize.FASTOCTREE)
        for frame in frames
    ]
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )


def record(args: argparse.Namespace) -> None:
    """Record a GIF of the first greedy episode that reaches --min-score."""
    from PIL import Image  # only needed for recording

    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be at least 1")
    if args.fps < 1 or args.width < 1:
        raise ValueError("--fps and --width must be positive")

    # Rendering to an array does not need a display, but pygame still initializes
    # a video backend, so use a dummy one on headless machines.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    agent, checkpoint = DQN.load(args.model)
    env_id = args.env_id or checkpoint.get("env_id") or DEFAULT_ENV_ID
    agent.policy_net.eval()

    env = make_env(env_id, render_mode="rgb_array")
    try:
        check_env_compat(agent, env)
    except ValueError:
        env.close()
        raise

    def capture(frame) -> object:
        image = Image.fromarray(frame)
        if image.width != args.width:
            height = max(1, round(image.height * args.width / image.width))
            image = image.resize((args.width, height), Image.Resampling.LANCZOS)
        return image

    duration = max(1, round(1000 / args.fps))
    best_score = float("-inf")
    best_frames: list = []

    print(
        f"Recording {env_id} with the model from {args.model}; "
        f"stopping at the first episode scoring >= {args.min_score:.0f}."
    )
    try:
        for episode in range(1, args.max_episodes + 1):
            seed = args.seed + episode - 1 if args.seed is not None else None
            state, _ = env.reset(seed=seed)
            frames: list = []
            episode_reward = 0.0
            step = 0

            while True:
                if step % args.frame_skip == 0:
                    frame = env.render()
                    if frame is not None:
                        frames.append(capture(frame))
                action = agent.select_action(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                step += 1
                if terminated or truncated:
                    break

            frame = env.render()
            if frame is not None:
                frames.append(capture(frame))

            print(f"episode {episode}: score {episode_reward:.1f} ({len(frames)} frames)")
            if not frames:
                continue

            if episode_reward >= args.min_score:
                save_gif(frames, args.output, duration, Image)
                print(f"Saved successful landing GIF to {args.output}")
                return
            if episode_reward > best_score:
                best_score, best_frames = episode_reward, frames
    finally:
        env.close()

    if best_frames:
        save_gif(best_frames, args.output, duration, Image)
        print(
            f"No episode reached {args.min_score:.0f}; "
            f"saved the best one ({best_score:.1f}) to {args.output}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deep Q-Learning agent for Gymnasium Lunar Lander."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a DQN agent from scratch")
    train_parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    train_parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="where to save the model"
    )
    train_parser.add_argument("--episodes", type=int, default=1000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--solved-score",
        type=float,
        default=200.0,
        help="stop early when the 100-episode average reaches this score",
    )
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument(
        "--resume", action="store_true", help="continue from an existing checkpoint"
    )
    train_parser.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. cpu or cuda (default: auto-detect)",
    )

    defaults = DQNConfig()
    train_parser.add_argument("--gamma", type=float, default=defaults.gamma)
    train_parser.add_argument(
        "--learning-rate", type=float, default=defaults.learning_rate
    )
    train_parser.add_argument("--buffer-size", type=int, default=defaults.buffer_size)
    train_parser.add_argument(
        "--min-buffer-size", type=int, default=defaults.min_buffer_size
    )
    train_parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    train_parser.add_argument("--tau", type=float, default=defaults.tau)
    train_parser.add_argument(
        "--epsilon-start", type=float, default=defaults.epsilon_start
    )
    train_parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end)
    train_parser.add_argument(
        "--epsilon-decay", type=float, default=defaults.epsilon_decay
    )
    train_parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)

    watch_parser = subparsers.add_parser("watch", help="watch a saved agent play")
    watch_parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="saved model to load"
    )
    watch_parser.add_argument("--env-id", default=None, help="defaults to the saved env")
    watch_parser.add_argument("--episodes", type=int, default=5)
    watch_parser.add_argument("--seed", type=int, default=None)
    watch_parser.add_argument(
        "--render-mode",
        default="human",
        help="Gymnasium render mode (use rgb_array on headless machines)",
    )

    record_parser = subparsers.add_parser(
        "record", help="record a GIF of a successful greedy episode"
    )
    record_parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="saved model to load"
    )
    record_parser.add_argument(
        "--output", type=Path, default=DEFAULT_GIF_PATH, help="where to write the GIF"
    )
    record_parser.add_argument("--env-id", default=None, help="defaults to the saved env")
    record_parser.add_argument(
        "--min-score",
        type=float,
        default=200.0,
        help="only save an episode scoring at least this much",
    )
    record_parser.add_argument(
        "--max-episodes", type=int, default=20, help="maximum attempts to find one"
    )
    record_parser.add_argument(
        "--frame-skip", type=int, default=2, help="keep every Nth rendered frame"
    )
    record_parser.add_argument("--width", type=int, default=480, help="GIF frame width")
    record_parser.add_argument("--fps", type=int, default=20, help="GIF playback speed")
    record_parser.add_argument("--seed", type=int, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "watch":
        watch(args)
    else:
        record(args)


if __name__ == "__main__":
    main()
