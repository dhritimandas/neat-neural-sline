# play_neat.py - Run a trained NEAT model against SlimeVolley
import os
import pickle
import time
import argparse
import numpy as np
import gym
import slimevolleygym  # noqa: F401
import neat
from collections import deque 

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

def make_env():
    try:
        return gym.make("SlimeVolley-v0", render_mode="human")
    except TypeError:
        env = gym.make("SlimeVolley-v0")
        return env

def obs_to_np(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32)

def normalize_obs(obs):
    """Normalize observations to improve generalization"""
    # Copy to avoid modifying original
    norm_obs = np.copy(obs)
    
    # Normalize positions to [-1, 1] range
    norm_obs[0] = np.clip(norm_obs[0] * 1.2, -1, 1)  # Ball x
    norm_obs[4] = np.clip(norm_obs[4] * 1.2, -1, 1)  # Player x
    norm_obs[8] = np.clip(norm_obs[8] * 1.2, -1, 1)  # Opponent x
    
    # Normalize velocities
    for i in [2, 3, 6, 7, 10, 11]:  # All velocity components
        norm_obs[i] = np.tanh(norm_obs[i] * 0.5)  # Squash to [-1, 1]
        
    return norm_obs

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = terminated or truncated
        return obs, rew, done, info
    else:
        return out

def action_from_output(env, out_vec):
    out = np.asarray(out_vec, dtype=np.float32)
    # Always return 3-bit vector [left, right, jump] expected by slimevolleygym
    return (out[:3] > 0.0).astype(np.int8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/winner.pkl", help="Path to model file")
    parser.add_argument("--best", action="store_true", help="Use best_genome.pkl instead of winner.pkl")
    parser.add_argument("--jax", type=int, default=0, help="Use JAX for forward pass (1) or not (0)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    args = parser.parse_args()
    
    if args.best and "winner.pkl" in args.model:
        args.model = args.model.replace("winner.pkl", "best_genome.pkl")
        print(f"Using best genome: {args.model}")
    
    with open(args.model, "rb") as f:
        genome, config = pickle.load(f)
    
    if args.jax:
        try:
            import jax
            import jax.numpy as jnp
            
            # Create weight matrices and biases from genome
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            def jax_forward(inputs):
                # This is a simplified JAX version - you'd need to adapt this to match
                # the exact structure of your NEAT network for a real implementation
                x = jnp.array(inputs)
                
                # Mock implementation - in reality, you would extract the actual weights
                # from the NEAT network and use them properly
                for i, (key, conn) in enumerate(net.connections.items()):
                    if i == 0:  # Just for demonstration - extract real weights
                        x = jnp.tanh(x @ jnp.array(conn))
                
                return np.array(x)
            
            print("Using JAX for forward pass")
            forward_fn = jax.jit(jax_forward)
        except ImportError:
            print("JAX not available, using regular forward pass")
            forward_fn = lambda x: net.activate(x)
    else:
        # Standard NEAT forward pass
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        forward_fn = lambda x: net.activate(x)
    
    env = make_env()
    
    # Track overall stats
    total_raw_reward = 0
    total_shaped_reward = 0
    wins = 0
    losses = 0
    
    for ep in range(args.episodes):
        if args.seed is not None:
            seed = args.seed + ep
            try:
                obs = obs_to_np(env.reset(seed=seed))
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(seed)
                obs = obs_to_np(env.reset())
        else:
            obs = obs_to_np(env.reset())
        
        raw_ep_rew = 0.0  # Original game score
        shaped_ep_rew = 0.0  # Shaped reward
        steps = 0
        done = False
        
        # Track metrics for reward shaping
        ball_hits = 0
        successful_defenses = 0
        last_ball_y = obs[1]  # Ball Y position
        last_player_x = obs[4]  # Player X position
        last_ball_x = obs[0]  # Ball X position
        
        # Track position range for analysis
        max_x_position = obs[4]  # Initialize max player position
        min_x_position = obs[4]  # Initialize min player position
        
        # Add memory of recent actions to encourage coordination
        last_actions = [None, None, None]  # Remember last 3 actions
        
        # Track successful action sequences
        action_history = deque(maxlen=10)
        
        while not done:
            # Print debug info every 30 steps if debug mode is on
            if args.debug and steps > 0 and steps % 30 == 0:
                print(f"Step {steps}: ball_pos=({obs[0]:.2f}, {obs[1]:.2f}), player_pos={obs[4]:.2f}, action={action if steps > 0 else 'N/A'}")
            
            # Get action from neural network using normalized observations
            action = action_from_output(env, forward_fn(normalize_obs(obs)))
            
            # Store the history of actions
            if steps > 0:  # Only after first step
                last_actions.pop(0)  # Remove oldest action
                last_actions.append(action)  # Add newest action
                action_history.append(action)
            
            # Take step in environment
            next_obs, rew, done, info = step_env(env, action)
            next_obs = obs_to_np(next_obs)
            
            # Original game reward (scoring system)
            raw_ep_rew += float(rew)
            
            # BASE REWARD: Direct game score (+1 for winning point, -1 for losing point) - amplified
            shaped_reward = float(rew) * 10.0  # Amplify game score impact by 10x
            
            # REWARD SHAPING (matching train_neat.py):
            # 1. Reward for hitting the ball
            ball_y = next_obs[1]
            if last_ball_y > 0.5 and ball_y < 0.5 and abs(next_obs[0] - next_obs[4]) < 0.1:
                ball_hits += 1
                shaped_reward += 2.0  # Increased bonus for any ball contact
                
                # Bonus for successful action sequences
                if len(action_history) >= 5:
                    shaped_reward += 1.0  # Extra reward for building up to this hit
            
            # 2. Reward for positioning (staying close to ball x-position)
            ball_x = next_obs[0]
            player_x = next_obs[4]
            min_x_position = min(min_x_position, player_x)
            max_x_position = max(max_x_position, player_x)
            
            distance_to_ball = abs(ball_x - player_x)
            position_reward = 0.02 * (1.0 - min(1.0, distance_to_ball * 2))  # Reduced from 0.05
            shaped_reward += position_reward
            
            if abs(player_x - last_player_x) > 0.02:
                shaped_reward += 0.02  # Reduced from 0.1 - less emphasis on just movement
            else:
                shaped_reward -= 0.01  # Reduced penalty
            
            # 3. Penalize unnecessary jumping (when ball is far away)
            if action[2] > 0 and distance_to_ball > 0.3 and ball_y > 0.5:
                shaped_reward -= 0.02  # Small penalty for pointless jumps
            
            # Reward for coordinated action sequences
            if steps > 3 and all(a is not None for a in last_actions):
                # Check if agent has a pattern of appropriate actions based on ball position
                if ball_y < 0.3 and any(a[2] > 0 for a in last_actions):  # Jump when ball is low
                    shaped_reward += 0.2  # Reward for appropriate jump timing
                
                # Reward for moving consistently toward the ball
                if ball_x > player_x and all(a[1] > 0 for a in last_actions):  # Moving right toward ball
                    shaped_reward += 0.1
                elif ball_x < player_x and all(a[0] > 0 for a in last_actions):  # Moving left toward ball
                    shaped_reward += 0.1
            
            elif ball_x < -0.5:  # Ball is far on opponent side
                # Reward returning to center court position when ball is away
                center_reward = 0.05 * (1.0 - min(1.0, abs(player_x - 0.5)))
                shaped_reward += center_reward
            
            # Track successful defenses (when ball crosses back from opponent's side)
            if last_ball_x < 0 and ball_x > 0:
                successful_defenses += 1
                shaped_reward += 1.0  # Reward for successfully defending
            
            # 4. Reward for hitting ball to opponent's side
            if ball_hits > 0 and ball_x < 0 and last_ball_x > 0:
                shaped_reward += 3.0  # Major bonus for getting ball to opponent's side
            
            if ball_hits > 0 and ball_x < 0 and next_obs[3] > 0.2:  # Ball has positive y-velocity (downward)
                shaped_reward += 0.3  # Bonus for making the ball harder to return
            
            # 5. Reward for successful defensive position
            if ball_x < 0:  # Ball on opponent side
                defensive_reward = 0.05 * (1.0 - min(1.0, abs(player_x - 0.7)))  # Reward being near center-right
                shaped_reward += defensive_reward
            
            # 6. Reward for aggressive play when opponent is off-position
            opponent_x = next_obs[8]  # Opponent X position
            if abs(opponent_x) > 0.6 and ball_x > 0:  # Opponent out of position and ball on our side
                shaped_reward += 0.1  # Opportunity to score
            
            # 7. Reward for extended rallies (builds skill)
            if steps > 50:
                shaped_reward += 0.001 * steps  # Small cumulative bonus for longer play
            
            if ball_y < 0.8 and steps > 10:  # Ball is still in play after some steps
                shaped_reward += 0.01  # Small continuous reward for keeping rally going
            
            # 8. Combo reward for sequences of effective actions
            if ball_hits > 1:
                # Reward for multiple successful hits in same episode
                shaped_reward += 0.5 * ball_hits  # Progressive reward for continued success
            
            # Update for next step
            last_ball_y = ball_y
            last_player_x = player_x
            last_ball_x = ball_x
            obs = next_obs
            shaped_ep_rew += shaped_reward
            steps += 1
            
            # Render and maintain frame rate
            env.render()
            time.sleep(1.0 / args.fps)
        
        # End of episode stats
        total_raw_reward += raw_ep_rew
        total_shaped_reward += shaped_ep_rew
        
        # Track wins and losses
        if raw_ep_rew > 0:
            wins += 1
        elif raw_ep_rew < 0:
            losses += 1
        
        print(f"Episode {ep+1}: raw_reward={raw_ep_rew:.2f}, shaped_reward={shaped_ep_rew:.2f}, steps={steps}")
        print(f"Position range: min_x={min_x_position:.2f}, max_x={max_x_position:.2f}, range={max_x_position-min_x_position:.2f}")
        if max_x_position - min_x_position < 0.3:
            print("WARNING: Agent barely moved during episode!")
        
    print(f"\nOverall Performance:")
    print(f"Average Raw Reward: {total_raw_reward/args.episodes:.2f}")
    print(f"Average Shaped Reward: {total_shaped_reward/args.episodes:.2f}")
    print(f"Points Won: {wins}, Points Lost: {losses}, Net Score: {wins-losses}")
    
    env.close()

if __name__ == "__main__":
    main()