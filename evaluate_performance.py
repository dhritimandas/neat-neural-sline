#!/usr/bin/env python3
"""
Performance Evaluation and Training Optimization Script for NEAT SlimeVolley Agent
Tracks training progress and identifies specific areas for improvement
"""

import os
import sys
import pickle
import argparse
import numpy as np
import gym
import neat
import time
from pathlib import Path
from collections import defaultdict, deque
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

try:
    import slimevolleygym
except ImportError:
    print("Error: slimevolleygym not installed. Run: pip install slimevolleygym")
    sys.exit(1)

def make_env():
    try:
        return gym.make("SlimeVolley-v0", render_mode=None)
    except TypeError:
        return gym.make("SlimeVolley-v0")

def obs_to_np(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32)

def normalize_obs(obs):
    """Normalize observations to improve generalization"""
    norm_obs = np.copy(obs)
    norm_obs[0] = np.clip(norm_obs[0] * 1.2, -1, 1)  # Ball x
    norm_obs[4] = np.clip(norm_obs[4] * 1.2, -1, 1)  # Player x
    norm_obs[8] = np.clip(norm_obs[8] * 1.2, -1, 1)  # Opponent x
    
    for i in [2, 3, 6, 7, 10, 11]:  # All velocity components
        norm_obs[i] = np.tanh(norm_obs[i] * 0.5)
        
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
    return (out[:3] > 0.0).astype(np.int8)

class TrainingAnalyzer:
    """Analyzes training checkpoints to identify improvement areas"""
    
    def __init__(self, checkpoint_dir="models/robust"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints = self._load_checkpoints()
        
    def _load_checkpoints(self):
        """Load all available checkpoints"""
        checkpoints = []
        for ckpt_file in sorted(self.checkpoint_dir.glob("chkpt-*")):
            try:
                with open(ckpt_file, 'rb') as f:
                    data = pickle.load(f)
                    generation = int(ckpt_file.name.split('-')[1])
                    checkpoints.append((generation, data))
            except:
                pass
        return checkpoints
    
    def analyze_fitness_progression(self):
        """Analyze how fitness evolved over generations"""
        if not self.checkpoints:
            return None
            
        progression = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'fitness_std': [],
            'population_diversity': [],
            'stagnation_periods': []
        }
        
        last_best = -float('inf')
        stagnation_count = 0
        
        for gen, (pop, species, generation_actual, config, *_) in self.checkpoints:
            fitnesses = [g.fitness for g in pop.values() if g.fitness is not None]
            if not fitnesses:
                continue
                
            best_fit = max(fitnesses)
            avg_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            
            progression['generations'].append(gen)
            progression['best_fitness'].append(best_fit)
            progression['avg_fitness'].append(avg_fit)
            progression['fitness_std'].append(std_fit)
            
            # Track stagnation
            if best_fit <= last_best * 1.01:  # Less than 1% improvement
                stagnation_count += 1
            else:
                if stagnation_count > 3:
                    progression['stagnation_periods'].append((gen - stagnation_count, gen))
                stagnation_count = 0
                last_best = best_fit
            
            # Calculate diversity (number of species)
            progression['population_diversity'].append(len(species.species))
        
        return progression
    
    def identify_training_issues(self, progression):
        """Identify specific issues in training"""
        issues = []
        recommendations = []
        
        if not progression:
            return issues, recommendations
            
        # Check for early convergence
        if len(progression['generations']) > 10:
            late_improvement = progression['best_fitness'][-1] - progression['best_fitness'][-10]
            if late_improvement < progression['best_fitness'][-10] * 0.1:
                issues.append("EARLY CONVERGENCE: Training plateaued in last 10 generations")
                recommendations.append({
                    'issue': 'Early Convergence',
                    'fix': 'Increase mutation rates in neat_config_slime.cfg',
                    'params': {
                        'weight_mutate_rate': '0.95 (from 0.9)',
                        'conn_add_prob': '0.8 (from 0.7)',
                        'node_add_prob': '0.6 (from 0.5)'
                    }
                })
        
        # Check for low diversity
        if progression['population_diversity'][-1] < 3:
            issues.append("LOW DIVERSITY: Only {} species remaining".format(
                progression['population_diversity'][-1]))
            recommendations.append({
                'issue': 'Low Population Diversity',
                'fix': 'Adjust species parameters',
                'params': {
                    'compatibility_threshold': '4.0 (from 3.0)',
                    'species_elitism': '3 (from 2)'
                }
            })
        
        # Check for stagnation periods
        if progression['stagnation_periods']:
            issues.append("STAGNATION: {} periods of no improvement".format(
                len(progression['stagnation_periods'])))
            recommendations.append({
                'issue': 'Training Stagnation',
                'fix': 'Implement dynamic difficulty adjustment',
                'code_change': 'Add curriculum learning with progressive difficulty in train_neat.py'
            })
        
        # Check fitness variance
        if progression['fitness_std'][-1] < 10:
            issues.append("LOW FITNESS VARIANCE: Population too homogeneous")
            recommendations.append({
                'issue': 'Homogeneous Population',
                'fix': 'Increase population size and mutation diversity',
                'params': {
                    'pop_size': '1000 (from 500)',
                    'activation_mutate_rate': '0.3 (from 0.2)'
                }
            })
        
        return issues, recommendations

class PerformanceEvaluator:
    def __init__(self, model_path):
        """Initialize evaluator with a trained model"""
        with open(model_path, "rb") as f:
            self.genome, self.config = pickle.load(f)
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self.model_path = model_path
        
    def evaluate_comprehensive(self, episodes=100, verbose=True):
        """
        Comprehensive evaluation against internal AI with detailed skill metrics
        """
        metrics = {
            'total_episodes': episodes,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'points_scored': 0,
            'points_conceded': 0,
            
            # Core performance metrics
            'win_rate': 0.0,
            'point_differential': 0,
            'avg_episode_length': [],
            
            # Skill-specific metrics
            'ball_control': {
                'hits_per_game': [],
                'hit_accuracy': [],  # Successful hits / attempted hits
                'avg_hit_distance': [],  # Distance to ball when hitting
                'reaction_time': []  # Steps between ball coming to our side and hit
            },
            
            'positioning': {
                'court_coverage': [],  # Movement range
                'center_tendency': [],  # Time spent near center
                'defensive_positioning': [],  # Quality of defensive stance
                'offensive_positioning': []  # Positioning for attacks
            },
            
            'tactical': {
                'rally_wins': [],  # Rallies won vs lost
                'comeback_ability': 0,  # Games won from behind
                'pressure_handling': [],  # Performance when behind
                'finishing_ability': []  # Converting match points
            },
            
            'movement': {
                'movement_efficiency': [],  # Purposeful vs random movement
                'jump_accuracy': [],  # Successful jumps / total jumps
                'speed_to_ball': [],  # Average time to reach ball
                'wasted_actions': []  # Unnecessary jumps/movements
            },
            
            # Game flow metrics
            'dominating_wins': 0,
            'close_losses': 0,
            'shutouts': 0,
            'got_shutout': 0,
            'longest_rally': 0,
            'shortest_win': float('inf'),
            'momentum_shifts': 0
        }
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE PERFORMANCE EVALUATION")
        print(f"Model: {os.path.basename(self.model_path)}")
        print(f"Episodes: {episodes}")
        print(f"{'='*60}\n")
        
        for ep in range(episodes):
            ep_metrics = self.evaluate_single_episode_detailed(seed=ep)
            
            # Update win/loss records
            if ep_metrics['final_score']['agent'] > ep_metrics['final_score']['opponent']:
                metrics['wins'] += 1
                if ep_metrics['final_score']['agent'] - ep_metrics['final_score']['opponent'] >= 3:
                    metrics['dominating_wins'] += 1
                if ep_metrics['was_comeback']:
                    metrics['tactical']['comeback_ability'] += 1
                if ep_metrics['final_score']['opponent'] == 0:
                    metrics['shutouts'] += 1
                if ep_metrics['total_steps'] < metrics['shortest_win']:
                    metrics['shortest_win'] = ep_metrics['total_steps']
            elif ep_metrics['final_score']['agent'] < ep_metrics['final_score']['opponent']:
                metrics['losses'] += 1
                if ep_metrics['final_score']['opponent'] - ep_metrics['final_score']['agent'] == 1:
                    metrics['close_losses'] += 1
                if ep_metrics['final_score']['agent'] == 0:
                    metrics['got_shutout'] += 1
            else:
                metrics['draws'] += 1
            
            # Update detailed metrics
            metrics['points_scored'] += ep_metrics['final_score']['agent']
            metrics['points_conceded'] += ep_metrics['final_score']['opponent']
            metrics['avg_episode_length'].append(ep_metrics['total_steps'])
            
            # Ball control metrics
            metrics['ball_control']['hits_per_game'].append(ep_metrics['ball_hits'])
            metrics['ball_control']['hit_accuracy'].append(ep_metrics['hit_accuracy'])
            metrics['ball_control']['avg_hit_distance'].append(ep_metrics['avg_hit_distance'])
            metrics['ball_control']['reaction_time'].append(ep_metrics['avg_reaction_time'])
            
            # Positioning metrics
            metrics['positioning']['court_coverage'].append(ep_metrics['movement_range'])
            metrics['positioning']['center_tendency'].append(ep_metrics['center_time_ratio'])
            metrics['positioning']['defensive_positioning'].append(ep_metrics['defensive_quality'])
            metrics['positioning']['offensive_positioning'].append(ep_metrics['offensive_quality'])
            
            # Movement metrics
            metrics['movement']['movement_efficiency'].append(ep_metrics['movement_efficiency'])
            metrics['movement']['jump_accuracy'].append(ep_metrics['jump_accuracy'])
            metrics['movement']['speed_to_ball'].append(ep_metrics['avg_speed_to_ball'])
            metrics['movement']['wasted_actions'].append(ep_metrics['wasted_actions'])
            
            # Tactical metrics
            metrics['tactical']['rally_wins'].append(ep_metrics['rallies_won_ratio'])
            metrics['tactical']['pressure_handling'].append(ep_metrics['pressure_performance'])
            metrics['tactical']['finishing_ability'].append(ep_metrics['finishing_rate'])
            
            # Track longest rally
            if ep_metrics['longest_rally'] > metrics['longest_rally']:
                metrics['longest_rally'] = ep_metrics['longest_rally']
            
            metrics['momentum_shifts'] += ep_metrics['momentum_shifts']
            
            if verbose and (ep + 1) % 10 == 0:
                win_rate = metrics['wins'] / (ep + 1) * 100
                print(f"Progress: {ep+1}/{episodes} | Win Rate: {win_rate:.1f}% | "
                      f"W/L: {metrics['wins']}/{metrics['losses']}")
        
        # Calculate final statistics
        metrics['win_rate'] = (metrics['wins'] / episodes) * 100
        metrics['point_differential'] = metrics['points_scored'] - metrics['points_conceded']
        
        # Average all collected metrics
        for category in ['ball_control', 'positioning', 'movement', 'tactical']:
            for key in metrics[category]:
                if isinstance(metrics[category][key], list) and metrics[category][key]:
                    metrics[category][key] = np.mean(metrics[category][key])
        
        metrics['avg_episode_length'] = np.mean(metrics['avg_episode_length']) if metrics['avg_episode_length'] else 0
        
        return metrics
    
    def evaluate_single_episode_detailed(self, seed=None, render=False):
        """Evaluate a single episode with detailed skill tracking"""
        env = make_env()
        
        if seed is not None:
            try:
                obs = obs_to_np(env.reset(seed=seed))
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(seed)
                obs = obs_to_np(env.reset())
        else:
            obs = obs_to_np(env.reset())
        
        metrics = {
            'final_score': {'agent': 0, 'opponent': 0},
            'ball_hits': 0,
            'attempted_hits': 0,
            'successful_defenses': 0,
            'movement_range': 0,
            'total_steps': 0,
            'was_comeback': False,
            'longest_rally': 0,
            'rallies_won_ratio': 0,
            'momentum_shifts': 0,
            
            # Detailed skill metrics
            'hit_accuracy': 0,
            'avg_hit_distance': 0,
            'avg_reaction_time': 0,
            'center_time_ratio': 0,
            'defensive_quality': 0,
            'offensive_quality': 0,
            'movement_efficiency': 0,
            'jump_accuracy': 0,
            'avg_speed_to_ball': 0,
            'wasted_actions': 0,
            'pressure_performance': 0,
            'finishing_rate': 0
        }
        
        done = False
        steps = 0
        
        # Tracking variables
        current_rally_length = 0
        rally_outcomes = []  # True for won rallies, False for lost
        
        min_x = obs[4]
        max_x = obs[4]
        center_time = 0
        
        agent_score = 0
        opponent_score = 0
        max_deficit = 0
        lead_changes = 0
        last_leader = None
        
        # Ball tracking
        last_ball_x = obs[0]
        last_ball_y = obs[1]
        last_player_x = obs[4]
        
        # Action tracking
        total_jumps = 0
        successful_jumps = 0
        unnecessary_jumps = 0
        total_movements = 0
        purposeful_movements = 0
        
        # Hit tracking
        hit_distances = []
        reaction_times = []
        ball_on_our_side_timer = 0
        
        # Position quality tracking
        defensive_positions = []
        offensive_positions = []
        
        while not done and steps < 3000:
            # Get action from neural network
            action = action_from_output(env, self.net.activate(normalize_obs(obs)))
            
            # Take step
            next_obs, rew, done, info = step_env(env, action)
            next_obs = obs_to_np(next_obs)
            
            # Track scores and rallies
            if rew > 0:
                agent_score += 1
                rally_outcomes.append(True)
                current_rally_length = 0
            elif rew < 0:
                opponent_score += 1
                rally_outcomes.append(False)
                current_rally_length = 0
                deficit = opponent_score - agent_score
                max_deficit = max(max_deficit, deficit)
            else:
                current_rally_length += 1
                if current_rally_length > metrics['longest_rally']:
                    metrics['longest_rally'] = current_rally_length
            
            # Track momentum shifts
            current_leader = 'agent' if agent_score > opponent_score else ('opponent' if opponent_score > agent_score else 'tie')
            if last_leader and current_leader != last_leader and current_leader != 'tie':
                lead_changes += 1
            last_leader = current_leader
            
            # Track ball hits and accuracy
            if last_ball_y > 0.5 and next_obs[1] < 0.5 and abs(next_obs[0] - next_obs[4]) < 0.15:
                metrics['ball_hits'] += 1
                hit_distances.append(abs(next_obs[0] - next_obs[4]))
                if ball_on_our_side_timer > 0:
                    reaction_times.append(ball_on_our_side_timer)
                    ball_on_our_side_timer = 0
                successful_jumps += 1 if action[2] > 0 else 0
            
            # Track attempted hits (near misses)
            if last_ball_y > 0.5 and next_obs[1] < 0.5 and abs(next_obs[0] - next_obs[4]) < 0.3:
                metrics['attempted_hits'] += 1
            
            # Track reaction time
            if next_obs[0] > 0 and last_ball_x <= 0:  # Ball just entered our side
                ball_on_our_side_timer = 1
            elif next_obs[0] > 0:
                ball_on_our_side_timer += 1
            
            # Track successful defenses
            if last_ball_x < 0 and next_obs[0] > 0:
                metrics['successful_defenses'] += 1
            
            # Track position quality
            if next_obs[0] < 0:  # Ball on opponent side - defensive positioning
                dist_to_center = abs(next_obs[4] - 0.7)
                defensive_positions.append(1.0 - min(1.0, dist_to_center))
            else:  # Ball on our side - offensive positioning
                dist_to_ball = abs(next_obs[0] - next_obs[4])
                offensive_positions.append(1.0 - min(1.0, dist_to_ball * 2))
            
            # Track movement efficiency
            if action[0] > 0 or action[1] > 0:
                total_movements += 1
                # Movement is purposeful if it reduces distance to ball
                if (next_obs[0] > next_obs[4] and action[1] > 0) or (next_obs[0] < next_obs[4] and action[0] > 0):
                    purposeful_movements += 1
            
            # Track jump accuracy
            if action[2] > 0:
                total_jumps += 1
                # Jump is unnecessary if ball is far
                if abs(next_obs[0] - next_obs[4]) > 0.3 or next_obs[1] > 0.7:
                    unnecessary_jumps += 1
            
            # Track center tendency
            if abs(next_obs[4] - 0.5) < 0.3:
                center_time += 1
            
            # Update position range
            min_x = min(min_x, next_obs[4])
            max_x = max(max_x, next_obs[4])
            
            # Update for next iteration
            last_ball_x = next_obs[0]
            last_ball_y = next_obs[1]
            last_player_x = next_obs[4]
            obs = next_obs
            steps += 1
            
            if render:
                env.render()
                time.sleep(1/30)
        
        # Calculate final metrics
        metrics['final_score']['agent'] = agent_score
        metrics['final_score']['opponent'] = opponent_score
        metrics['movement_range'] = max_x - min_x
        metrics['total_steps'] = steps
        metrics['was_comeback'] = (max_deficit >= 2 and agent_score > opponent_score)
        metrics['momentum_shifts'] = lead_changes
        
        # Calculate skill metrics
        metrics['hit_accuracy'] = metrics['ball_hits'] / max(1, metrics['attempted_hits'])
        metrics['avg_hit_distance'] = np.mean(hit_distances) if hit_distances else 0.5
        metrics['avg_reaction_time'] = np.mean(reaction_times) if reaction_times else 10
        metrics['center_time_ratio'] = center_time / max(1, steps)
        metrics['defensive_quality'] = np.mean(defensive_positions) if defensive_positions else 0
        metrics['offensive_quality'] = np.mean(offensive_positions) if offensive_positions else 0
        metrics['movement_efficiency'] = purposeful_movements / max(1, total_movements)
        metrics['jump_accuracy'] = 1.0 - (unnecessary_jumps / max(1, total_jumps))
        metrics['avg_speed_to_ball'] = metrics['avg_reaction_time']  # Simplified
        metrics['wasted_actions'] = unnecessary_jumps
        metrics['rallies_won_ratio'] = sum(rally_outcomes) / max(1, len(rally_outcomes))
        
        # Pressure performance (performance when behind)
        if opponent_score > agent_score:
            metrics['pressure_performance'] = metrics['rallies_won_ratio']
        else:
            metrics['pressure_performance'] = 1.0
        
        # Finishing ability (converting when ahead)
        if agent_score >= 4:  # Had match point
            metrics['finishing_rate'] = 1.0 if agent_score > opponent_score else 0.0
        else:
            metrics['finishing_rate'] = 0.5  # Neutral if no match point
        
        env.close()
        return metrics

def generate_training_optimization_report(metrics, training_analyzer=None):
    """Generate detailed optimization report with specific improvements"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING OPTIMIZATION REPORT")
    print(f"{'='*60}")
    
    # Overall Performance Grade
    performance_grade = calculate_performance_grade(metrics)
    print(f"\n🎯 OVERALL GRADE: {performance_grade['grade']} ({performance_grade['score']:.1f}/100)")
    print(f"   Status: {performance_grade['status']}")
    
    # Skill Breakdown
    print(f"\n📊 SKILL BREAKDOWN:")
    skills = analyze_skill_levels(metrics)
    for skill_name, skill_data in skills.items():
        print(f"\n   {skill_name}:")
        print(f"      Score: {skill_data['score']:.1f}/10")
        print(f"      Level: {skill_data['level']}")
        if skill_data['issues']:
            print(f"      Issues: {', '.join(skill_data['issues'])}")
    
    # Critical Weaknesses
    print(f"\n⚠️  CRITICAL WEAKNESSES:")
    weaknesses = identify_critical_weaknesses(metrics, skills)
    for i, weakness in enumerate(weaknesses, 1):
        print(f"\n   {i}. {weakness['name']}")
        print(f"      Impact: {weakness['impact']}")
        print(f"      Current: {weakness['current']}")
        print(f"      Target: {weakness['target']}")
    
    # Specific Training Improvements
    print(f"\n🔧 REQUIRED TRAINING IMPROVEMENTS:")
    improvements = generate_training_improvements(metrics, weaknesses)
    for i, improvement in enumerate(improvements, 1):
        print(f"\n   {i}. {improvement['title']}")
        print(f"      Priority: {improvement['priority']}")
        print(f"      File: {improvement['file']}")
        print(f"      Line: {improvement['line']}")
        print(f"      Current Code: {improvement['current']}")
        print(f"      New Code: {improvement['new']}")
        print(f"      Expected Impact: {improvement['impact']}")
    
    # Training Progress Analysis
    if training_analyzer:
        print(f"\n📈 TRAINING PROGRESS ANALYSIS:")
        progression = training_analyzer.analyze_fitness_progression()
        if progression:
            issues, recommendations = training_analyzer.identify_training_issues(progression)
            
            if issues:
                print(f"\n   Issues Found:")
                for issue in issues:
                    print(f"      • {issue}")
            
            if recommendations:
                print(f"\n   Recommendations:")
                for rec in recommendations:
                    print(f"      • {rec['issue']}: {rec['fix']}")
                    if 'params' in rec:
                        for param, value in rec['params'].items():
                            print(f"          - {param}: {value}")
    
    # Generate visualization
    if training_analyzer:
        generate_training_plots(metrics, training_analyzer, "training_analysis.png")
        print(f"\n📊 Visualization saved to: training_analysis.png")
    
    return performance_grade, weaknesses, improvements

def calculate_performance_grade(metrics):
    """Calculate overall performance grade"""
    score = 0
    max_score = 100
    
    # Win rate (40 points max)
    score += min(40, metrics['win_rate'] * 0.4)
    
    # Ball control (20 points max)
    ball_control_score = (
        metrics['ball_control']['hit_accuracy'] * 5 +
        min(5, metrics['ball_control']['hits_per_game']) +
        max(0, 10 - metrics['ball_control']['reaction_time'])
    )
    score += min(20, ball_control_score)
    
    # Positioning (15 points max)
    positioning_score = (
        metrics['positioning'].get('defensive_quality', 0.5) * 7.5 +
        metrics['positioning'].get('offensive_quality', 0.5) * 7.5
    )
    score += min(15, positioning_score)
    
    # Movement (15 points max)
    movement_score = (
        metrics['movement']['movement_efficiency'] * 10 +
        metrics['movement']['jump_accuracy'] * 5
    )
    score += min(15, movement_score)
    
    # Tactical (10 points max)
    tactical_score = (
        metrics['tactical']['rally_wins'] * 5 +
        metrics['tactical']['pressure_handling'] * 3 +
        metrics['tactical']['finishing_ability'] * 2
    )
    score += min(10, tactical_score)
    
    # Determine grade
    if score >= 90:
        grade = "A+ (EXPERT)"
        status = "Agent has mastered the game!"
    elif score >= 80:
        grade = "A (ADVANCED)"
        status = "Agent is highly skilled"
    elif score >= 70:
        grade = "B (PROFICIENT)"
        status = "Agent shows good understanding"
    elif score >= 60:
        grade = "C (COMPETENT)"
        status = "Agent has basic skills"
    elif score >= 50:
        grade = "D (DEVELOPING)"
        status = "Agent needs improvement"
    else:
        grade = "F (BEGINNER)"
        status = "Agent requires significant training"
    
    return {'score': score, 'grade': grade, 'status': status}

def analyze_skill_levels(metrics):
    """Analyze individual skill levels"""
    skills = {}
    
    # Ball Control
    ball_score = (
        metrics['ball_control']['hit_accuracy'] * 3 +
        min(3, metrics['ball_control']['hits_per_game']) +
        max(0, 4 - metrics['ball_control']['reaction_time'] / 2.5)
    )
    skills['Ball Control'] = {
        'score': min(10, ball_score),
        'level': get_skill_level(ball_score),
        'issues': []
    }
    if metrics['ball_control']['hit_accuracy'] < 0.5:
        skills['Ball Control']['issues'].append('Low hit accuracy')
    if metrics['ball_control']['hits_per_game'] < 1:
        skills['Ball Control']['issues'].append('Few ball contacts')
    
    # Positioning
    pos_score = (
        metrics['positioning'].get('defensive_quality', 0.5) * 5 +
        metrics['positioning'].get('offensive_quality', 0.5) * 3 +
        metrics['positioning'].get('court_coverage', 0.5) * 2
    )
    skills['Positioning'] = {
        'score': min(10, pos_score),
        'level': get_skill_level(pos_score),
        'issues': []
    }
    if metrics['positioning'].get('court_coverage', 0.5) < 0.5:
        skills['Positioning']['issues'].append('Limited court coverage')
    if metrics['positioning'].get('defensive_quality', 0.5) < 0.5:
        skills['Positioning']['issues'].append('Poor defensive positioning')
    
    # Movement
    mov_score = (
        metrics['movement']['movement_efficiency'] * 5 +
        metrics['movement']['jump_accuracy'] * 3 +
        max(0, 2 - metrics['movement']['wasted_actions'] / 5)
    )
    skills['Movement'] = {
        'score': min(10, mov_score),
        'level': get_skill_level(mov_score),
        'issues': []
    }
    if metrics['movement']['movement_efficiency'] < 0.6:
        skills['Movement']['issues'].append('Inefficient movement')
    if metrics['movement']['jump_accuracy'] < 0.7:
        skills['Movement']['issues'].append('Poor jump timing')
    
    # Tactical Play
    tac_score = (
        metrics['tactical']['rally_wins'] * 4 +
        metrics['tactical']['pressure_handling'] * 3 +
        metrics['tactical']['finishing_ability'] * 3
    )
    skills['Tactical Play'] = {
        'score': min(10, tac_score),
        'level': get_skill_level(tac_score),
        'issues': []
    }
    if metrics['tactical']['rally_wins'] < 0.4:
        skills['Tactical Play']['issues'].append('Loses most rallies')
    if metrics['tactical']['comeback_ability'] == 0:
        skills['Tactical Play']['issues'].append('No comeback ability')
    
    return skills

def get_skill_level(score):
    """Convert skill score to level"""
    if score >= 9:
        return "MASTER"
    elif score >= 7:
        return "ADVANCED"
    elif score >= 5:
        return "INTERMEDIATE"
    elif score >= 3:
        return "BASIC"
    else:
        return "NOVICE"

def identify_critical_weaknesses(metrics, skills):
    """Identify the most critical weaknesses to address"""
    weaknesses = []
    
    # Check win rate
    if metrics['win_rate'] < 50:
        weaknesses.append({
            'name': 'Low Win Rate',
            'impact': 'CRITICAL - Cannot beat AI consistently',
            'current': f"{metrics['win_rate']:.1f}%",
            'target': '75%+',
            'priority': 1
        })
    
    # Check ball control
    if metrics['ball_control']['hits_per_game'] < 1.5:
        weaknesses.append({
            'name': 'Poor Ball Contact',
            'impact': 'HIGH - Missing opportunities to return ball',
            'current': f"{metrics['ball_control']['hits_per_game']:.1f} hits/game",
            'target': '3+ hits/game',
            'priority': 2
        })
    
    # Check movement
    if metrics['movement']['movement_efficiency'] < 0.5:
        weaknesses.append({
            'name': 'Inefficient Movement',
            'impact': 'MEDIUM - Wasting energy on random movements',
            'current': f"{metrics['movement']['movement_efficiency']*100:.0f}% efficient",
            'target': '75%+ efficient',
            'priority': 3
        })
    
    # Check positioning
    if metrics['positioning'].get('court_coverage', 0.5) < 0.4:
        weaknesses.append({
            'name': 'Static Positioning',
            'impact': 'HIGH - Not covering enough court area',
            'current': f"{metrics['positioning'].get('court_coverage', 0.5):.2f} range",
            'target': '0.6+ range',
            'priority': 2
        })
    
    # Check reaction time
    if metrics['ball_control']['reaction_time'] > 5:
        weaknesses.append({
            'name': 'Slow Reactions',
            'impact': 'MEDIUM - Late to respond to ball',
            'current': f"{metrics['ball_control']['reaction_time']:.1f} steps",
            'target': '<3 steps',
            'priority': 3
        })
    
    return sorted(weaknesses, key=lambda x: x['priority'])

def generate_training_improvements(metrics, weaknesses):
    """Generate specific code improvements for training"""
    improvements = []
    
    for weakness in weaknesses[:3]:  # Focus on top 3 weaknesses
        if weakness['name'] == 'Low Win Rate':
            improvements.append({
                'title': 'Increase Win Bonus and Game Score Impact',
                'priority': 'CRITICAL',
                'file': 'train_neat.py',
                'line': '227',
                'current': 'shaped_reward = float(rew) * 10.0',
                'new': 'shaped_reward = float(rew) * 25.0  # Increased from 10x to 25x',
                'impact': 'Stronger incentive to win points'
            })
            improvements.append({
                'title': 'Add Progressive Opponent Difficulty',
                'priority': 'HIGH',
                'file': 'train_neat.py',
                'line': '126',
                'current': 'curriculum_level = min(1.0, 0.5 + 0.5 * (generation / 50.0))',
                'new': 'curriculum_level = min(1.0, 0.3 + 0.7 * (generation / 100.0))',
                'impact': 'Slower difficulty progression for better learning'
            })
        
        elif weakness['name'] == 'Poor Ball Contact':
            improvements.append({
                'title': 'Increase Ball Hit Reward',
                'priority': 'CRITICAL',
                'file': 'train_neat.py',
                'line': '235',
                'current': 'shaped_reward += 10.0',
                'new': 'shaped_reward += 25.0  # Increased from 10 to 25',
                'impact': 'Much stronger reward for hitting ball'
            })
            improvements.append({
                'title': 'Add Proximity Bonus for Near Misses',
                'priority': 'HIGH',
                'file': 'train_neat.py',
                'line': '245',
                'current': '# 2. Reward for positioning',
                'new': '''# Add near-miss bonus
                if last_ball_y > 0.5 and ball_y < 0.5 and abs(ball_x - player_x) < 0.25:
                    shaped_reward += 5.0  # Reward for being close even if missed''',
                'impact': 'Encourages getting closer to ball'
            })
        
        elif weakness['name'] == 'Inefficient Movement':
            improvements.append({
                'title': 'Penalize Random Movement More',
                'priority': 'MEDIUM',
                'file': 'train_neat.py',
                'line': '260',
                'current': 'shaped_reward -= 0.05',
                'new': 'shaped_reward -= 0.15  # Increased penalty for standing still',
                'impact': 'Discourages static behavior'
            })
            improvements.append({
                'title': 'Increase Directional Movement Reward',
                'priority': 'MEDIUM',
                'file': 'train_neat.py',
                'line': '264',
                'current': 'shaped_reward += 1',
                'new': 'shaped_reward += 3  # Triple reward for purposeful movement',
                'impact': 'Strongly rewards moving toward ball'
            })
        
        elif weakness['name'] == 'Static Positioning':
            improvements.append({
                'title': 'Add Movement Range Bonus',
                'priority': 'HIGH',
                'file': 'train_neat.py',
                'line': '361',
                'current': '# Extra reward for competing well',
                'new': '''# Reward for good court coverage
                if max_x_position - min_x_position > 0.6:
                    ep_rew += 10.0  # Bonus for covering court''',
                'impact': 'Encourages exploring the court'
            })
        
        elif weakness['name'] == 'Slow Reactions':
            improvements.append({
                'title': 'Add Reaction Time Bonus',
                'priority': 'MEDIUM',
                'file': 'train_neat.py',
                'line': '235',
                'current': 'shaped_reward += 10.0',
                'new': '''shaped_reward += 10.0
                    # Extra bonus for quick reactions
                    if ball_on_our_side_timer < 3:
                        shaped_reward += 5.0''',
                'impact': 'Rewards faster responses'
            })
    
    # Add configuration improvements
    improvements.append({
        'title': 'Increase Population Size',
        'priority': 'MEDIUM',
        'file': 'neat_config_slime.cfg',
        'line': '5',
        'current': 'pop_size                  = 500',
        'new': 'pop_size                  = 1000',
        'impact': 'More diversity for finding better solutions'
    })
    
    improvements.append({
        'title': 'Increase Hidden Nodes',
        'priority': 'LOW',
        'file': 'neat_config_slime.cfg',
        'line': '48',
        'current': 'num_hidden                = 20',
        'new': 'num_hidden                = 30',
        'impact': 'More complex strategies possible'
    })
    
    return improvements

def generate_training_plots(metrics, training_analyzer, output_file):
    """Generate visualization plots for training analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('NEAT SlimeVolley Training Analysis', fontsize=16)
    
    # Plot 1: Fitness Progression
    if training_analyzer:
        progression = training_analyzer.analyze_fitness_progression()
        if progression:
            ax = axes[0, 0]
            ax.plot(progression['generations'], progression['best_fitness'], 'b-', label='Best', linewidth=2)
            ax.plot(progression['generations'], progression['avg_fitness'], 'g--', label='Average', linewidth=1)
            ax.fill_between(progression['generations'], 
                           np.array(progression['avg_fitness']) - np.array(progression['fitness_std']),
                           np.array(progression['avg_fitness']) + np.array(progression['fitness_std']),
                           alpha=0.3, color='green')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title('Fitness Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Plot 2: Skill Levels
    ax = axes[0, 1]
    skills = analyze_skill_levels(metrics)
    skill_names = list(skills.keys())
    skill_scores = [skills[s]['score'] for s in skill_names]
    colors = ['green' if s >= 7 else 'orange' if s >= 5 else 'red' for s in skill_scores]
    ax.bar(range(len(skill_names)), skill_scores, color=colors)
    ax.set_xticks(range(len(skill_names)))
    ax.set_xticklabels(skill_names, rotation=45, ha='right')
    ax.set_ylabel('Score (0-10)')
    ax.set_title('Skill Assessment')
    ax.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Expert Level')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Competent Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Win/Loss Distribution
    ax = axes[0, 2]
    outcomes = ['Wins', 'Losses', 'Draws']
    counts = [metrics['wins'], metrics['losses'], metrics['draws']]
    colors = ['green', 'red', 'gray']
    ax.pie(counts, labels=outcomes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Game Outcomes (n={metrics['total_episodes']})")
    
    # Plot 4: Performance Metrics
    ax = axes[1, 0]
    perf_metrics = {
        'Hit\nAccuracy': metrics['ball_control']['hit_accuracy'] * 100,
        'Movement\nEfficiency': metrics['movement']['movement_efficiency'] * 100,
        'Jump\nAccuracy': metrics['movement']['jump_accuracy'] * 100,
        'Defensive\nQuality': metrics['positioning'].get('defensive_quality', 0.5) * 100,
        'Offensive\nQuality': metrics['positioning'].get('offensive_quality', 0.5) * 100
    }
    ax.bar(range(len(perf_metrics)), list(perf_metrics.values()), 
           color=['green' if v >= 70 else 'orange' if v >= 50 else 'red' for v in perf_metrics.values()])
    ax.set_xticks(range(len(perf_metrics)))
    ax.set_xticklabels(list(perf_metrics.keys()))
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Performance Metrics')
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Game Statistics
    ax = axes[1, 1]
    game_stats = {
        'Avg Hits/Game': metrics['ball_control']['hits_per_game'],
        'Avg Rally': metrics['longest_rally'] / max(1, metrics['total_episodes']),
        'Court Coverage': metrics['positioning'].get('court_coverage', 0.5) * 10,
        'Reaction Time': 10 - min(10, metrics['ball_control']['reaction_time'])
    }
    ax.bar(range(len(game_stats)), list(game_stats.values()))
    ax.set_xticks(range(len(game_stats)))
    ax.set_xticklabels(list(game_stats.keys()), rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Game Statistics')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Score Distribution
    ax = axes[1, 2]
    score_types = ['Dominating\nWins', 'Close\nLosses', 'Shutouts', 'Got\nShutout', 'Comebacks']
    score_counts = [
        metrics['dominating_wins'],
        metrics['close_losses'],
        metrics['shutouts'],
        metrics['got_shutout'],
        metrics['tactical']['comeback_ability']
    ]
    colors = ['darkgreen', 'orange', 'green', 'red', 'blue']
    ax.bar(range(len(score_types)), score_counts, color=colors)
    ax.set_xticks(range(len(score_types)))
    ax.set_xticklabels(score_types)
    ax.set_ylabel('Count')
    ax.set_title('Special Game Outcomes')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate and Optimize NEAT SlimeVolley Training")
    parser.add_argument("--model", type=str, default="models/robust/winner.pkl", help="Path to model file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--checkpoint-dir", type=str, default="models/robust", help="Directory with checkpoints")
    parser.add_argument("--optimize", action="store_true", help="Generate optimization report")
    parser.add_argument("--save-report", type=str, help="Save report to JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation (25 episodes)")
    args = parser.parse_args()
    
    if args.quick:
        args.episodes = 25
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    print(f"Current model fitness: {pickle.load(open(args.model, 'rb'))[0].fitness:.2f}")
    
    # Initialize evaluator and analyzer
    evaluator = PerformanceEvaluator(args.model)
    training_analyzer = TrainingAnalyzer(args.checkpoint_dir) if args.optimize else None
    
    # Run comprehensive evaluation
    metrics = evaluator.evaluate_comprehensive(episodes=args.episodes)
    
    # Generate optimization report
    if args.optimize:
        performance_grade, weaknesses, improvements = generate_training_optimization_report(metrics, training_analyzer)
        
        # Save improvements to file for easy implementation
        with open("training_improvements.txt", "w") as f:
            f.write("TRAINING IMPROVEMENTS TO IMPLEMENT\n")
            f.write("="*50 + "\n\n")
            for imp in improvements:
                f.write(f"{imp['title']}\n")
                f.write(f"Priority: {imp['priority']}\n")
                f.write(f"File: {imp['file']} (Line {imp['line']})\n")
                f.write(f"Change:\n")
                f.write(f"  FROM: {imp['current']}\n")
                f.write(f"  TO:   {imp['new']}\n")
                f.write(f"Impact: {imp['impact']}\n")
                f.write("-"*50 + "\n\n")
        print(f"\n💾 Improvements saved to: training_improvements.txt")
    
    # Save report if requested
    if args.save_report:
        report_data = {
            'model': args.model,
            'metrics': metrics,
            'performance_grade': performance_grade if args.optimize else None,
            'weaknesses': weaknesses if args.optimize else None,
            'improvements': improvements if args.optimize else None
        }
        with open(args.save_report, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\n📊 Full report saved to: {args.save_report}")

if __name__ == "__main__":
    main()