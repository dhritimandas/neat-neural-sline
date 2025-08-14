#!/usr/bin/env python3
"""
Real-time Training Monitor for NEAT SlimeVolley
Displays live performance metrics during training
"""

import os
import sys
import pickle
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

def load_generation_stats(stats_file):
    """Load generation statistics"""
    if not os.path.exists(stats_file):
        return []
    try:
        with open(stats_file, 'rb') as f:
            return pickle.load(f)
    except:
        return []

def load_performance_history(history_file):
    """Load detailed performance history"""
    if not os.path.exists(history_file):
        return {}
    try:
        with open(history_file, 'rb') as f:
            return pickle.load(f)
    except:
        return {}

def calculate_improvement_rate(stats, window=10):
    """Calculate fitness improvement rate over recent generations"""
    if len(stats) < 2:
        return 0.0
    
    recent = stats[-min(window, len(stats)):]
    if len(recent) < 2:
        return 0.0
    
    start_fitness = recent[0]['best_fitness']
    end_fitness = recent[-1]['best_fitness']
    generations = len(recent) - 1
    
    if generations == 0 or start_fitness == 0:
        return 0.0
    
    return ((end_fitness - start_fitness) / start_fitness) * 100 / generations

def display_dashboard(model_dir, refresh_rate=5):
    """Display live training dashboard"""
    model_path = Path(model_dir)
    stats_file = model_path / "generation_stats.pkl"
    history_file = model_path / "performance_history.pkl"
    
    print("\033[2J\033[H")  # Clear screen
    print("="*80)
    print("NEAT SLIMEVOLLEY TRAINING MONITOR")
    print("="*80)
    print(f"Model Directory: {model_path}")
    print(f"Refresh Rate: {refresh_rate}s")
    print("Press Ctrl+C to exit\n")
    
    last_generation = -1
    
    try:
        while True:
            # Load latest data
            stats = load_generation_stats(stats_file)
            history = load_performance_history(history_file)
            
            if not stats:
                print("\rWaiting for training data...", end="", flush=True)
                time.sleep(refresh_rate)
                continue
            
            latest = stats[-1]
            generation = latest['generation']
            
            # Only update if new generation
            if generation != last_generation:
                last_generation = generation
                
                # Clear and redraw
                print("\033[2J\033[H")  # Clear screen
                print("="*80)
                print(f"NEAT SLIMEVOLLEY TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
                print("="*80)
                
                # Current Generation Stats
                print(f"\n📊 GENERATION {generation}")
                print(f"   Best Fitness: {latest['best_fitness']:.2f}")
                print(f"   Average Fitness: {latest['avg_fitness']:.2f} ± {latest['std_fitness']:.2f}")
                print(f"   Population Size: {latest['population_size']}")
                print(f"   Agents with Wins: {latest.get('agents_with_wins', 0)}")
                
                # Progress Indicators
                if len(stats) > 1:
                    prev = stats[-2]
                    fitness_change = latest['best_fitness'] - prev['best_fitness']
                    avg_change = latest['avg_fitness'] - prev['avg_fitness']
                    
                    print(f"\n📈 PROGRESS")
                    print(f"   Best Fitness Change: {fitness_change:+.2f}")
                    print(f"   Avg Fitness Change: {avg_change:+.2f}")
                    print(f"   Improvement Rate (10 gen): {calculate_improvement_rate(stats):.2f}% per gen")
                
                # Performance Milestones
                print(f"\n🏆 MILESTONES")
                all_time_best = max(s['best_fitness'] for s in stats)
                best_gen = next(s['generation'] for s in stats if s['best_fitness'] == all_time_best)
                print(f"   All-Time Best: {all_time_best:.2f} (Gen {best_gen})")
                
                # Count key achievements
                wins_achieved = any(s.get('agents_with_wins', 0) > 0 for s in stats)
                high_performers = sum(1 for s in stats if s['best_fitness'] > 100)
                elite_performers = sum(1 for s in stats if s['best_fitness'] > 500)
                
                print(f"   Generations with Winners: {sum(1 for s in stats if s.get('agents_with_wins', 0) > 0)}")
                print(f"   High Performance Gens (>100): {high_performers}")
                print(f"   Elite Performance Gens (>500): {elite_performers}")
                
                # Training Health Indicators
                print(f"\n⚡ TRAINING HEALTH")
                
                # Check for stagnation
                if len(stats) >= 10:
                    recent_best = [s['best_fitness'] for s in stats[-10:]]
                    if max(recent_best) - min(recent_best) < 10:
                        print("   ⚠️  WARNING: Fitness stagnating (last 10 generations)")
                    else:
                        print("   ✅ Fitness improving steadily")
                
                # Check diversity
                if latest['std_fitness'] < 10:
                    print("   ⚠️  WARNING: Low population diversity (std < 10)")
                else:
                    print(f"   ✅ Good population diversity (std = {latest['std_fitness']:.1f})")
                
                # Check for winners
                if latest.get('agents_with_wins', 0) == 0:
                    print("   ⚠️  WARNING: No agents winning points yet")
                else:
                    win_rate = latest.get('agents_with_wins', 0) / latest['population_size'] * 100
                    print(f"   ✅ {win_rate:.1f}% of population winning points")
                
                # Expert Status Check
                print(f"\n🎯 EXPERT STATUS CHECK")
                expert_criteria = {
                    'Fitness > 500': all_time_best > 500,
                    'Consistent Winners': sum(1 for s in stats[-10:] if s.get('agents_with_wins', 0) > 0) >= 8 if len(stats) >= 10 else False,
                    'Population Winners > 50%': latest.get('agents_with_wins', 0) > latest['population_size'] * 0.5,
                    'Low Variance': latest['std_fitness'] < 50 and latest['avg_fitness'] > 100
                }
                
                met_criteria = sum(expert_criteria.values())
                for criterion, met in expert_criteria.items():
                    status = "✅" if met else "❌"
                    print(f"   {status} {criterion}")
                
                print(f"\n   Expert Level: {met_criteria}/4 criteria met")
                if met_criteria == 4:
                    print("   🏆 EXPERT LEVEL ACHIEVED!")
                elif met_criteria >= 3:
                    print("   ⭐ Close to expert level!")
                elif met_criteria >= 2:
                    print("   📈 Making good progress")
                else:
                    print("   💪 Keep training!")
                
                # Recommendations
                print(f"\n💡 RECOMMENDATIONS")
                if latest['best_fitness'] < 50:
                    print("   • Focus on basic ball contact - consider easier opponent")
                elif latest['best_fitness'] < 200:
                    print("   • Improving! Consider increasing episode count for stability")
                elif latest['best_fitness'] < 500:
                    print("   • Good progress! Try elite competition for final push")
                else:
                    print("   • Excellent! Fine-tune with self-play and harder opponents")
                
                if latest['std_fitness'] < 10:
                    print("   • Increase mutation rates to maintain diversity")
                
                if calculate_improvement_rate(stats) < 1:
                    print("   • Consider adjusting reward weights or curriculum difficulty")
                
                print("\n" + "="*80)
                print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("Monitoring... (Ctrl+C to exit)")
            
            time.sleep(refresh_rate)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        return

def main():
    parser = argparse.ArgumentParser(description="Monitor NEAT SlimeVolley Training")
    parser.add_argument("--model-dir", type=str, default="models/optimized", 
                       help="Directory containing training data")
    parser.add_argument("--refresh", type=int, default=5, 
                       help="Refresh rate in seconds")
    parser.add_argument("--summary", action="store_true",
                       help="Show summary and exit")
    args = parser.parse_args()
    
    if args.summary:
        # Just show final summary
        model_path = Path(args.model_dir)
        stats_file = model_path / "generation_stats.pkl"
        stats = load_generation_stats(stats_file)
        
        if not stats:
            print("No training data found.")
            return
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Total Generations: {len(stats)}")
        print(f"Best Fitness Achieved: {max(s['best_fitness'] for s in stats):.2f}")
        print(f"Final Average Fitness: {stats[-1]['avg_fitness']:.2f}")
        print(f"Final Population Winners: {stats[-1].get('agents_with_wins', 0)}/{stats[-1]['population_size']}")
        
        # Plot fitness progression if matplotlib available
        try:
            import matplotlib.pyplot as plt
            
            generations = [s['generation'] for s in stats]
            best_fitness = [s['best_fitness'] for s in stats]
            avg_fitness = [s['avg_fitness'] for s in stats]
            
            plt.figure(figsize=(10, 6))
            plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
            plt.plot(generations, avg_fitness, 'g--', label='Average Fitness', linewidth=1)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('NEAT SlimeVolley Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = model_path / "training_progress.png"
            plt.savefig(plot_file)
            print(f"\nPlot saved to: {plot_file}")
            plt.show()
            
        except ImportError:
            print("\n(Install matplotlib to see training plots)")
    else:
        # Live monitoring
        display_dashboard(args.model_dir, args.refresh)

if __name__ == "__main__":
    main()