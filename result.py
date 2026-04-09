import wandb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Check if wandb is authenticated
print("Checking wandb authentication...")
api = wandb.Api()

# Try to access the run with timeout
print("Accessing run...")
try:
    run = api.run("/jackytu/jaxgcrl_benchmark/runs/ig3f8p5s")
    print(f"Successfully accessed run: {run.name}")
except Exception as e:
    print(f"Error accessing run: {e}")
    print("Please ensure you are logged into wandb with: wandb login")
    exit(1)

# Fetch success metric data using pandas
print("Fetching success metric data using pandas...")

# Get history as pandas DataFrame
history_df = run.history(pandas=True)
print(f"Loaded {len(history_df)} rows of data")
print(f"Available columns: {list(history_df.columns)}")

# Check if success metric exists
if 'eval/episode_success_any' in history_df.columns:
    # Get non-NaN success data
    success_mask = history_df['eval/episode_success_any'].notna()
    success_data = history_df.loc[success_mask, 'eval/episode_success_any'].tolist()
    steps = history_df.loc[success_mask, '_step'].tolist()
    print(f"Found {len(success_data)} data points for success metric")
else:
    print("Column 'eval/episode_success_any' not found in data")
    success_data = []
    steps = []

# Also check summary for final success rate
summary_data = run.summary
config_data = run.config
print(f"Final episode reward: {summary_data.get('eval/episode_reward', 'N/A')}")

if success_data:
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, success_data, 'b-', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Success Rate (eval/episode_success_any)')
    plt.title('PPO Reacher Benchmark - Success Rate Over Time')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('results/success_rate_plot.png', dpi=150, bbox_inches='tight')
    print("Plot saved to results/success_rate_plot.png")
    
    # Show final success rate
    print(f"Final success rate: {success_data[-1]:.4f}")
    print(f"Max success rate: {max(success_data):.4f}")
    print(f"Average success rate: {np.mean(success_data):.4f}")
else:
    print("No success metric data found in the run")