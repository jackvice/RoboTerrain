import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def main():
    actor_clip = -0.7
    max_steps = 3_800_000
    #max_steps = 
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_tensorboard_log>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    
    try:
        data = load_and_process_data(log_file, actor_clip, max_steps)
        plot_metrics(data, actor_clip)
        print("Plot saved as 'training_metrics.png'")
    except Exception as e:
        print(f"Error processing log file: {e}")
        sys.exit(1)

def load_and_process_data(log_file, actor_clip, max_steps):
    # Load the log file
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    
    # Extract the metrics
    reward_data = ea.Scalars('rollout/ep_rew_mean')
    actor_loss_data = ea.Scalars('train/actor_loss')
    critic_loss_data = ea.Scalars('train/critic_loss')
    
    # Convert to numpy arrays and limit to 1M steps, with value clipping
    def process_metric(data, min_steps=0, min_clip=None, max_clip=None):
        steps = np.array([x.step for x in data])
        values = np.array([x.value for x in data])
        #mask = (steps <= 900000) & (steps >= min_steps)
        mask = (steps <= max_steps) & (steps >= min_steps)
        steps = steps[mask]
        values = values[mask]
        
        if min_clip is not None:
            values = np.maximum(values, min_clip)
        if max_clip is not None:
            values = np.minimum(values, max_clip)
            
        return steps, values
    
    reward_steps, reward_values = process_metric(reward_data)
    actor_steps, actor_values = process_metric(actor_loss_data, 
                                             min_steps=1,
                                               min_clip=actor_clip)  # Clip actor loss below -0.x
    critic_steps, critic_values = process_metric(critic_loss_data,
                                               max_clip=0.02)  # Clip critic loss above 0.02
    
    return {
        'reward': (reward_steps, reward_values),
        'actor': (actor_steps, actor_values),
        'critic': (critic_steps, critic_values)
    }

def plot_metrics(data, actor_clip):
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
    
    # Plot losses on first subplot with two different y-axes
    color1, color2 = 'red', 'orange'
    
    # Plot critic loss on left y-axis
    ln1 = ax1.plot(data['critic'][0], data['critic'][1], color=color1, label='Critic Loss (clipped at 0.02)', alpha=0.8)
    ax1.set_ylabel('Critic Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for actor loss
    ax1_twin = ax1.twinx()
    ln2 = ax1_twin.plot(data['actor'][0], data['actor'][1], color=color2, label='Actor Loss (clipped at'
                        + str(actor_clip) +')', alpha=0.8)
    ax1_twin.set_ylabel('Actor Loss', color=color2)
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    # Add legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    ax1.set_title('Actor and Critic Losses (Clipped)')
    ax1.set_xlabel('Steps')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Plot reward on second subplot
    ax2.plot(data['reward'][0], data['reward'][1], 'b-', label='Episode Reward Mean', alpha=0.8)
    ax2.set_title('Episode Reward Mean')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



if __name__ == "__main__":
    main()
