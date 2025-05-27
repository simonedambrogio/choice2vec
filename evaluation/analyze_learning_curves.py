#!/usr/bin/env python3
"""
Analyze Learning Curves in Psychological Data

This script analyzes the generated psychological data to verify that:
1. Engaged participants show clear learning curves (improvement over trials)
2. Learning curves are less variable and reach higher accuracy
3. Meta-learning occurs across subtasks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_learning_curves():
    """Analyze learning curves in the psychological data."""
    
    # Load data
    df = pd.read_csv('psychological_behavioral_data.csv')
    
    print("📊 LEARNING CURVE ANALYSIS")
    print("=" * 60)
    
    # Analyze within-subtask learning for engaged participants
    print("\n🧠 WITHIN-SUBTASK LEARNING (Engaged Participants)")
    print("-" * 50)
    
    engaged_improvements = []
    
    for subtask in range(5):
        subtask_data = df[(df['subtask'] == subtask) & (df['psychological_state'] == 'engaged')]
        
        if len(subtask_data) < 40:  # Need enough trials
            continue
            
        # Calculate accuracy in first vs last portions
        n_trials = len(subtask_data)
        first_quarter = subtask_data.head(n_trials//4)['choice_correct'].mean()
        last_quarter = subtask_data.tail(n_trials//4)['choice_correct'].mean()
        
        improvement = last_quarter - first_quarter
        engaged_improvements.append(improvement)
        
        print(f"Subtask {subtask}:")
        print(f"  First quarter: {first_quarter:.3f}")
        print(f"  Last quarter:  {last_quarter:.3f}")
        print(f"  Improvement:   {improvement:.3f}")
    
    avg_improvement = np.mean(engaged_improvements)
    print(f"\n📈 Average within-subtask improvement: {avg_improvement:.3f}")
    
    # Analyze meta-learning across subtasks
    print("\n🎯 META-LEARNING ACROSS SUBTASKS")
    print("-" * 40)
    
    for state in ['engaged', 'disengaged']:
        print(f"\n{state.upper()} participants:")
        subtask_accuracies = []
        
        for subtask in range(5):
            subtask_acc = df[(df['subtask'] == subtask) & 
                           (df['psychological_state'] == state)]['choice_correct'].mean()
            subtask_accuracies.append(subtask_acc)
            print(f"  Subtask {subtask}: {subtask_acc:.3f}")
        
        # Calculate trend
        if len(subtask_accuracies) >= 3:
            trend = np.polyfit(range(len(subtask_accuracies)), subtask_accuracies, 1)[0]
            print(f"  Meta-learning trend: {trend:.3f} per subtask")
    
    # Create visualization
    create_learning_visualization(df)
    
    return df

def create_learning_visualization(df):
    """Create detailed learning curve visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Within-subtask learning curves
    ax1 = axes[0, 0]
    
    for subtask in range(5):
        for state in ['engaged', 'disengaged']:
            subtask_data = df[(df['subtask'] == subtask) & 
                            (df['psychological_state'] == state)]
            
            if len(subtask_data) < 20:
                continue
            
            # Calculate rolling accuracy
            window_size = min(50, len(subtask_data) // 4)
            rolling_acc = subtask_data['choice_correct'].rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
            
            color = 'blue' if state == 'engaged' else 'red'
            alpha = 0.7 if state == 'engaged' else 0.3
            linestyle = '-' if state == 'engaged' else '--'
            
            ax1.plot(subtask_data['trial_in_subtask'], rolling_acc, 
                    color=color, alpha=alpha, linestyle=linestyle,
                    label=f'{state.title()} (S{subtask})' if subtask == 0 else "")
    
    ax1.set_title('Learning Curves by Subtask', fontweight='bold')
    ax1.set_xlabel('Trial in Subtask')
    ax1.set_ylabel('Rolling Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Meta-learning across subtasks
    ax2 = axes[0, 1]
    
    meta_data = []
    for subtask in range(5):
        for state in ['engaged', 'disengaged']:
            acc = df[(df['subtask'] == subtask) & 
                    (df['psychological_state'] == state)]['choice_correct'].mean()
            meta_data.append({'subtask': subtask, 'state': state, 'accuracy': acc})
    
    meta_df = pd.DataFrame(meta_data)
    
    for state in ['engaged', 'disengaged']:
        state_data = meta_df[meta_df['state'] == state]
        color = 'blue' if state == 'engaged' else 'red'
        ax2.plot(state_data['subtask'], state_data['accuracy'], 
                'o-', color=color, linewidth=2, markersize=8, label=state.title())
    
    ax2.set_title('Meta-Learning Across Subtasks', fontweight='bold')
    ax2.set_xlabel('Subtask')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Accuracy distribution by state
    ax3 = axes[1, 0]
    
    engaged_acc = df[df['psychological_state'] == 'engaged']['choice_correct']
    disengaged_acc = df[df['psychological_state'] == 'disengaged']['choice_correct']
    
    ax3.hist(engaged_acc, bins=2, alpha=0.7, label='Engaged', color='blue', density=True)
    ax3.hist(disengaged_acc, bins=2, alpha=0.7, label='Disengaged', color='red', density=True)
    
    ax3.set_title('Choice Accuracy Distribution', fontweight='bold')
    ax3.set_xlabel('Choice Correct (0/1)')
    ax3.set_ylabel('Density')
    ax3.legend()
    
    # 4. Response time by accuracy and state
    ax4 = axes[1, 1]
    
    # Create accuracy bins
    df['acc_bin'] = df['choice_correct'].map({0: 'Incorrect', 1: 'Correct'})
    
    sns.boxplot(data=df, x='acc_bin', y='rt', hue='psychological_state', ax=ax4)
    ax4.set_title('Response Time by Accuracy and State', fontweight='bold')
    ax4.set_xlabel('Choice Outcome')
    ax4.set_ylabel('Response Time (s)')
    
    plt.tight_layout()
    plt.savefig('learning_curves_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Learning curve analysis saved as 'learning_curves_analysis.png'")
    plt.show()

if __name__ == "__main__":
    df = analyze_learning_curves() 