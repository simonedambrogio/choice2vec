import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Dict
import seaborn as sns

class BehavioralDataGenerator:
    """
    Generates behavioral data for a two-alternative choice task with:
    - Rescorla-Wagner learning dynamics
    - Psychological states (engaged/disengaged)
    - State-dependent response time distributions
    - Context-dependent behavior (learning vs learned)
    """
    
    def __init__(self, 
                 n_images: int = 3,
                 n_subtasks: int = 5,
                 trials_per_subtask: int = 100,
                 learning_rate_engaged: float = 0.1,
                 learning_rate_disengaged: float = 0.02,
                 transition_prob_engage: float = 0.95,  # P(engaged -> engaged)
                 transition_prob_disengage: float = 0.85,  # P(disengaged -> disengaged)
                 noise_level: float = 0.1,
                 meta_learning_strength: float = 0.8,  # How much learning-to-learn effect
                 random_seed: int = 42):
        
        self.n_images = n_images
        self.n_subtasks = n_subtasks
        self.trials_per_subtask = trials_per_subtask
        self.learning_rate_engaged = learning_rate_engaged
        self.learning_rate_disengaged = learning_rate_disengaged
        self.transition_prob_engage = transition_prob_engage
        self.transition_prob_disengage = transition_prob_disengage
        self.noise_level = noise_level
        self.meta_learning_strength = meta_learning_strength
        
        np.random.seed(random_seed)
        
        # Define reward values for images (will be shuffled each subtask)
        self.reward_values = [0.2, 0.5, 0.8]  # low, medium, high
        
        # Initialize psychological state transition matrix
        self.transition_matrix = np.array([
            [transition_prob_engage, 1 - transition_prob_engage],      # engaged -> [engaged, disengaged]
            [1 - transition_prob_disengage, transition_prob_disengage] # disengaged -> [engaged, disengaged]
        ])
        
    def generate_subtask_mapping(self) -> Dict[int, float]:
        """Generate random image-reward mapping for a subtask"""
        shuffled_rewards = np.random.permutation(self.reward_values)
        return {i: shuffled_rewards[i] for i in range(self.n_images)}
    
    def get_adaptive_learning_rate(self, base_learning_rate: float, subtask: int, state: str) -> float:
        """
        Calculate adaptive learning rate based on subtask experience (learning-to-learn effect)
        
        Args:
            base_learning_rate: Base learning rate for this state
            subtask: Current subtask number (0-indexed)
            state: 'engaged' or 'disengaged'
        
        Returns:
            Adaptive learning rate that increases with subtask experience
        """
        if state == 'disengaged':
            # Disengaged agents show minimal learning-to-learn effect
            meta_effect = 1 + (self.meta_learning_strength * 0.2) * (subtask / (self.n_subtasks - 1))
        else:
            # Engaged agents show strong learning-to-learn effect
            # Learning rate increases exponentially with subtask experience
            meta_effect = 1 + self.meta_learning_strength * (subtask / (self.n_subtasks - 1))
        
        adaptive_lr = base_learning_rate * meta_effect
        
        # Cap the learning rate to prevent unrealistic values
        max_lr = base_learning_rate * (1 + self.meta_learning_strength)
        return min(adaptive_lr, max_lr)
    
    def rescorla_wagner_update(self, q_value: float, reward: float, learning_rate: float) -> float:
        """Update Q-value using Rescorla-Wagner rule"""
        prediction_error = reward - q_value
        return q_value + learning_rate * prediction_error
    
    def softmax_choice(self, q_values: np.ndarray, temperature: float = 1.0) -> int:
        """Make choice using softmax policy"""
        exp_q = np.exp(q_values / temperature)
        probabilities = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probabilities)
    
    def generate_rt(self, 
                   state: str, 
                   value_difference: float, 
                   learning_progress: float,
                   trial_in_subtask: int) -> float:
        """
        Generate response time based on psychological state and task factors
        
        Args:
            state: 'engaged' or 'disengaged'
            value_difference: absolute difference in Q-values between options
            learning_progress: how well learned the task is (0-1)
            trial_in_subtask: trial number within current subtask
        """
        
        if state == 'engaged':
            # Engaged: RT depends on value difference and learning
            base_rt = 0.8  # base RT in seconds
            
            # Harder decisions (small value diff) take longer
            difficulty_effect = 0.3 * np.exp(-5 * abs(value_difference))
            
            # Early in learning, decisions take longer
            learning_effect = 0.4 * np.exp(-learning_progress * 3)
            
            # Small trial effect (fatigue/practice)
            trial_effect = 0.1 * (trial_in_subtask / self.trials_per_subtask)
            
            mean_rt = base_rt + difficulty_effect + learning_effect + trial_effect
            
            # Gamma distribution for realistic RT shape
            shape = 2.0
            scale = mean_rt / shape
            rt = np.random.gamma(shape, scale)
            
        else:  # disengaged
            # Disengaged: RT is more variable and less dependent on task factors
            base_rt = 1.2  # slower overall
            
            # Less sensitivity to difficulty
            difficulty_effect = 0.1 * np.exp(-2 * abs(value_difference))
            
            # Random variability
            random_effect = np.random.normal(0, 0.3)
            
            mean_rt = base_rt + difficulty_effect + random_effect
            mean_rt = max(0.3, mean_rt)  # minimum RT
            
            # More variable RT distribution
            shape = 1.5
            scale = mean_rt / shape
            rt = np.random.gamma(shape, scale)
        
        return max(0.2, rt)  # minimum physiologically plausible RT
    
    def generate_choice(self, 
                       q_values: np.ndarray, 
                       state: str, 
                       learning_progress: float) -> int:
        """
        Generate choice based on Q-values, state, and learning progress
        """
        
        if state == 'engaged':
            # Engaged: choices follow Q-values with appropriate exploration/exploitation
            if learning_progress < 0.2:  # very early learning - high exploration
                temperature = 1.5
            elif learning_progress < 0.5:  # mid learning - moderate exploration
                temperature = 1
            else:  # later learning - more exploitation
                temperature = 0.4
                
            choice = self.softmax_choice(q_values, temperature)
            
        else:  # disengaged
            # Disengaged: much more random choices, poor decision making
            random_prob = 0.8  # 80% completely random choices when disengaged
            
            if np.random.random() < random_prob:
                # Completely random choice (ignoring Q-values)
                choice = np.random.choice(len(q_values))
            else:
                # Even when "trying", very poor decision making (high temperature)
                # Also add some noise to Q-values to simulate inattention
                noisy_q_values = q_values + np.random.normal(0, 0.3, size=len(q_values))
                choice = self.softmax_choice(noisy_q_values, temperature=4.0)
        
        return choice
    
    def simulate_psychological_states(self, total_trials: int) -> List[str]:
        """Simulate sequence of psychological states using transition matrix"""
        states = []
        current_state = 0  # start engaged
        
        for _ in range(total_trials):
            states.append('engaged' if current_state == 0 else 'disengaged')
            
            # Transition to next state
            current_state = np.random.choice(2, p=self.transition_matrix[current_state])
        
        return states
    
    def generate_data(self) -> pd.DataFrame:
        """Generate complete behavioral dataset"""
        
        all_data = []
        total_trials = self.n_subtasks * self.trials_per_subtask
        
        # Generate psychological states for entire session
        psychological_states = self.simulate_psychological_states(total_trials)
        
        trial_idx = 0
        
        for subtask in range(self.n_subtasks):
            print(f"Generating subtask {subtask + 1}/{self.n_subtasks}")
            
            # New image-reward mapping for this subtask
            reward_mapping = self.generate_subtask_mapping()
            
            # Initialize Q-values
            q_values = np.ones(self.n_images) * 0.5  # neutral starting values
            
            for trial_in_subtask in range(self.trials_per_subtask):
                
                # Current psychological state
                current_state = psychological_states[trial_idx]
                
                # Calculate learning progress (how well learned the current mapping is)
                # Based on how close Q-values are to true rewards
                true_rewards = np.array([reward_mapping[i] for i in range(self.n_images)])
                learning_progress = 1 - np.mean(np.abs(q_values - true_rewards))
                learning_progress = max(0, min(1, learning_progress))
                
                # Present two random images
                presented_images = np.random.choice(self.n_images, size=2, replace=False)
                image_a, image_b = presented_images
                
                # Get Q-values for presented images
                q_a, q_b = q_values[image_a], q_values[image_b]
                q_presented = np.array([q_a, q_b])
                
                # Calculate value difference
                value_difference = q_a - q_b
                
                # Generate choice (0 for image_a, 1 for image_b)
                choice_idx = self.generate_choice(q_presented, current_state, learning_progress)
                chosen_image = presented_images[choice_idx]
                
                # Generate response time
                rt = self.generate_rt(current_state, value_difference, learning_progress, trial_in_subtask)
                
                # Get reward for chosen image
                reward = reward_mapping[chosen_image]
                
                # Add noise to reward
                noisy_reward = reward + np.random.normal(0, self.noise_level)
                noisy_reward = max(0, min(1, noisy_reward))  # clip to [0,1]
                
                # Update Q-value using Rescorla-Wagner with adaptive learning rate
                base_learning_rate = (self.learning_rate_engaged if current_state == 'engaged' 
                                     else self.learning_rate_disengaged)
                
                # Apply learning-to-learn effect
                learning_rate = self.get_adaptive_learning_rate(base_learning_rate, subtask, current_state)
                
                q_values[chosen_image] = self.rescorla_wagner_update(
                    q_values[chosen_image], noisy_reward, learning_rate
                )
                
                # Store trial data
                trial_data = {
                    'trial_global': trial_idx,
                    'subtask': subtask,
                    'trial_in_subtask': trial_in_subtask,
                    'image_a': image_a,
                    'image_b': image_b,
                    'choice': choice_idx,  # 0 or 1
                    'chosen_image': chosen_image,
                    'rt': rt,
                    'reward': noisy_reward,
                    'value_difference': abs(value_difference),
                    'q_value_chosen': q_values[chosen_image],
                    'q_value_unchosen': q_presented[1 - choice_idx],
                    'learning_progress': learning_progress,
                    'psychological_state': current_state,
                    'true_reward_chosen': reward_mapping[chosen_image],
                    'learning_rate': learning_rate
                }
                
                all_data.append(trial_data)
                trial_idx += 1
        
        return pd.DataFrame(all_data)
    
    def plot_data_summary(self, df: pd.DataFrame, save_path: str = None):
        """Create summary plots of the generated data"""
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 1. Psychological states over time
        ax = axes[0, 0]
        state_numeric = df['psychological_state'].map({'engaged': 1, 'disengaged': 0})
        ax.plot(df['trial_global'], state_numeric, alpha=0.7)
        ax.set_ylabel('Psychological State')
        ax.set_xlabel('Trial')
        ax.set_title('Psychological States Over Time')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Disengaged', 'Engaged'])
        
        # 2. RT by state
        ax = axes[0, 1]
        sns.boxplot(data=df, x='psychological_state', y='rt', ax=ax)
        ax.set_title('Response Time by Psychological State')
        
        # 3. Learning rate evolution (meta-learning effect)
        ax = axes[0, 3]
        for state in ['engaged', 'disengaged']:
            state_data = df[df['psychological_state'] == state]
            lr_by_subtask = state_data.groupby('subtask')['learning_rate'].mean()
            ax.plot(lr_by_subtask.index, lr_by_subtask.values, 
                   marker='o', linewidth=2, label=state)
        ax.set_xlabel('Subtask')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Meta-Learning: LR Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Learning progress over subtasks (engaged trials only for clarity)
        ax = axes[0, 2]
        engaged_data = df[df['psychological_state'] == 'engaged']
        for subtask in engaged_data['subtask'].unique():
            subtask_data = engaged_data[engaged_data['subtask'] == subtask]
            ax.plot(subtask_data['trial_in_subtask'], subtask_data['learning_progress'], 
                   alpha=0.8, label=f'Subtask {subtask}', linewidth=2)
        ax.set_xlabel('Trial in Subtask')
        ax.set_ylabel('Learning Progress')
        ax.set_title('Learning Curves by Subtask (Engaged Only)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Choice accuracy by state and learning progress
        ax = axes[1, 0]
        # Define "correct" choice as choosing higher value option
        # Handle case where Q-values are equal (random choice is neither correct nor incorrect)
        df['correct_choice'] = (
            (df['q_value_chosen'] > df['q_value_unchosen'])
        )
        
        accuracy_by_state = df.groupby(['psychological_state', 'trial_in_subtask'])['correct_choice'].mean().reset_index()
        for state in ['engaged', 'disengaged']:
            state_data = accuracy_by_state[accuracy_by_state['psychological_state'] == state]
            ax.plot(state_data['trial_in_subtask'], state_data['correct_choice'], 
                   label=state, alpha=0.8)
        ax.set_xlabel('Trial in Subtask')
        ax.set_ylabel('Choice Accuracy')
        ax.set_title('Choice Accuracy by State')
        ax.legend()
        
        # 6. RT vs Value Difference by State
        ax = axes[1, 1]
        for state in ['engaged', 'disengaged']:
            state_data = df[df['psychological_state'] == state]
            ax.scatter(state_data['value_difference'], state_data['rt'], 
                      alpha=0.5, label=state, s=10)
        ax.set_xlabel('Value Difference')
        ax.set_ylabel('Response Time')
        ax.set_title('RT vs Value Difference by State')
        ax.legend()
        
        # 8. State transition frequencies
        ax = axes[1, 2]
        transitions = []
        for i in range(1, len(df)):
            prev_state = df.iloc[i-1]['psychological_state']
            curr_state = df.iloc[i]['psychological_state']
            transitions.append(f"{prev_state} -> {curr_state}")
        
        transition_counts = pd.Series(transitions).value_counts()
        transition_counts.plot(kind='bar', ax=ax)
        ax.set_title('State Transition Frequencies')
        ax.tick_params(axis='x', rotation=45)
        
        # 8. Learning-to-learn effect: Learning rate by subtask and state
        ax = axes[1, 3]
        learning_rate_by_subtask = df.groupby(['subtask', 'psychological_state'])['learning_rate'].mean().unstack()
        learning_rate_by_subtask.plot(kind='line', ax=ax, marker='o')
        ax.set_xlabel('Subtask')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning-to-Learn Effect')
        ax.legend(title='State')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Generate and save behavioral data"""
    
    # Initialize generator
    generator = BehavioralDataGenerator(
        n_images=3,
        n_subtasks=5,
        trials_per_subtask=500,  # 500 trials per subtask
        learning_rate_engaged=0.15,
        learning_rate_disengaged=0.03,
        transition_prob_engage=0.92,
        transition_prob_disengage=0.88,
        noise_level=0.1,
        meta_learning_strength=0.8,  # Strong learning-to-learn effect
        random_seed=42
    )
    
    # Generate data
    print("Generating behavioral data...")
    df = generator.generate_data()
    
    # Save data
    df.to_csv('behavioral_data.csv', index=False)
    print(f"Data saved to behavioral_data.csv")
    print(f"Generated {len(df)} trials across {df['subtask'].nunique()} subtasks")
    
    # Print summary statistics
    print("\n=== Data Summary ===")
    print(f"Total trials: {len(df)}")
    print(f"Subtasks: {df['subtask'].nunique()}")
    print(f"Psychological state distribution:")
    print(df['psychological_state'].value_counts(normalize=True))
    print(f"\nMean RT by state:")
    print(df.groupby('psychological_state')['rt'].mean())
    print(f"\nMean learning rate by state:")
    print(df.groupby('psychological_state')['learning_rate'].mean())
    
    # Create summary plots
    print("\nCreating summary plots...")
    generator.plot_data_summary(df, 'behavioral_data_summary.png')
    
    return df


if __name__ == "__main__":
    df = main()
