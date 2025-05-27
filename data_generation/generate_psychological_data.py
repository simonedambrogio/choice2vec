#!/usr/bin/env python3
"""
Generate Psychological Behavioral Data for Choice2Vec

This version focuses on correct/incorrect choices rather than left/right choices,
and removes value_difference to force the model to learn psychological engagement patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PsychologicalDataGenerator:
    """
    Generate behavioral data focusing on psychological competence rather than stimulus features.
    
    Key changes:
    1. Choice is now correct/incorrect (not left/right)
    2. Value difference is removed from features
    3. Model must learn engagement patterns from choice accuracy and response times
    """
    
    def __init__(self, 
                 n_images: int = 3,
                 n_subtasks: int = 5,
                 trials_per_subtask: int = 100,
                 learning_rate_engaged: float = 0.35,
                 learning_rate_disengaged: float = 0.05,
                 transition_prob_engage: float = 0.92,
                 transition_prob_disengage: float = 0.88,
                 meta_learning_strength: float = 0.8,
                 random_seed: int = 42):
        
        self.n_images = n_images
        self.n_subtasks = n_subtasks
        self.trials_per_subtask = trials_per_subtask
        self.learning_rate_engaged = learning_rate_engaged
        self.learning_rate_disengaged = learning_rate_disengaged
        self.transition_prob_engage = transition_prob_engage
        self.transition_prob_disengage = transition_prob_disengage
        self.meta_learning_strength = meta_learning_strength
        
        np.random.seed(random_seed)
        
        # Initialize reward probabilities for each subtask
        self.reward_probs = self._generate_reward_probabilities()
        
    def _generate_reward_probabilities(self):
        """Generate reward probabilities for each image in each subtask."""
        reward_probs = {}
        
        for subtask in range(self.n_subtasks):
            # Create clear winner and loser for each subtask with better separation
            probs = np.random.uniform(0.05, 0.25, self.n_images)  # Even lower baseline
            
            # Ensure there's a clear best option (high reward)
            best_idx = np.random.randint(self.n_images)
            probs[best_idx] = np.random.uniform(0.8, 0.95)  # Even higher for best option
            
            # Ensure there's a clear worst option (low reward)
            worst_idx = np.random.choice([i for i in range(self.n_images) if i != best_idx])
            probs[worst_idx] = np.random.uniform(0.02, 0.15)  # Even lower for worst
            
            # If there's a middle option, make it clearly intermediate
            if self.n_images > 2:
                middle_indices = [i for i in range(self.n_images) if i != best_idx and i != worst_idx]
                for mid_idx in middle_indices:
                    probs[mid_idx] = np.random.uniform(0.25, 0.5)  # Clear intermediate range
            
            reward_probs[subtask] = probs
            
        return reward_probs
    
    def _get_psychological_state(self, current_state: str, trial_in_subtask: int) -> str:
        """Determine psychological state based on Markov chain transitions."""
        
        # Early trials have higher chance of being engaged (novelty effect)
        if trial_in_subtask < 10:
            engagement_boost = 0.1
        else:
            engagement_boost = 0.0
        
        if current_state == 'engaged':
            # Probability of staying engaged
            stay_prob = self.transition_prob_engage + engagement_boost
            if np.random.random() < stay_prob:
                return 'engaged'
            else:
                return 'disengaged'
        else:
            # Probability of staying disengaged
            stay_prob = self.transition_prob_disengage - engagement_boost
            if np.random.random() < stay_prob:
                return 'disengaged'
            else:
                return 'engaged'
    
    def _get_adaptive_learning_rate(self, base_lr: float, subtask: int, state: str) -> float:
        """Get learning rate that improves across subtasks (meta-learning)."""
        
        if state == 'engaged':
            meta_strength = self.meta_learning_strength
        else:
            meta_strength = self.meta_learning_strength * 0.2  # Disengaged participants show less meta-learning
        
        # Learning rate improves across subtasks
        improvement_factor = 1 + meta_strength * (subtask / (self.n_subtasks - 1))
        
        return base_lr * improvement_factor
    
    def _make_choice_and_get_outcome(self, q_values: np.ndarray, state: str, 
                                   trial_in_subtask: int, subtask: int) -> tuple:
        """
        Make choice and determine if it's correct, plus generate response time.
        
        Returns:
            choice_correct: bool (True if correct choice was made)
            response_time: float (response time in seconds)
            chosen_image: int (which image was chosen)
            was_rewarded: bool (whether reward was received)
        """
        
        if state == 'engaged':
            # Engaged participants make systematic choices based on learned values
            # Much lower temperature for more deterministic choices
            base_temperature = 0.5  # Reduced from 0.8
            min_temperature = 0.1   # Reduced from 0.2
            progress = trial_in_subtask / self.trials_per_subtask
            temperature = base_temperature - (base_temperature - min_temperature) * progress
            
            # Add small amount of noise to Q-values to simulate uncertainty
            noisy_q_values = q_values + np.random.normal(0, 0.02, len(q_values))  # Reduced noise further
            
            # Add exploration bonus for early trials (UCB-like)
            if trial_in_subtask < 15:  # Reduced from 20 trials
                exploration_bonus = 0.05 * (15 - trial_in_subtask) / 15  # Reduced bonus
                noisy_q_values += exploration_bonus
            
            # Softmax choice with temperature
            exp_values = np.exp(noisy_q_values / temperature)
            choice_probs = exp_values / np.sum(exp_values)
            
        else:
            # Disengaged participants make mostly random choices
            random_choice_prob = 0.7  # Reduced from 0.8 to allow some learning
            
            if np.random.random() < random_choice_prob:
                # Completely random choice
                choice_probs = np.ones(self.n_images) / self.n_images
            else:
                # Some influence from Q-values but with very high temperature
                temperature = 3.0  # Reduced from 4.0
                noisy_q_values = q_values + np.random.normal(0, 0.2, len(q_values))
                exp_values = np.exp(noisy_q_values / temperature)
                choice_probs = exp_values / np.sum(exp_values)
        
        # Make choice
        chosen_image = np.random.choice(self.n_images, p=choice_probs)
        
        # Determine if reward was received
        reward_prob = self.reward_probs[subtask][chosen_image]
        was_rewarded = np.random.random() < reward_prob
        
        # Determine if choice was "correct" (chose the image with highest reward probability)
        best_image = np.argmax(self.reward_probs[subtask])
        choice_correct = (chosen_image == best_image)
        
        # Generate response time based on state and task difficulty
        response_time = self._generate_response_time(state, q_values, chosen_image, trial_in_subtask)
        
        return choice_correct, response_time, chosen_image, was_rewarded
    
    def _generate_response_time(self, state: str, q_values: np.ndarray, 
                              chosen_image: int, trial_in_subtask: int) -> float:
        """Generate realistic response times based on psychological state and task factors."""
        
        if state == 'engaged':
            # Base RT for engaged participants
            base_rt = 0.8
            
            # Difficulty effect: harder decisions take longer
            value_range = np.max(q_values) - np.min(q_values)
            if value_range > 0:
                chosen_value = q_values[chosen_image]
                max_value = np.max(q_values)
                difficulty = 1 - (chosen_value / max_value)  # Higher when choice is suboptimal
                difficulty_effect = difficulty * 0.3
            else:
                difficulty_effect = 0.2  # Default uncertainty
            
            # Learning effect: early trials take longer
            learning_progress = trial_in_subtask / self.trials_per_subtask
            learning_effect = (1 - learning_progress) * 0.4
            
            # Combine effects
            mean_rt = base_rt + difficulty_effect + learning_effect
            
            # Generate from gamma distribution (realistic RT distribution)
            shape = 2.0  # Moderate variability
            scale = mean_rt / shape
            rt = np.random.gamma(shape, scale)
            
        else:
            # Disengaged participants have slower, more variable RTs
            base_rt = 1.2
            
            # Less sensitivity to task difficulty
            difficulty_effect = np.random.uniform(0, 0.2)
            
            # Higher random variability
            random_effect = np.random.uniform(0, 0.5)
            
            mean_rt = base_rt + difficulty_effect + random_effect
            
            # More variable RT distribution
            shape = 1.5  # Higher variability
            scale = mean_rt / shape
            rt = np.random.gamma(shape, scale)
        
        # Ensure reasonable RT bounds
        rt = np.clip(rt, 0.2, 5.0)
        
        return rt
    
    def generate_data(self) -> pd.DataFrame:
        """Generate complete behavioral dataset."""
        
        data = []
        
        for subtask in range(self.n_subtasks):
            print(f"Generating subtask {subtask + 1}/{self.n_subtasks}...")
            
            # Initialize Q-values with optimistic initialization (encourages exploration)
            q_values = np.random.uniform(0.4, 0.8, self.n_images)  # Higher optimistic start
            
            # Initialize psychological state (start engaged)
            current_state = 'engaged'
            
            for trial in range(self.trials_per_subtask):
                # Update psychological state
                current_state = self._get_psychological_state(current_state, trial)
                
                # Get adaptive learning rate
                if current_state == 'engaged':
                    base_lr = self.learning_rate_engaged
                else:
                    base_lr = self.learning_rate_disengaged
                
                learning_rate = self._get_adaptive_learning_rate(base_lr, subtask, current_state)
                
                # Make choice and get outcome
                choice_correct, rt, chosen_image, was_rewarded = self._make_choice_and_get_outcome(
                    q_values, current_state, trial, subtask
                )
                
                # Update Q-values based on outcome (Rescorla-Wagner learning)
                reward_value = 1.0 if was_rewarded else 0.0
                prediction_error = reward_value - q_values[chosen_image]
                q_values[chosen_image] += learning_rate * prediction_error
                
                # Calculate learning progress (how well task is learned)
                optimal_q = np.max(self.reward_probs[subtask])
                current_best_q = np.max(q_values)
                learning_progress = min(current_best_q / optimal_q, 1.0) if optimal_q > 0 else 0.0
                
                # Store trial data
                trial_data = {
                    'trial': len(data),
                    'subtask': subtask,
                    'trial_in_subtask': trial,
                    'psychological_state': current_state,
                    'choice_correct': choice_correct,  # NEW: correct/incorrect instead of left/right
                    'rt': rt,
                    'chosen_image': chosen_image,
                    'was_rewarded': was_rewarded,
                    'learning_rate': learning_rate,
                    'learning_progress': learning_progress,
                    # Note: value_difference is removed - no longer available to model
                }
                
                data.append(trial_data)
        
        df = pd.DataFrame(data)
        
        # Convert boolean to int for choice_correct
        df['choice_correct'] = df['choice_correct'].astype(int)
        
        print(f"\n✅ Generated {len(df)} trials")
        print(f"   Psychological states: {df['psychological_state'].value_counts().to_dict()}")
        print(f"   Choice accuracy: {df['choice_correct'].mean():.3f}")
        print(f"   Mean RT: {df['rt'].mean():.3f}s")
        
        return df
    
    def visualize_data(self, df: pd.DataFrame, save_plot: bool = True):
        """Create comprehensive visualizations of the generated data."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Choice accuracy by psychological state
        ax1 = axes[0, 0]
        accuracy_by_state = df.groupby('psychological_state')['choice_correct'].mean()
        bars = ax1.bar(accuracy_by_state.index, accuracy_by_state.values, 
                      color=['lightcoral', 'lightblue'], alpha=0.7)
        ax1.set_title('Choice Accuracy by Psychological State', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, accuracy_by_state.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Response time distributions by state
        ax2 = axes[0, 1]
        for state in ['engaged', 'disengaged']:
            state_data = df[df['psychological_state'] == state]['rt']
            ax2.hist(state_data, bins=30, alpha=0.6, label=state, density=True)
        ax2.set_title('Response Time Distributions', fontweight='bold')
        ax2.set_xlabel('Response Time (s)')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 3. Learning curves by subtask and state
        ax3 = axes[0, 2]
        for subtask in range(self.n_subtasks):
            subtask_data = df[df['subtask'] == subtask]
            
            # Calculate rolling accuracy
            window_size = 20
            rolling_acc = subtask_data['choice_correct'].rolling(window=window_size, center=True).mean()
            
            ax3.plot(subtask_data['trial_in_subtask'], rolling_acc, 
                    label=f'Subtask {subtask}', alpha=0.7)
        
        ax3.set_title('Learning Curves by Subtask', fontweight='bold')
        ax3.set_xlabel('Trial in Subtask')
        ax3.set_ylabel('Rolling Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. State transitions over time
        ax4 = axes[1, 0]
        
        # Calculate state persistence
        state_changes = []
        for i in range(1, len(df)):
            if df.iloc[i]['psychological_state'] != df.iloc[i-1]['psychological_state']:
                state_changes.append(i)
        
        # Plot state over time (sample every 10th trial for clarity)
        sample_indices = range(0, len(df), 10)
        sample_states = [1 if df.iloc[i]['psychological_state'] == 'engaged' else 0 
                        for i in sample_indices]
        sample_trials = [df.iloc[i]['trial'] for i in sample_indices]
        
        ax4.plot(sample_trials, sample_states, 'o-', alpha=0.6, markersize=3)
        ax4.set_title('Psychological State Over Time', fontweight='bold')
        ax4.set_xlabel('Trial Number')
        ax4.set_ylabel('State (0=Disengaged, 1=Engaged)')
        ax4.set_ylim(-0.1, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # 5. Accuracy vs Response Time by state
        ax5 = axes[1, 1]
        for state in ['engaged', 'disengaged']:
            state_data = df[df['psychological_state'] == state]
            ax5.scatter(state_data['rt'], state_data['choice_correct'], 
                       alpha=0.3, label=state, s=10)
        
        ax5.set_title('Accuracy vs Response Time', fontweight='bold')
        ax5.set_xlabel('Response Time (s)')
        ax5.set_ylabel('Choice Correct')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Meta-learning effect
        ax6 = axes[1, 2]
        
        # Calculate accuracy by subtask and state
        meta_learning_data = []
        for subtask in range(self.n_subtasks):
            for state in ['engaged', 'disengaged']:
                subtask_state_data = df[(df['subtask'] == subtask) & 
                                      (df['psychological_state'] == state)]
                if len(subtask_state_data) > 0:
                    accuracy = subtask_state_data['choice_correct'].mean()
                    meta_learning_data.append({
                        'subtask': subtask,
                        'state': state,
                        'accuracy': accuracy
                    })
        
        meta_df = pd.DataFrame(meta_learning_data)
        
        for state in ['engaged', 'disengaged']:
            state_meta = meta_df[meta_df['state'] == state]
            ax6.plot(state_meta['subtask'], state_meta['accuracy'], 
                    'o-', label=state, linewidth=2, markersize=6)
        
        ax6.set_title('Meta-Learning: Accuracy Across Subtasks', fontweight='bold')
        ax6.set_xlabel('Subtask')
        ax6.set_ylabel('Accuracy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'results/psychological_behavioral_data_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved as '{filename}'")
        
        plt.show()


def main():
    """Generate psychological behavioral data focusing on competence rather than stimulus features."""
    
    print("🧠 Generating Psychological Behavioral Data")
    print("Focus: Correct/Incorrect choices (not Left/Right)")
    print("=" * 60)
    
    # Create data generator
    generator = PsychologicalDataGenerator(
        n_images=3,
        n_subtasks=5,
        trials_per_subtask=1000,
        learning_rate_engaged=0.35,
        learning_rate_disengaged=0.05,
        transition_prob_engage=0.92,
        meta_learning_strength=0.8,
        random_seed=42
    )
    
    # Generate data
    df = generator.generate_data()
    
    # Save data
    df.to_csv('results/psychological_behavioral_data.csv', index=False)
    print(f"\n💾 Data saved as 'results/psychological_behavioral_data.csv'")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    generator.visualize_data(df, save_plot=True)
    
    # Print summary statistics
    print(f"\n📋 DATASET SUMMARY")
    print(f"=" * 40)
    print(f"Total trials: {len(df):,}")
    print(f"Subtasks: {df['subtask'].nunique()}")
    print(f"Psychological states: {df['psychological_state'].value_counts().to_dict()}")
    print(f"Overall accuracy: {df['choice_correct'].mean():.3f}")
    print(f"Engaged accuracy: {df[df['psychological_state'] == 'engaged']['choice_correct'].mean():.3f}")
    print(f"Disengaged accuracy: {df[df['psychological_state'] == 'disengaged']['choice_correct'].mean():.3f}")
    print(f"Mean RT: {df['rt'].mean():.3f}s (±{df['rt'].std():.3f})")
    
    print(f"\n🎯 KEY CHANGES FOR CHOICE2VEC:")
    print(f"✅ Choice is now correct/incorrect (not left/right)")
    print(f"✅ Value difference removed from features")
    print(f"✅ Model must learn psychological patterns from:")
    print(f"   - Choice accuracy patterns")
    print(f"   - Response time patterns") 
    print(f"   - Trial position and subtask context")
    print(f"✅ No trivial solution available!")


if __name__ == "__main__":
    main() 