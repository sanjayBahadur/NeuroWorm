"""DQN baseline agent for comparison against the worm brain."""
# Requires: torch>=2.2
# Usage: python training/dqn_baseline.py --train
#        python training/dqn_baseline.py --eval

# Full DQN implementation coming in June sprint.
# Architecture: 3-layer MLP, input=game state (11 binary features), output=3 actions
# Training: 1M steps, epsilon-greedy, experience replay
