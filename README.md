# RLVR Experiments

This repository contains experiments and frameworks for Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) on reasoning tasks. It is designed to train and evaluate small language models on tasks like Battleship, Mazes, GSM8K, and Syllogisms, focusing on generating reasoning traces (`<think>...</think>`).

## Project Structure

- **`configs/`**: Configuration files (YAML) for different experiments (e.g., `config_battleship.yaml`, `sft_maze.yaml`).
- **`data_gen/`**: Scripts for generating synthetic training data.
  - `generate_sft_data.py`: Generates SFT data with reasoning traces.
- **`environments/`**: Environment logic for the supported tasks.
  - `battleship_logic.py`, `battleship_env.py`: Battleship game logic.
  - `maze_env.py`: Maze navigation task.
  - `gsm8k.py`: Grade School Math 8K task.
  - `syllogism_env.py`: Syllogism reasoning task.
- **`notebooks/`**: Jupyter notebooks for exploration and analysis.
  - `play_battleship.ipynb`: Interactive notebook to play/simulate Battleship with agents.
- **`train_sft.py`**: Script for Supervised Fine-Tuning of models.
- **`train_grpo.py`**: Script for Group Relative Policy Optimization (GRPO) training.
- **`evaluate_sft.py`**: Script to evaluate SFT models on correctness and reasoning format compliance.
- **`evaluate.py`**: General evaluation script.
- **`run_chain.py`**: Helper script for running chained experiments.
- **`verify_battleship.py`**: Script to verify Battleship environment mechanics.

## Installation

Ensure you have the required dependencies installed (likely `transformers`, `torch`, `trl`, `datasets`, `numpy`, `yaml`, etc.).

```bash
pip install -r requirements.txt
# (Note: Create a requirements.txt if one doesn't exist)
```

## Usage

### 1. Data Generation

To generate SFT training data (e.g., for the Maze task):

```bash
python data_gen/generate_sft_data.py configs/data_gen_maze.yaml
```

This will produce a JSONL file (e.g., `sft_data.jsonl`) containing prompts and reasoning-augmented answers.

### 2. Supervised Fine-Tuning (SFT)

To train a model using the generated data:

```bash
python train_sft.py configs/sft_maze.yaml
```

This script loads the model and dataset specified in the config and runs SFT.

### 3. RL / GRPO Training

To run RL training (e.g., GRPO) on an environment:

```bash
python train_grpo.py configs/config_battleship.yaml
```

### 4. Evaluation

To evaluate a trained model:

```bash
python evaluate_sft.py configs/eval.yaml [optional_model_path] [num_samples]
```

## Environments

### Battleship

A grid-based game where the agent must find ships. Returns a tuple `(row, col)`.

- Logic: `environments/battleship_logic.py`
- Notebook: `play_battleship.ipynb`

### Maze

Pathfinding tasks where the agent navigates a maze.

### GSM8K

Math word problems requiring multi-step reasoning.

### Syllogism

Logical deduction tasks.

## Reasoning Format

The models are trained to output reasoning traces in the following format:

```
<think>
[Step-by-step reasoning goes here]
</think>
<answer>
[Final Answer]
</answer>
```

This format allows for extracting and verifying the "thought process" separate from the final answer.
