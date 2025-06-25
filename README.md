# RL Explainer

An interactive chat interface that uses Google's Gemini AI to analyze and explain the strategies of reinforcement learning agents playing Atari games.

## Features

- Analyze RL agent gameplay videos using Gemini 2.0 Flash
- Interactive chat interface to ask questions about agent behavior
- Support for multiple Atari games (Breakout, Space Invaders)
- Explainable AI insights into RL agent strategies

## Prerequisites

- Python 3.13
- Poetry (for dependency management)
- Google Gemini API key

## Setup Instructions

### 1. Create a Virtual Environment

```bash
# Create virtual environment with Python 3.13
python3.13 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Poetry

```bash
# Install Poetry
pip install poetry
```

### 3. Install Dependencies

With your virtual environment activated, install the project dependencies:

```bash
poetry install
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
touch .env
```

Add your Gemini API key to the `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application

With your virtual environment activated and the `.env` file configured, run the application:

```bash
cd src
python -m rl_explainer.gemini
```

## Usage

1. When you run the application, you'll be prompted to choose a game to analyze:
   - Option 1: Breakout
   - Option 2: Space Invaders

2. The application will generate an initial analysis of the RL agent's strategy using Gemini AI.

3. You can then ask follow-up questions about the agent's behavior in an interactive chat interface.

4. Type your questions and press Enter to get responses from Gemini.

5. To exit the chat, you can use Ctrl+C.
