# Connect 4

This small online app lets you play against a Connect 4 agent that I created for Kaggle's [ConnectX](https://www.kaggle.com/competitions/connectx/overview) competition.


## Summary

The main idea for this project was to create an agent capable of playing connect4 using Minimax algorithm with alpha-beta pruning. I later created this app with a slightly less potent but faster-running version of my agent for demonstration purposes. The game allows you to play against it and test your skills.

## Features

- Play against an AI agent in a Connect 4 game.
- Interactive and dynamic game interface using Streamlit.
- AI uses Minimax algorithm with limited depth for an enjoyable experience.

## Technical Details

- **Streamlit**: The entire UI is built using Streamlit, making it easy to create and manage the interactive components.
- **Minimax Algorithm**: The AI uses the Minimax algorithm with alpha-beta pruning to decide its moves.
- **Heuristic Evaluation**: A simplified heuristic is used for the AI's decision-making process.

## Dependencies

- streamlit
- numpy
- random
- math

## Usage

The game is accessible here: [Connect 4](https://connect4.streamlit.app/)

have fun!
