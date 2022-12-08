# CS667-project
This repository will contain our implementation of the CS667 (Introduction to AI) semester project comparing a Markov-chain text model and an actual LLM (Large Language Model)

## Markov chain Model
The Markov chain model first generates a probability transition table (Ptable) using the available domain data. The preprocessing steps are as follows:
- Remove punctuations and other unnecessary parts from the data corpus
- Count the number of word pairs, i.e., count the number of times a certain word follows another certain word
- Normalize these counts into probabilities
