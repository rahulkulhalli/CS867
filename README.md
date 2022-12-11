# CS667-project
This repository will contain our implementation of the CS667 (Introduction to AI) semester project comparing a Markov-chain text model, an LSTM, and an LLM (Large Language Model). We will be working in the character-level domain instead of the word-level due to the following two main reasons:
1. Dearth of training data - Nowadays, LLMs are trained on billions of lines of text scraped from all over the internet. Since this is a mini project, we limit our scope to books obtained from Project Gutenberg. To maximize our training data, using a character language model is a much better choice.
2. Induce the possibility of cycles - Word language models seldom sample the same word one after the other. This is because there are very few instances of word repitition as compared to character repititon.
3. Limit the size of the vocabulary - A word level language model needs to represent all of its vocabulary in memory while training. Since the number of posisble character combinations is near limitless, the size of the word vocabulary grows exponentially as a function of the corpus size. Instead, if we use character level models, we can assert the maximum size of the vocabulary to be the length of the unique characters in the text. Even after including special characters, this vocabulary seldom exceeds a length of 100.

## Markov chain Model
The Markov chain model first generates a probability transition table (Ptable) using the available domain data. The preprocessing steps are as follows:
- Remove punctuations and other unnecessary parts from the data corpus
- Count the number of word pairs, i.e., count the number of times a certain word follows another certain word
- Normalize these counts into probabilities
