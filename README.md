# CS667-project
This repository will contain our implementation of the CS667 (Introduction to AI) semester project comparing a Markov-chain text model, an LSTM, and an LLM (Large Language Model)


## Choice of domain
We will be working in the character-level domain instead of the word-level due to the following two main reasons:
1. Dearth of training data - LLMs are trained on billions of lines of text scraped from all over the internet. Since this is a mini project, we limit our scope to books obtained from [Project Gutenberg](https://www.gutenberg.org/). To maximize our training data, using a character language model is a much better choice.
2. Induce the possibility of cycles - Word language models seldom sample the same word one after the other. This is because there are very few instances of word repitition as compared to character repititon.
3. Limit the size of the vocabulary - A word level language model needs to represent all of its vocabulary in memory while training. Since the number of posisble character combinations is near limitless, the size of the word vocabulary grows exponentially as a function of the corpus size. Instead, if we use character level models, we can assert the maximum size of the vocabulary to be the length of the unique characters in the text. Even after including special characters, this vocabulary seldom exceeds a length of 100.

# Choice of Models
## Baseline - Markov chain Model
The Markov chain model first generates a probability transition table (Ptable) using the available domain data. The preprocessing steps are as follows:
- Remove punctuations and other unnecessary parts from the data corpus
- Count the number of word pairs, i.e., count the number of times a certain word follows another certain word
- Normalize these counts into probabilities

### Evaluation
To evaluate the model, invoke `eval_markov.py`

## Middle ground - LSTM
Long Short-Term Memory Networks significantly outpeform regular bigram models because they model the probability of the current word based on a context window. P(word[i]) = argmax P(word[i] | word[i-1], word[i-2], ... , word[i-k]) where k is the sequence length (sometimes also called the window size.) We train a relatively simple LSTM (no bidirectional connections, no stacked LSTM cells) for 6-15 epochs per dataset depending upon how well the model generalizes to the validation data.

### Training
To train the LSTM model, invoke `train_lstm.py`. Be sure to modify the parameters defined under the main function to suit your requirements. 

### Evaluation
To evaluate the model, invoke `eval_lstm.py`

## SOTA - MinGPT
The minGPT model is directly borrowed from [Andrej Karpathy's ](https://github.com/karpathy) implementation of the GPT model. We choose to train on the minGPT variant (which limits the number of Transformer block layers and reduces the sequence length) and train every dataset for around 10k - 25k iterations depending on how well the model performs on the validation data.

### Training
To train the minGPT model, invoke `train_chargpt.py`. Be sure to modify the parameters to suit your requirements.

### Evaluation
To evaluate the model, invoke `eval_chargpt.py`.
