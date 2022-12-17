# CS667-project
This repository will contain our implementation of the CS667 (Introduction to AI) semester project comparing a Markov-chain text model, an LSTM, and an LLM (Large Language Model)

## Setup

We strongly recommend that you start off with a fresh conda environment. All our experiments were carried out on Python 3.8.6.

- After installing conda, create a new environment with a clean Python 3.8.6 image using the following command:

`conda create -n cs667 -y python==3.8`

- Activate the environment:

`conda activate cs667`

- Next, clone this repository:

`git clone https://github.com/rahulkulhalli/CS867.git`


- cd into the repo and install torch:
`
cd CS867`

`conda install pytorch torchvision torchaudio cpuonly -c pytorch -y`


- Finally, run the interactive evaluation script:

`python3 eval_all.py`
