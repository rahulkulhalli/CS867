# CS667-project
This repository will contain our implementation of the CS667 (Introduction to AI) semester project comparing a Markov-chain text model, an LSTM, and an LLM (Large Language Model)

## Team members
| Team member | SU NetID |
|--|--|
| Rahul Kulhalli | rmkulhal@syr.edu |
| Aishwarya Nadkarni | ainadkar@syr.edu |
| Kruthi N Raj | kraj03@syr.edu |

## Acknowledgements
- [Andrej Karpathy](https://www.github.com/karpathy)'s minGPT implementation: https://github.com/karpathy/minGPT
- Project Gutenberg: https://www.gutenberg.org/

## Setup

We strongly recommend that you start off with a fresh conda environment. All our experiments were carried out on Python 3.8.6.

If you do not have anaconda or miniconda installed on your device, start from **step 1**. Otherwise, skip to **step 2**.

#### [1] Download and configure miniconda, a lightweight version of anaconda
Download the latest release of miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

#### [2] Create a new environment:

`conda create -n cs667 -y python==3.8`

#### [3] Activate the environment:

`conda activate cs667`

#### [4] Clone this repository:

`git clone https://github.com/rahulkulhalli/CS867.git`

#### [5]  cd into the repo and install torch:

`cd CS867`

`conda install pytorch torchvision torchaudio cpuonly -c pytorch -y`

#### 6) Run the interactive evaluation script:

`python3 eval_all.py`

## Running `eval_all.py`
Upon running the `eval_all.py`script, you will be greeted with a welcome message asking you to input the domain of your choice. You will then be asked to type the text prompt. Our models will do everything they can to generate the best possible story trailing your input! Please note - the maximum generation length of the text is an input parameter than can be configured by the user. To override the default character generation length, use the --gen-limit flag. For instance,
`python3 eval_all.py --gen-limit 5000`
