import argparse
from src.utils.misc_utils import Domain, Model
from pathlib import Path
from eval_markov import get_markov_ouput
from eval_lstm import get_lstm_output
from eval_chargpt import get_gpt_output


def parse_domain():
    domain_inp = input("Select the domain (Metamorphosis [m], Crime and Punishment [c], Dracula [d], Frankenstein [f], Little Women [l]): ")
    if domain_inp != "" and domain_inp.lower() in ["m", "c", "d", "f", "l"]:
        domain = domain_inp
    else:
        # let's default to CrimeAndPunishment.
        domain = "c"

    # Convert the string to ENUM for ease.
    domain_enum = Domain.from_str(domain)

    domain_mapper = {
        Model.MARKOV: {
            Domain.Metamorphosis: Path("data/kafka.txt"),
            Domain.CrimeAndPunishment: Path("data/c_and_p.txt"),
            Domain.Dracula: Path("data/dracula.txt"),
            Domain.Frankenstein: Path("data/frankenstein.txt"),
            Domain.LittleWomen: Path("data/little_women.txt")
        },
        Model.LSTM: {
            Domain.Metamorphosis:  Path("models/lstm_weights/lstm_final_CrimeAndPunishment.pt"),
            Domain.CrimeAndPunishment: Path("models/lstm_weights/lstm_final_Dracula.pt"),
            Domain.Dracula:  Path("models/lstm_weights/lstm_final_Kafka.pt"),
            Domain.Frankenstein:  Path("models/lstm_weights/lstm_final_Frankenstein.pt"),
            Domain.LittleWomen: Path("models/lstm_weights/lstm_final_LittleWomen.pt")
        },
        Model.GPT: {
            Domain.Metamorphosis:  Path("models/gpt_weights/model_iter8000_CrimeAndPunishment.pt"),
            Domain.CrimeAndPunishment: Path("models/gpt_weights/model_iter34000_Dracula.pt"),
            Domain.Dracula:  Path("models/gpt_weights/model_iter9000_Kafka.pt"),
            Domain.Frankenstein:  Path("models/gpt_weights/minGPT_23000_Frankenstein.pt"),
            Domain.LittleWomen: Path("models/gpt_weights/model_iter32000_LittleWomen.pt")
        }
    }

    return {d: domain_mapper[d][domain_enum] for d in domain_mapper.keys()}


def print_sep():
    print()
    print(50*'*')
    print()


if __name__ == "__main__":
    print(150*'=')
    print("Hello, welcome to our CIS667 project implementation! Below, you may choose one of the five playground domains and create your own story!")
    print(150*'=')

    while True:
        domain_weights = parse_domain()
        print()
        user_input = input("Start writing from here :: ")

        # Load and call Markov.
        markov_output = get_markov_ouput(
            data_path=domain_weights[Model.MARKOV],
            seed_string=user_input
        )

        print_sep()

        print("Generated Markov Output ::", end='\n\n')
        print(markov_output)

        print_sep()

        lstm_output = get_lstm_output(
            model_weights_path=domain_weights[Model.LSTM],
            seed_string=user_input
        )

        print("Generated LSTM Output ::", end='\n\n')
        print(lstm_output)

        print_sep()

        gpt_output = get_gpt_output(
            model_weights_path=domain_weights[Model.GPT],
            seed_string=user_input
        )

        print("Generated minGPT Output ::", end='\n\n')
        print(gpt_output)

        print_sep()

        break_prompt = input("Continue? [y/n] :: ")
        if break_prompt.lower() not in ["y", "n"]:
            print("Unknown input. Breaking sequence...")
            break

        if break_prompt.lower() == "n":
            print("Breaking sequence. Thank you for trying our project!")
            break
