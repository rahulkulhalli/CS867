from pathlib import Path
import torch
import matplotlib.pyplot as plt


def generate_gpt_plots(model_path, save_dir = Path("outputs/")):
    with open(model_path, 'rb') as f:
        ckpt_config = torch.load(f, map_location='cpu')

    train_history = ckpt_config['train_history']
    print(f"{model_path.name} has train history of size {len(train_history)}.")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    ax.plot(train_history, 'b-')
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("CE Loss")
    ax.legend(["Training loss"])
    ax.set_title(f"minGPT model: {model_path.name}")

    plt.savefig(save_dir / model_path.name.replace(".pt", ".png"))
    # plt.show()


def generate_lstm_plots(model_path, save_dir = Path("outputs/")):
    with open(model_path, 'rb') as f:
        ckpt_config = torch.load(f, map_location='cpu')

    train_history = ckpt_config['train_history']
    val_history = ckpt_config['val_history']

    print(f"{model_path.name} has train history of size {len(train_history)} and val history of size {len(val_history)}.")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax[0].plot(train_history, 'b-')
    ax[0].set_xlabel("# Iterations")
    ax[0].set_ylabel("CE Loss")
    ax[0].legend(["Training loss"])

    ax[1].plot(val_history, 'r-')
    ax[1].set_xlabel("# Iterations")
    ax[1].set_ylabel("CE Loss")
    ax[1].legend(["Validation loss"])

    plt.suptitle(f"LSTM model: {model_path.name}")
    # ax[1].set_title(f"minGPT model: {model_path.name}")

    plt.savefig(save_dir / model_path.name.replace(".pt", ".png"))
    # plt.show()


if __name__ == "__main__":
    for path in [
        Path('models/gpt_weights/model_iter8000_CrimeAndPunishment.pt'),
        Path("models/gpt_weights/model_iter34000_Dracula.pt"),
        Path("models/gpt_weights/model_iter9000_Kafka.pt"),
        Path("models/gpt_weights/minGPT_23000_Frankenstein.pt"),
        Path("models/gpt_weights/model_iter32000_LittleWomen.pt")
    ]:
        generate_gpt_plots(path)

    for path in [
        Path("models/lstm_weights/lstm_final_CrimeAndPunishment.pt"),
        Path("models/lstm_weights/lstm_final_Dracula.pt"),
        Path("models/lstm_weights/lstm_final_Kafka.pt"),
        Path("models/lstm_weights/lstm_final_Frankenstein.pt"),
        Path("models/lstm_weights/lstm_final_LittleWomen.pt")
    ]:
        generate_lstm_plots(path)
