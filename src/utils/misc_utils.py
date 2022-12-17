import enum


class Model(enum.Enum):
    MARKOV = 0
    LSTM = 1
    GPT = 2

class Domain(enum.Enum):
    CrimeAndPunishment = 0
    LittleWomen = 1
    Dracula = 2
    Frankenstein = 3
    Metamorphosis = 4

    @staticmethod
    def from_str(label):
        if label == 'd':
            return Domain.Dracula
        elif label == 'c':
            return Domain.CrimeAndPunishment
        elif label == 'l':
            return Domain.LittleWomen
        elif label == 'f':
            return Domain.Frankenstein
        elif label == 'm':
            return Domain.Metamorphosis
        else:
            raise NotImplementedError
