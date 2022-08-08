
import urllib.request
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

import pretty_midi  # to install with conda or pip

DATA_DIR = Path() / 'midi_bach'
DATA_DIR.mkdir(exist_ok=True)

MIDI_FLES = [
    'cs1-2all.mid', 'cs5-1pre.mid', 'cs4-1pre.mid', 'cs3-5bou.mid',
    'cs1-4sar.mid', 'cs2-5men.mid', 'cs3-3cou.mid', 'cs2-3cou.mid',
    'cs1-6gig.mid', 'cs6-4sar.mid', 'cs4-5bou.mid', 'cs4-3cou.mid',
    'cs5-3cou.mid', 'cs6-5gav.mid', 'cs6-6gig.mid', 'cs6-2all.mid',
    'cs2-1pre.mid', 'cs3-1pre.mid', 'cs3-6gig.mid', 'cs2-6gig.mid',
    'cs2-4sar.mid', 'cs3-4sar.mid', 'cs1-5men.mid', 'cs1-3cou.mid',
    'cs6-1pre.mid', 'cs2-2all.mid', 'cs3-2all.mid', 'cs1-1pre.mid',
    'cs5-2all.mid', 'cs4-2all.mid', 'cs5-5gav.mid', 'cs4-6gig.mid',
    'cs5-6gig.mid', 'cs5-4sar.mid', 'cs4-4sar.mid', 'cs6-3cou.mid'
]

N_NOTES = 79


def load_data():

    n_files = len(MIDI_FLES)
    for i, midiFile in enumerate(MIDI_FLES):
        print(f"Loading MIDI file {i:2} / {n_files}\r", end='', flush=True)
        f_path = DATA_DIR / midiFile
        if not f_path.exists():
            urllib.request.urlretrieve(
                "http://www.jsbach.net/midi/" + midiFile, DATA_DIR / midiFile
            )

    list_midi_files = list(DATA_DIR.glob("*.mid"))
    n_files = len(list_midi_files)
    print(f"Loaded {n_files} MIDI files in folder {DATA_DIR}")

    max_T_x = 1000  # maximum number of notes to consider in each file

    X_list = []

    for midi_file in list_midi_files:
        # read the MIDI file
        midi_data = pretty_midi.PrettyMIDI(str(midi_file))
        notes_idx = [note.pitch for note in midi_data.instruments[0].notes]
        # convert to one-hot-encoding
        notes_idx = np.array(notes_idx[:max_T_x])
        T_x = len(notes_idx)
        X_ohe = np.zeros((T_x, N_NOTES))
        X_ohe[np.arange(T_x), notes_idx - 1] = 1
        # add to the list
        X_list.append(X_ohe)

    print(f"We have {len(X_list)} MIDI tracks")

    return X_list


class BachSynth(torch.nn.Module):

    def __init__(self):

        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=N_NOTES, hidden_size=256, batch_first=True
        )
        self.lin_1 = torch.nn.Linear(256, 256)
        self.dropout = torch.nn.Dropout(0.3)
        self.lin_2 = torch.nn.Linear(256, N_NOTES)

    def forward(self, X):

        h_x, c_x = torch.zeros((2, 1, X.shape[0], 256), device=X.device)
        out, h_x = self.lstm(X, (h_x, c_x))
        out = self.lin_1(out)
        out = self.dropout(out)
        return self.lin_2(out)


def generate(probe, n_times, greedy=True):
    generated = torch.tensor(probe).float()
    if torch.cuda.is_available():
        generated.cuda()
    for _ in range(n_times):
        with torch.no_grad():
            p_note = torch.nn.functional.softmax(
                model(generated[None]), dim=-1
            )[0, -1]

            if greedy:
                idx_note = p_note.argmax()
            else:
                idx_note = np.random.choice(
                    N_NOTES, p=p_note.detach().cpu().numpy()
                )
            note = torch.zeros(N_NOTES, device=generated.device)
            note[idx_note] = 1.

            generated = torch.concat([generated, note[None]], dim=0)

    return generated


if __name__ == "__main__":

    X_list = load_data()

    # create the model
    model = BachSynth()

    # Convert the sequences in torch
    X_list = [torch.from_numpy(x).float() for x in X_list]

    if not torch.cuda.is_available():
        print("Training without GPU...")
        n_epochs = 50  # the bigger the better !

    else:
        print("Training with GPU...")
        n_epochs = 100  # the bigger the better !

        # move data and model to GPU.
        X_list = [x.cuda() for x in X_list]
        model.cuda()

    # Keep one sequence for test
    X_test = X_list.pop(-1)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())

    pobj = []
    for e in range(n_epochs):
        loss_e = 0
        for i, x in enumerate(X_list):
            opt.zero_grad()
            y_pred = model(x[:-1][None])[0]
            loss = criterion(y_pred, x[1:])
            loss.backward()
            opt.step()
            loss_e = loss.item()
            if i % 5 == 0:
                with torch.no_grad():
                    y_pred = model(X_test[:-1][None])[0]
                    loss = criterion(y_pred, X_test[1:])
                print(f"Epoch {e} - Iteration {i} - Loss = {loss_e}"
                      f" ({loss.item()})\r",
                      end='', flush=True)
        pobj.append(loss_e)

    plt.figure()
    plt.plot(pobj)
    plt.savefig("training.pdf")

    for greedy in [True, False]:
        g = generate(X_test[:20], 128, greedy=greedy).detach().cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.imshow(g.T[:, :100], aspect='auto')
        plt.set_cmap('gray_r')
        plt.grid(True)
        plt.savefig(f"generate{'' if greedy else '_rand'}.pdf")
