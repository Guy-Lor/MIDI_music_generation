# http: // craffel.github.io / pretty - midi /  #

import pretty_midi as pm
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation

# TODO Piano roll for drums!
# TODO Add instruments join for piano roll (same instrument, same piano roll)
# Sampling freq of the columns for piano roll. The higher, the more "timeline" columns we have.
from tensorflow.python.keras.optimizers import RMSprop

SAMPLING_FREQ = 5
WINDOW_SIZE = 100


def get_piano_roll(instrument, end_time):
    # All instruments get the same "times" parameter in order to make the piano roll timeline the same.
    piano_roll = instrument.get_piano_roll(fs=SAMPLING_FREQ, times=np.arange(0, end_time, 1. / SAMPLING_FREQ))
    return piano_roll


def print_instrument_info(i, instrument):
    instrument_notes_len = len(instrument.notes)
    end_time = instrument.get_end_time()
    print(f"Instrument #{i} - {instrument.name} ({instrument.program}) | {instrument_notes_len} notes | {end_time:.2f} seconds")


def get_time_note_dict(piano_roll):
    times = np.unique(np.where(piano_roll > 0)[1])
    index = np.where(piano_roll > 0)
    dict_keys_time = {}

    for time in times:
        index_where = np.where(index[1] == time)
        notes = index[0][index_where]
        dict_keys_time[time] = notes

    return dict_keys_time
    # return list_of_dict_keys_time


def get_RNN_input_target(notes_list):
    # Creates input, target np arrays in the requested window size
    input_windows = np.lib.stride_tricks.sliding_window_view([1] * WINDOW_SIZE + notes_list, WINDOW_SIZE)
    target_windows = np.lib.stride_tricks.sliding_window_view(notes_list, 1)

    return input_windows, target_windows


def midi_preprocess(path, notes_hash, instruments_dependencies=False, print_info=False, separate_midi_file=False):
    instruments_piano_roll = {}
    midi_name = path.split('.', 1)[0].split("/")[-1]
    # Load MIDI file into PrettyMIDI object
    midi_data = pm.PrettyMIDI(path)
    end_time = midi_data.get_end_time()
    if print_info:
        print(f"\n---------- {midi_name} | {end_time:.2f} seconds -----------")

    # Separate tracks and print info
    for i, instrument in enumerate(midi_data.instruments):
        # Fix instruments names
        instrument.name = "Drums" if instrument.is_drum else pm.program_to_instrument_name(instrument.program)

        instruments_piano_roll[i] = get_piano_roll(instrument, end_time)

        if print_info:
            print_instrument_info(i, instrument)
        if separate_midi_file:
            # Write MIDI files for each instrument for debugging
            instrument_midi = pm.PrettyMIDI()
            instrument_midi.instruments.append(instrument)
            instrument_midi.write(f'{midi_name}_{i}_instrument.mid')

    ############################# Reminder from here on we only use 1 instrument !!! #############################################
    dict_keys_time = get_time_note_dict(instruments_piano_roll[0])

    for key in dict_keys_time.keys():
        # print(str(dict_keys_time[key]))
        dict_keys_time[key] = str(dict_keys_time[key])
        notes_hash.add_new_note(dict_keys_time[key])

    # total time of piano roll, not of the midi file in seconds
    total_time = instruments_piano_roll[0].shape[1]
    # print(instruments_piano_roll[0].shape)
    notes_list = []
    for time in range(0, total_time, 1):
        if time not in dict_keys_time:
            notes_list += [notes_hash.notes_dict['e']]
        else:
            current_note = dict_keys_time[time]
            notes_list += [notes_hash.notes_dict[current_note]]

    print(notes_list)
    input_windows, target_windows = get_RNN_input_target(notes_list)
    print(input_windows)
    print(target_windows)
    # print(input_windows[0], target_windows[0])

    return input_windows, target_windows


class ModelTrainer:
    def __init__(self, x, y, epochs, batches):
        self.notes_hash = NotesHash()
        self.epochs = epochs
        self.batches = batches
        self.input_data = x
        self.target_data = y
        self.model = self.create_model()  # Remember to use Cross-batch statefulness


    def create_model(self):
        # create sequential network, because we are passing activations
        # down the network
        model = Sequential()

        # add LSTM layer
        model.add(LSTM(self.batches, input_shape=(WINDOW_SIZE, 1)))

        # add Softmax layer to output one character
        # model.add(Dense(len(chars)))
        # model.add(Activation('softmax'))

        model.add(Dense(50))
        model.add(Dense(50))

        model.add(Dense(1))

        # compile the model and pick the loss and optimizer
        model.compile(loss='mse', optimizer='adam')

        return model

    def train(self):
        # train the model
        self.model.fit(self.input_data, self.target_data, batch_size=self.batches, epochs=self.epochs)


# 10 songs --> instruments (1) --> array of windows -->16 windows per batch
# Remember: Add one-hot encoding to target/input
class NotesHash:
    def __init__(self):
        self.notes_dict = {'e': 1}
        self.reversed_notes_dict = {1: 'e'}
        self.token_counter = 2

    def add_new_note(self, new_note):
        if new_note not in self.notes_dict.keys():
            self.notes_dict[new_note] = self.token_counter
            self.reversed_notes_dict[self.token_counter] = new_note
            self.token_counter += 1


def main():
    path = 'blues/'
    path = 'classic_piano/'
    files = [i for i in os.listdir(path) if i.endswith(".mid")]
    print(files)
    notes_hash = NotesHash()

    input_windows, target_windows = midi_preprocess(path='all_blues-Miles-Davis_dz.mid', notes_hash=notes_hash, instruments_dependencies=False, print_info=True, separate_midi_file=False)
    model = ModelTrainer(x=input_windows, y=target_windows, epochs=1, batches=16)
    model.train()
    # reading each midi file
    # for file in files:
    #     midi_preprocess(path + file, notes_hash=notes_hash, instruments_dependencies=False, print_info=True, separate_midi_file=False)


if __name__ == '__main__':
    main()
