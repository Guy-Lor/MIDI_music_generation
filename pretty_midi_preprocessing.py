# http: // craffel.github.io / pretty - midi /  #

import pretty_midi as pm
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, ReLU
import matplotlib.pyplot as plt

# TODO Piano roll for drums!
# TODO Add instruments join for piano roll (same instrument, same piano roll)
# Sampling freq of the columns for piano roll. The higher, the more "timeline" columns we have.
# from tensorflow.python.keras.optimizers import RMSprop
from keras.utils import to_categorical

SAMPLING_FREQ = 5
WINDOW_SIZE = 50


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
    input_windows = rolling_window([1] * WINDOW_SIZE + notes_list, WINDOW_SIZE)[:-1]
    target_windows = rolling_window(notes_list, 1)
    return input_windows, target_windows


def rolling_window(a, window):
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


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

    # print(notes_list)
    input_windows, target_windows = get_RNN_input_target(notes_list)
    # print(input_windows)
    # print(target_windows)
    # print(input_windows[0], target_windows[0])

    return notes_list, input_windows, target_windows


class ModelTrainer:
    def __init__(self, x, y, epochs, batches, is_stateful=False, one_hot_encode=True):
        self.notes_hash = NotesHash()
        self.epochs = epochs
        self.batches = batches
        self.is_stateful = is_stateful
        self.one_hot_encode = one_hot_encode
        self.num_of_batches = int(len(x) / batches)
        if self.is_stateful:
            x = x[:self.num_of_batches * batches]
            y = y[:self.num_of_batches * batches]

        self.input_data = np.reshape(x, (x.shape[0], x.shape[1], 1))
        self.target_data = to_categorical(y) if one_hot_encode else y
        self.model = self.create_model()

    def create_model(self):
        # create sequential network, because we are passing activations
        # down the network
        model = Sequential()

        # add LSTM layer
        if self.is_stateful:
            model.add(LSTM(self.batches, input_shape=(WINDOW_SIZE, 1), stateful=self.is_stateful, batch_input_shape=(self.batches, WINDOW_SIZE, 1)))
        else:
            model.add(LSTM(self.batches, input_shape=(WINDOW_SIZE, 1)))

        # add Softmax layer to output one character
        # model.add(Dense(len(chars)))
        # model.add(Activation('softmax'))

        model.add(Dense(100))
        model.add(ReLU())
        model.add(Dense(100))

        # compile the model and pick the loss and optimizer
        if self.one_hot_encode:
            model.add(Dense(self.target_data.shape[1], activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        else:
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='adam')

        return model

    def train(self):
        # train the model
        self.model.fit(self.input_data, self.target_data, batch_size=self.batches, epochs=self.epochs)

    def generate_MIDI(self, initial_sample: list, length):
        length = length - WINDOW_SIZE
        current_window = initial_sample
        current_window_list = list(current_window[0])

        total_song = list(initial_sample[0])
        for i in range(length):
            # print("Iteration: " ,i)
            current_window = np.array([current_window_list])
            current_window = np.reshape(current_window, (current_window.shape[0], current_window.shape[1], 1))
            if self.one_hot_encode:
                y = self.model.predict_classes(current_window)
            else:
                y = self.model.predict(current_window)

            # print("Current window length before: ", len(current_window_list))
            current_window_list += [int(y)]
            current_window_list.pop(0)
            # print("Current window length after: ", len(current_window_list))

            total_song += [int(y)]
        return total_song


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


def draw_compare_graph(real_input, predicted_input, time):
    plt.scatter(range(time), real_input, c='blue', alpha=0.25)
    plt.scatter(range(time), predicted_input, c='red', alpha=0.5)
    plt.show()


def main():
    path = 'blues/'
    path = 'classic_piano/'
    files = [i for i in os.listdir(path) if i.endswith(".mid")]
    print(files)
    notes_hash = NotesHash()

    real_notes_list, input_windows, target_windows = midi_preprocess(path=path + 'alb_esp1.mid', notes_hash=notes_hash,
                                                                     instruments_dependencies=False,
                                                                     print_info=True, separate_midi_file=False)
    model = ModelTrainer(x=input_windows, y=target_windows, epochs=1, batches=100, one_hot_encode=True)
    model.train()
    # length of the real song
    midi_length = len(real_notes_list)
    # length is actually
    pred_notes_list = model.generate_MIDI([input_windows[WINDOW_SIZE]], length=midi_length)
    print(real_notes_list)
    print(pred_notes_list)
    draw_compare_graph(real_input=real_notes_list, predicted_input=pred_notes_list, time=midi_length)

    matches_count = 0
    for real, pred in zip(real_notes_list, pred_notes_list):
        if real == pred:
            matches_count += 1
    print(f"Length of song's notes is {len(real_notes_list)}")
    print(f"There is {matches_count / len(pred_notes_list) * 100:.2f}% match between Real song and Pred of window size")

    # reading each midi file
    # for file in files:
    #     midi_preprocess(path + file, notes_hash=notes_hash, instruments_dependencies=False, print_info=True, separate_midi_file=False)

    # print(len(notes_hash.notes_dict))


if __name__ == '__main__':
    main()
