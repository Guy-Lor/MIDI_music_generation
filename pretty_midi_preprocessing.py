# http: // craffel.github.io / pretty - midi /  #

import pretty_midi as pm
import numpy as np
import os

# Sampling freq of the columns for piano roll. The higher, the more "timeline" columns we have.
SAMPLING_FREQ = 2


def get_piano_roll(instrument, end_time):
    # All instruments get the same "times" parameter in order to make the piano roll timeline the same.
    piano_roll = instrument.get_piano_roll(fs=SAMPLING_FREQ, times=np.arange(0, end_time, 1. / SAMPLING_FREQ))
    return piano_roll


def print_instrument_info(i, instrument):
    instrument_name = pm.program_to_instrument_name(instrument.program)
    instrument_notes_len = len(instrument.notes)
    end_time = instrument.get_end_time()
    print(f"Instrument #{i} - {instrument_name} ({instrument.program}) | {instrument_notes_len} notes | {end_time:.2f} seconds")


def midi_preprocess(path='all_blues-Miles-Davis_dz.mid', instruments_dependencies=False, print_info=False, separate_midi_file=False):

    instruments_piano_roll = {}
    midi_name = path.split('.', 1)[0].split("/")[-1]
    # Load MIDI file into PrettyMIDI object
    midi_data = pm.PrettyMIDI(path)
    end_time = midi_data.get_end_time()

    if print_info:
        print(f"\n---------- {midi_name} | {end_time:.2f} seconds -----------")

    # Print all the instruments WITH is_drum flag
    # print(midi_data.instruments)

    # Separate tracks and gives info
    for i, instrument in enumerate(midi_data.instruments):

        instruments_piano_roll[i] = get_piano_roll(instrument, end_time)

        if print_info:
            print_instrument_info(i, instrument)
        if separate_midi_file:
            # Write MIDI files for each instrument for debugging
            instrument_midi = pm.PrettyMIDI()
            instrument_midi.instruments.append(instrument)
            instrument_midi.write(f'{midi_name}_{i}_instrument.mid')

    # print(instruments_piano_roll)


def main():
    path = 'blues/'
    files = [i for i in os.listdir(path) if i.endswith(".mid")]
    print(files)
    # reading each midi file
    for file in files:
        midi_preprocess(path+file, instruments_dependencies=False, print_info=True, separate_midi_file=False)

if __name__ == '__main__':
    main()
