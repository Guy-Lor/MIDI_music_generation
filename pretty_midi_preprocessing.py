# http: // craffel.github.io / pretty - midi /  #

import pretty_midi
import numpy as np

# Sampling freq of the columns for piano roll
SAMPLING_FREQ = 30

# Load MIDI file into PrettyMIDI object
midi_data = pretty_midi.PrettyMIDI('all_blues-Miles-Davis_dz.mid')
# Print all the instruments WITH is_drum flag
print(midi_data.instruments)
print(len(midi_data.instruments[5].notes))

piano_roll = midi_data.instruments[0].get_piano_roll(fs=SAMPLING_FREQ, times=np.arange(0, midi_data.get_end_time(), 1. / SAMPLING_FREQ))
print(piano_roll.shape)
print(midi_data.get_end_time())

# print(dict(piano_roll))


# Separate tracks and gives info

for i, instrument in enumerate(midi_data.instruments):
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    instrument_notes_len = len(instrument.notes)
    print(f"Instrument #{i} - {instrument_name} ({instrument_notes_len} notes)")
    piano_roll = instrument.get_piano_roll(fs=SAMPLING_FREQ, times=np.arange(0, midi_data.get_end_time(), 1. / SAMPLING_FREQ))
    print(piano_roll.shape)
    instrument_midi = pretty_midi.PrettyMIDI()
    print(instrument.get_end_time())

    # Creates MIDI files for each instrument

    instrument_midi = pretty_midi.PrettyMIDI()
    instrument_midi.instruments.append(instrument)
    instrument_midi.write(f'{i}_instrument.mid')


