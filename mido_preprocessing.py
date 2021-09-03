# https: // mido.readthedocs.io / en / latest / intro.html
# Might have a problem with processing with the exact instrument (program change messages)

import mido

# from mido import Message, MidiFile, MidiTrack
# Check msg format
msg = mido.Message('note_on', note=60)
print(msg.type, msg.note)

# import midi file
mid = mido.MidiFile('blues/all_blues-Miles-Davis_dz.mid')
# print(mid.tracks)
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)

# creating new MIDI file
mid = mido.MidiFile()
track = mido.MidiTrack()

track.append(mido.Message('program_change', program=39, time=0))
track.append(mido.Message('note_on', note=64, velocity=127, time=32))
track.append(mido.Message('note_off', note=64, velocity=0, time=500))
track.append(mido.Message('program_change', program=10, time=0))
track.append(mido.Message('note_on', note=64, velocity=127, time=500))
track.append(mido.Message('note_off', note=64, velocity=0, time=1000))

mid.tracks.append(track)

mid.save('new_song.mid')

# Taking the "note_on" messages of each track
