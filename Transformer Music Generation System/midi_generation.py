#-*- coding: utf-8 -*-

from __future__ import division
import pretty_midi
import numpy as np


def load_data(path):
    en = []
    with open(path, 'r') as f:
        for line in f:   #一行行读，每行小写，每个加进去en里
            line = line.strip().split('\t')
            en=en+line
        #print(en)
    return en

def build_dict(sentences):
    pitch=[]
    for sentence in sentences:
        #print("sentence:"+sentence)
        for s in sentence.split(' '):
            pitch.append(s)
            #print("s:"+ s)
    print(pitch)
    return pitch

def music_generate(pitch_list,duration_list,midi_name, melody_instrument):

    # 此部分为写进旋律轨
    midi_generated = pretty_midi.PrettyMIDI()
    instrument_chosen = pretty_midi.instrument_name_to_program(melody_instrument)
    instrument = pretty_midi.Instrument(program=instrument_chosen)
    notes_pitch = pitch_list
    duration = duration_list
    #duration = 4/np.array(duration)

    start_time_i = 0

    for i in range(len(notes_pitch)):
        #print(type(notes_pitch[i]))
        pitch_i = int(float(notes_pitch[i]))
        duration_i = int(float(duration[i]))/4
        start_time_i += duration_i
        start_time = start_time_i - duration_i
        end_time_i = start_time_i
        note_i = pretty_midi.Note(velocity=127, pitch=pitch_i, start=start_time, end=end_time_i)
        instrument.notes.append(note_i)

    midi_generated.instruments.append(instrument)
    midi_generated.write(midi_name + ".mid")


if __name__ == '__main__':
    pi = load_data('G1_base_note.txt')
    pit = build_dict(pi)
    du = load_data('G1_base_duration.txt')
    dur = build_dict(du)
    midi = music_generate(pit,dur,"G1_Baseline",'Acoustic Grand Piano')








