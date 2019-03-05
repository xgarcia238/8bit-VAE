import glob
import pickle
import numpy as np
import os

import torch
import checkpoint
from checkpoint import Checkpoint
from torch.utils.data import Dataset
from collections import defaultdict

##(key,val) = (voice, (note_size,velocity_size, timbre_size))
VOICES = {"P1": (77,16,4),
                   "P2": (77,16,4),
                   "TR": (89,0,0),
                   "NO": (17,16,2)}


def dataset_builder(input_folder, output_folder, score_type,threshold, note_size,
                    num_steps, window_size, min_length = 10, TR=False):
    """
    ----------------------------------------------------------------------------
    Args:
        folder     (string): name of folder containing songs.
        score_type (string): "exprsco" for expressive, "seprsco" for separated.
        num_steps     (int): number of steps for each roll.
        window_size   (int): number of steps to move piano roll.
        min_length    (int): smallest length of songs (in seconds).
        TR           (bool): If True, use TR notes as well.

    Returns:
        songs: List of songs.
    ----------------------------------------------------------------------------
    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    samps_per_sec = 24000 #24 kHz
    idx_to_name = []
    voice_names = ["P1","P2","TR"]
    big_data = []
    big_lengths = []
    max_len = 32
    voices = 3 if TR else 2
    song_idx = defaultdict(list)
    f = np.vectorize(lambda x,shift : x + shift if x > 0 else 0)
    postproc = postprocessing_with_TR if TR else postprocessing

    for song in glob.glob(input_folder + '/*'):

        with open(song, 'rb') as song_info:
            rate, nsamps, score = pickle.load(song_info)
            song_duration       = nsamps//samps_per_sec
            if song_duration < min_length:
                continue
            if not TR:
                score[:,2] = 0
            score[:,3] = 0  #Keep only pulse.

            if score[score != 0].shape[0] == 0: #Check that there's actual music.        part_id = 0
                continue

            P_score = score[:,:2]
            TR_score = score[:,2]
            try:
                min_note_P = np.min(P_score[P_score != 0])
                min_note_TR = np.min(TR_score[TR_score != 0]) if TR else 108
            except ValueError:
                continue
            min_shift = min(min_note_P - 33, min_note_TR - 21)
            max_note = np.max(score[:voices])

            #Prepare shifts. We don't want to tranpose by more than 5 notes.
            max_shift = min(108-max_note,5)
            min_shift = -min(min_shift,4)
            song_step_count = score.shape[0]
            for shift in range(min_shift, max_shift+1):
                f_score = song_normalizer(f(score,shift), score_type, False, None)
                f_score = f_score.astype(np.uint16)[:,:voices]
                f_score, valid = compactify_score(f_score, note_size, max_len,True,TR)
                if not valid:
                    break

                #Checking we didn't mess up processing.
                assert np.array_equal(postproc([f_score], threshold), f(score, shift))
                left, right = 0, num_steps

                while right < f_score.shape[0]:
                    data   = torch.from_numpy(f_score[left:right]).type(torch.int16).unsqueeze(0)
                    left  += window_size
                    right += window_size
                    if shift == 0:
                        song_idx[song].append(len(big_data))
                    big_data.append(data)

    big_data = torch.cat(big_data,0)
    print("Shape of data: ", big_data.shape)
    print("Max value: ", torch.max(big_data))
    torch.save(big_data, os.path.join(output_folder, "data"))
    use_TR = "_TR" if TR else ""
    with open(f'{num_steps}_{window_size}{use_TR}_song_dict','wb') as file:
        pickle.dump(song_idx,file)
    return big_data.shape

def song_normalizer(score, score_type, reverse = False, threshold = None):
    """
    ----------------------------------------------------------------------------
    Takes a score and normalizes the score so that there are no gaps i.e. notes
    start at 0 and increase by one. We include an additional parameter to ignore
    notes below a certain threshold.

    Args:
        score (np.array): shape determined by score_type.
        score_type (str): either "exprsco" or "seprsco".
        reverse   (bool): If True, it denormalizes the song.
        threshold  (int): Notes below this value will be turned off.

    Returns:
        normalized_song:
    ----------------------------------------------------------------------------
    """
    dir = 1 if reverse else -1
    pulse_min = threshold if threshold else 32
    TR_min = threshold if threshold else 20
    P1_normalizer = np.vectorize(lambda x : max(x + dir*pulse_min,0) if x > 0 else 0)
    P2_normalizer = np.vectorize(lambda x : max(x + dir*pulse_min,0) if x > 0 else 0)
    TR_normalizer = np.vectorize(lambda x : max(x + dir*TR_min,0) if x > 0 else 0)

    if score_type == "exprsco":
        #Normalize the notes for the voices.
        score[:,0,0] = P1_normalizer(score[:,0,0])
        score[:,1,0] = P2_normalizer(score[:,1,0])
        score[:,2,0] = TR_normalizer(score[:,2,0])

    elif score_type == "seprsco":
        score[:,0] = P1_normalizer(score[:,0])
        score[:,1] = P2_normalizer(score[:,1])
        score[:,2] = TR_normalizer(score[:,2])

    else:
        raise Exception('Invalid score type.')

    return score


def postprocessing(data, threshold, separate=True):
    """
    --------------------------------------------------------------------------
    This takes the output data of our model turns into NES playable form. For
    now assume no NO notes.
    Args:
        data: List of tuples of size (batch_size, timestep, range)
        threshold: Minimum positive note value.
    Returns:
        result (np.array): Shape (batch_size,time,4)
    --------------------------------------------------------------------------
    """
    P1_normalizer = np.vectorize(lambda x : x + threshold if x > 0 else 0)
    P2_normalizer = np.vectorize(lambda x : x + threshold if x > 0 else 0)
    #TR_normalizer = np.vectorize(lambda x : x + threshold if x > 0 else 0)
    #NO_normalizer = np.vectorize(lambda x : x)

    normalizers = [P1_normalizer,P2_normalizer]

    res = []
    note_size = 108-threshold+1
    offset = 0
    res = []
    P2_offset = note_size if separate else 0
    cnt_offset = 2*note_size - 1 if separate else 0
    for song in data:
        notes = []
        for i,(P1, P2, cnt) in enumerate(zip(song[::3], song[1::3], song[2::3])):
            cur_notes = [[P1_normalizer(P1),P2_normalizer(P2-P2_offset),0,0]]
            #assert cnt + 1 - 2*note_size > 0, (i, cnt, song)
            assert  cnt - cnt_offset > 0, (cnt, cnt_offset)
            cur_notes *= int(cnt -cnt_offset)
            notes.append(np.array(cur_notes))
        res.append(np.vstack(notes))
    return np.vstack(res)

def postprocessing_with_TR(data, threshold, separate=True):
    """
    --------------------------------------------------------------------------
    This takes the output data of our model turns into NES playable form. For
    now assume no NO notes.
    Args:
        data: List of tuples of size (batch_size, timestep, range)
        threshold: Minimum positive note value.
    Returns:
        result (np.array): Shape (batch_size,time,4)
    --------------------------------------------------------------------------
    """
    P1_normalizer = np.vectorize(lambda x : x + 32 if x > 0 else 0)
    P2_normalizer = np.vectorize(lambda x : x + 32 if x > 0 else 0)
    TR_normalizer = np.vectorize(lambda x : x + 20 if x > 0 else 0)
    #NO_normalizer = np.vectorize(lambda x : x)

    normalizers = [P1_normalizer,P2_normalizer]

    res = []
    note_size = 108-32+1
    offset = 0
    res = []
    P2_offset = note_size if separate else 0
    TR_offset = 2*note_size if separate else 0
    cnt_offset = 2*note_size - 1 + (note_size + 12) if separate else 0
    for song in data:
        notes = []
        for i,(P1, P2,TR, cnt) in enumerate(zip(song[::4], song[1::4], song[2::4], song[3::4])):
            cur_notes = [[P1_normalizer(P1),
                          P2_normalizer(P2-P2_offset),
                          TR_normalizer(TR-TR_offset),
                          0]]
            #assert cnt + 1 - 2*note_size > 0, (i, cnt, song)
            assert cnt - cnt_offset > 0, (cnt, cnt_offset)
            cur_notes *= int(cnt - cnt_offset)
            notes.append(np.array(cur_notes))
        res.append(np.vstack(notes))
    return np.vstack(res)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def compactify_score(score, note_size, max_len, separate=True, TR=False):
    """
    ----------------------------------------------------------------------------
    This turns a piano_roll for a single voice into a sequence of events,
    similar to Magenta's event decomposition. We count how many steps a note is
    held, then turn into two events. The first event consists of a onehot version
    of the note, and the other consists of onehot version of the number of steps
    into the future.

    Args:
        score (np.array): Normalized score.
        note_size (int): Number of notes per instrument, including off.
        max_len (int): Maximum length allowed.
        separate (bool): If True, separate P1 notes, P2 notes and time shift by
        adding note_size.

    Returns:
        events (np.array): An array containing appropiate note values and counts,
         with an offset by timesteps. Shape =(3, 2*timesteps+1)
        lengths (np.array): An array containing lengths. Shape = (3,)
    ----------------------------------------------------------------------------
    """
    events = []
    voices = ["P1","P2","TR","NO"]
    timesteps = score.shape[0]
    voices = 3 if TR else 2
    lengths = []
    score_length = score.shape[0]
    last_notes = score[0]
    max_count = 0
    count = 0

    #Compute offsets.
    offset = 0
    P2_offset = note_size
    TR_offset = 2*note_size
    cnt_offset = 2*note_size - 1 + (note_size + 12)
    cnt_offset = cnt_offset if TR else 2*note_size -1
    voice_offsets = [0,P2_offset, TR_offset]


    events = [last_notes[i] + offset for i, offset in enumerate(voice_offsets[:voices])]
    for notes in score:
        #If same note, increase count.
        if np.array_equal(notes,last_notes):
            count += 1
            #cur_count += 1
            continue

        events.append(cnt_offset + count + offset)
        max_count = max(max_count,count)
        count = 1

        events += [offset + notes[i] for i,offset in enumerate(voice_offsets[:voices])]
        last_notes = notes

        if max_count > max_len:
            return (0,False)
    max_count = max(count, max_count)
    if max_count > max_len:
        return (0,False)

    events.append(cnt_offset + count + offset)
    events = np.asarray(events).astype(np.int16)

    return (events, True)
"""
def compactify_score(score, note_size, max_len, separate=True):

    ----------------------------------------------------------------------------
    This turns a piano_roll for a single voice into a sequence of events,
    similar to Magenta's event decomposition. We count how many steps a note is
    held, then turn into two events. The first event consists of a onehot version
    of the note, and the other consists of onehot version of the number of steps
    into the future.

    Args:
        score (np.array): Normalized score.
        note_size (int): Number of notes per instrument, including off.
        max_len (int): Maximum length allowed.
        separate (bool): If True, separate P1 notes, P2 notes and time shift by
        adding note_size.

    Returns:
        events (np.array): An array containing appropiate note values and counts,
         with an offset by timesteps. Shape =(3, 2*timesteps+1)
        lengths (np.array): An array containing lengths. Shape = (3,)
    ----------------------------------------------------------------------------

    events = []
    voices = ["P1","P2","TR","NO"]
    timesteps = score.shape[0]
    offset = 0
    P2_offset = note_size if separate else 0
    cnt_offset = 2*note_size - 1 if separate else 0
    lengths = []
    score_length = score.shape[0]

    last_notes = score[0]
    max_count = 0
    count = 0
    events = [last_notes[0], last_notes[1] + P2_offset]
    for notes in score:
        #If same note, increase count.
        if np.array_equal(notes,last_notes):
            count += 1
            #cur_count += 1
            continue

        events.append(cnt_offset + count + offset)
        max_count = max(max_count,count)
        count = 1
        events.append(notes[0])
        events.append(P2_offset + notes[1])
        if notes[0] != last_notes[0] and notes[1] != last_notes[1]:
            last_notes[1] = notes[1]
            last_notes[0] = notes[0]

        elif notes[0] != last_notes[0]:
            last_notes[0] = notes[0]

        elif notes[1] != last_notes[1]:
            last_notes[1] = notes[1]

        if max_count > max_len:
            return (0,False)
    max_count = max(count, max_count)
    if max_count > max_len:
        return (0,False)

    events.append(cnt_offset + count + offset)
    events = np.asarray(events).astype(np.uint8)

    return (events, True)
"""
