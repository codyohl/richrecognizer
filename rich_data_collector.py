from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave
import datetime
import threading 
import random

# thresholds defined as volume to consider between words.
different_thresholds = [1200, 1500, 1700, 1900, 2200]
volume_threshold = 1500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
SILENCE_LENGTH = 1 # number of consecutive 'silent' audio chunks before we decide to cut the word.
WAIT_FOR_SND = 3 # number of samples we should 'keep' after the silence is detected (keeps a dying off voice)

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < volume_threshold

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>volume_threshold:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record(stream):
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """

    num_silent = 0
    snd_started = False

    r = array('h')

    wait_for_snd = 0
    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)
        if silent and wait_for_snd:
            wait_for_snd -= 1
        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True
            wait_for_snd = WAIT_FOR_SND

        if snd_started and num_silent > SILENCE_LENGTH:
            break

    sample_width = p.get_sample_size(FORMAT)

    # format the data to be normalized and trimmed.
    r = normalize(r)
    r = trim(r)
    # randomly adds silence and silence thresholds on 20 percent of the data.
    # this helps varying amounts of silence to sounds and still classifying them correctly
    if random.random() < 0.2:
        r = add_silence(r, 0.2)
    volume_threshold = random.choice(different_thresholds)
    return sample_width, r

def record_to_file(path, stream):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record(stream)
    data = pack('<' + ('h'*len(data)), *data)

    filename = str(datetime.datetime.now()).replace('.', '').replace(':', '_') + ".wav"
    wf = wave.open(path + filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

if __name__ == '__main__':
    """
    Main program asks you to speak rich and not rich, and places words it detects into files.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    finished_recording = False
    def check_key_press():
        global finished_recording
        finished_recording = False
        input('')
        finished_recording = True

    print("please say RICH words into the microphone. press enter when finished.")
    threading.Thread(target = check_key_press).start()
    richs_gained = 0
    while not finished_recording:
        richs_gained += 1
        print('richs gained:', richs_gained, end='\r')
        record_to_file('rich/', stream)
    print("done - results written to 'rich' folder")

    print("please say NON RICH words into the microphone. press enter when finished.")
    threading.Thread(target = check_key_press).start()
    not_richs_gained = 0
    while not finished_recording:
        not_richs_gained += 1
        print('not riches gained:', not_richs_gained, end='\r')
        record_to_file('notrich/', stream)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("done - results written to 'notrich' folder")
    print('Goodbye.')
