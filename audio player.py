# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:12:45 2020

@author: jim
"""

import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pyaudio
import os
from scipy.io import wavfile
import librosa
import librosa.display
import pdb
import wave
import struct
import subprocess

def read_audio(filename, fs=16000):
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        sample_rate, data = wavfile.read(filename)
        # Expected mono channel
        if data.shape[-1] == 2:
            data = data[:,0]
        if data.dtype.name != 'float32' and data.dtype.name != 'float64':
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        # Resample
        if fs != sample_rate:
            data = librosa.core.resample(data, sample_rate, fs)
            sample_rate = fs
        # pad 0 if len(data) < sample rate
        if len(data)<fs:
            pad = np.zeros((fs-len(data)+20))
            #pdb.set_trace()
            data = np.concatenate((data, pad))
        return data, sample_rate
    return None, None

def crop_audio(BGM, sr=16000, bgm_duration=5):
    start_sample = np.random.randint(1, len(BGM) - int(bgm_duration*sr))
    BGM = BGM[start_sample : start_sample + bgm_duration*sr]
    return BGM

class Application(tkinter.Tk):
    def  __init__ (self, BGM_list):
        super().__init__()
        self.wm_title("Audio PLAYER")
        self.BGM_path = random.choice(BGM_list)
        self.fs = 16000
        self.CHUNK = 100
        
        self.att_predict = []
        self.values = []
        self.answer = []
        
        self.interface()
        self._init_after_()
        
        self.bind('P', lambda event: self._play_())
        self.bind('<space>', lambda event: self._click())
        self.bind('Q', lambda event: self._quit())
        self.bind('R', lambda event: self._reset())
        
    def _init_after_(self):
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self._create_bgm_stream()
        print('wait for loading....')
        print('Starting....')
        self.after(100, self.RealtimePloter)
        
        xAchse = np.arange(0,100,1)
        yAchse = np.array([0]*100)
        self.line1 = self.user_ax.plot(xAchse,yAchse,'-')
        
    def interface(self):
        fig = plt.figure(1, figsize=(10,3))
        self.user_ax = fig.add_subplot(111)
        self.user_ax.grid(True)
        self.user_ax.set_title("User")
        self.user_ax.set_ylabel("Amplitude")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.canvas.draw()
        
        button_frame = tkinter.Frame(self)
        button_frame.pack(side=tkinter.BOTTOM)
        
        quit_btn = tkinter.Button(master=button_frame, text='Quit', command=self._quit, height = 5, width = 20)
        quit_btn.pack(side=tkinter.LEFT)
        
        play_btn = tkinter.Button(master=button_frame, text='Play', command=self._play_, height = 5, width = 20)
        play_btn.pack(side=tkinter.LEFT)
        
        click_btn = tkinter.Button(master=button_frame, text='CLICK', command=self._click, height = 5, width = 50)
        click_btn.pack(side=tkinter.LEFT)
        
        reset_btn = tkinter.Button(master=button_frame, text='reset', command=self._reset, height = 5, width = 5)
        reset_btn.pack(side=tkinter.LEFT)
    
    def _reset(self):
        self.after_cancel(self.plot_id)
        
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.wf.close()
        
        self.values = []
        self.answer = []
        
        for axx in plt.gcf().get_axes():
            for artist in axx.lines + axx.collections:
                artist.remove()
        self.canvas.draw()
    
        self._init_after_()
    
    def _create_bgm_stream(self):
        audio, _ = read_audio(self.BGM_path, self.fs)
        self.audio = crop_audio(audio)
        self.user_ax.axis([0, len(self.audio), -1.5, 1.5])
        librosa.output.write_wav('./audio.wav', self.audio, self.fs)
        cmd2 = 'sox {} -b 16 -e signed-integer {}'.format('./audio.wav', './audio_1.wav')
        subprocess.call(cmd2.split(' '))
        print('DONE')
        
        self.wf = wave.open('./audio_1.wav', 'rb')
        self.p = pyaudio.PyAudio()
        def callback(in_data, frame_count, time_info, status):
            data = self.wf.readframes(frame_count)
            try:
                self.values.extend(np.array(struct.unpack("<" + "h" * self.CHUNK, data)) / np.iinfo(np.int16).max)
            except struct.error:
                pass
            return (data, pyaudio.paContinue)
        
        self.stream = self.p.open(format = self.p.get_format_from_width(self.wf.getsampwidth()),
                        channels = self.wf.getnchannels(),
                        rate = self.wf.getframerate(),
                        output = True,
                        stream_callback = callback,
                        frames_per_buffer = self.CHUNK,
                        start = False)
    
    def RealtimePloter(self):
        NumberSamples = min(len(self.values), self.CHUNK)
        CurrentXuser_axis = np.arange(len(self.values)-NumberSamples, len(self.values), 1)
        self.line1[0].set_data(CurrentXuser_axis, np.array(self.values[-NumberSamples:]))
        self.canvas.draw()
        self.plot_id = self.after(10, self.RealtimePloter)
    
    def _play_(self):
        self.stream.start_stream()
    
    def _click(self):
        self.answer.append(len(self.values))
        self.user_ax.axvline(x=self.answer[-1], linewidth=4, color='r')
    
    def _quit(self):
        self.quit()
        self.destroy()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        plt.close('all')
        self.wf.close()

if  __name__ == '__main__':
    BGM_list = glob.glob('./data/*.wav')
    gui = Application(BGM_list)
    gui.mainloop()