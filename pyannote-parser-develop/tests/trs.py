#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2015 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


from __future__ import print_function
import os

"""
Support for TRS file format

TRS (TRanScriber) is a file format used by TranScriber audio annotation tool.

References
----------
http://transag.sourceforge.net/
"""

import re
import numpy as np

try:
    from lxml import objectify
    from lxml import etree
except:
    pass

from pyannote.core import Segment
from base import AnnotationParser
from pyannote.core import Annotation
import shutil 

class TRSParser(AnnotationParser):

    def __init__(self):
        super(TRSParser, self).__init__()

    def _complete(self):
        for segment in self._incomplete:
            segment.end = self._sync
        self._incomplete = []

    def _parse_speakers(self, turn):
        string = turn.get('speaker')
        if string:
            return string.strip().split()
        else:
            return []

    def _parse_spoken(self, element):
        string = element.tail
        if not string:
            return []

        labels = []
        p = re.compile('.*?<pers=(.*?)>.*?</pers>', re.DOTALL)
        m = p.match(string)
        while(m):
            # split Jean-Marie_LEPEN,Marine_LEPEN ("les LEPEN")
            for label in m.group(1).split(','):
                try:
                    labels.append(str(label))
                except Exception as e:
                    print("label = %s" % label)
                    raise e
            string = string[m.end():]
            m = p.match(string)
        return labels

    def read(self, path, uri=None, **kwargs):
        print(path)  
        # objectify xml file and get root
        root = objectify.parse(path).getroot()

        # if uri is not provided, infer (part of) it from .trs file
        if uri is None:
            uri = root.get('audio_filename')

        # speaker names and genders
        name = {}
        gender = {}
        for speaker in root.Speakers.iterchildren():
            name[speaker.get('id')] = speaker.get('name')
            gender[speaker.get('id')] = speaker.get('type')

        reference = Annotation()
        # # incomplete segments
        # # ie without an actual end time
        # self._incomplete = []

        # speech turn number
        speakers_list = []
        track = 0
        start = 1505.931
        end = 1952.862
        first_time = True
        for section in root.Episode.iterchildren():

            # transcription status (report or nontrans)
            section_start = float(section.get('startTime'))
            section_end = float(section.get('endTime'))
            section_segment = Segment(start=section_start, end=section_end)
            label = section.get('type')
            #self._add(section_segment, None, label, uri, 'status')

            # # sync
            # self._sync = section_start
            # self._complete()
            
            #print(label)

            if label == 'report':
                for turn in section.iterchildren():
                    turn_start_time = float(turn.get('startTime'))
                    turn_end_time = float(turn.get('endTime'))
                    if first_time:
                        video_start = turn_start_time
                        first_time = False
                    if turn_start_time >= start and turn_end_time <= end:
                        
                        speaker = str(turn.get('speaker'))
                        if speaker.find(' spk') != -1:
                            speakers = speaker.split(' ')
                            num_speakers = len(speakers)
                            for i in range(num_speakers):
                                reference[Segment(start=turn_start_time-start, end=turn_end_time-start), i] = speakers[i]
                                if speakers[i] in speakers_list:
                                    continue
                                else:
                                    speakers_list.append(speakers[i]) 
                        else:
                            reference[Segment(start=turn_start_time-start, end=turn_end_time-start)] = speaker
                            if speaker in speakers_list:
                                    continue
                            else:
                                speakers_list.append(speaker)
                    
        video_end = section_end
        number_of_frames = int((end -start)/0.04)
        y_train_svm = np.zeros((number_of_frames,len(speakers_list)),dtype="int")
        

        for section in root.Episode.iterchildren():

            # transcription status (report or nontrans)
            section_start = float(section.get('startTime'))
            section_end = float(section.get('endTime'))
            section_segment = Segment(start=section_start, end=section_end)
            label = section.get('type')
            #self._add(section_segment, None, label, uri, 'status')

            # # sync
            # self._sync = section_start
            # self._complete()
            
            #print(label)

            if label == 'report':
                for turn in section.iterchildren():
                    turn_start_time = float(turn.get('startTime'))
                    turn_end_time = float(turn.get('endTime'))
                    if turn_start_time >= start and turn_end_time <= end :
                        speaker = str(turn.get('speaker'))
                        if speaker.find(' spk') != -1:
                            speakers = speaker.split(' ')
                            num_speakers = len(speakers)
                            for i in range(num_speakers):
                                speakers_index = speakers_list.index(speakers[i]) 
                                for j in range(int((turn_start_time-start)/0.04),int((turn_end_time-start)/0.04)):
                                    y_train_svm[j][speakers_index] = 1
                                
                        else:
                            speakers_index = speakers_list.index(speaker) 
                            for j in range(int((turn_start_time-start)/0.04),int((turn_end_time-start)/0.04)):
                                y_train_svm[j][speakers_index] = 1
        video = "NET20070331_thlep_1_2"
        file_path = "../../..visual_diarization_svm/targets/y_test_files/"+ video +"/y_test_file.txt"
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        y_train_svm_file = open(file_path, "w")
        speaker_id_int = []
        for speaker in speakers_list:
            speaker = int(re.search(r'\d+', speaker).group(0))
            speaker_id_int.append(speaker)
        for i in range(y_train_svm.shape[0]):
            for j in range(y_train_svm.shape[1]):
                y_train_svm_file.write(str(i) + ' ' + str(speaker_id_int[j]) + ' ' + str(y_train_svm[i][j]) + '\n')
        y_train_svm_file.close()
        
        return reference
if __name__ == "__main__":
    import doctest
    doctest.testmod()
