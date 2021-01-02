#!/usr/bin/env python3
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2015 CNRS

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
from trs import TRSParser
from diarization import DiarizationErrorRate
from pyannote.core import Annotation
from pyannote.core import Segment as PyannSegment


if __name__ == "__main__":
	parser = TRSParser()

	video = 'NET20070331_thlep_1_2'

	reference = parser.read("../../gridnews/NET20070331/NET20070331.trs")

	hypothesis_visual = Annotation()

	# svm_diarization
	hypothesis_file = open("../../visual_diarization_svm/visual_hypothesis_files/" + video + "/visual_hypothesis_file.txt", "r")
	hypothesis_file_concatenate_segments = open("../../visual_diarization_svm/visual_hypothesis_files/" + video + "/hypothesis_file_concatenate_segments.txt", "w")

	hypothesis_file_concatenate_segments_temp = open("../../visual_diarization_svm/visual_hypothesis_files/" + video + "/hypothesis_file_concatenate_segments_temp.txt", "w")

	first_start_speaker = True
	line_number = 0
	for line in hypothesis_file:
		line_number = line_number + 1
		words = line.rstrip().split(' ')

		if line_number == 1:
			words_prvs = words
		else:
			if (int(words[0]) == int(words_prvs[0]) and int(words[1]) == int(words_prvs[1])):
				continue
			else:
				hypothesis_file_concatenate_segments_temp.write(str(int(words_prvs[0])) + ' ' + str(int(words_prvs[1])) + ' ' + str(int(words_prvs[2])) + '\n')
		words_prvs = words
	hypothesis_file_concatenate_segments_temp.write(str(int(words[0])) + ' ' + str(int(words[1])) + ' ' + str(int(words[2])) + '\n')

	hypothesis_file_concatenate_segments_temp.close()
	hypothesis_file.close()

	# svm_diarization
	hypothesis_file = open("../../visual_diarization_svm/visual_hypothesis_files/" + video + "/hypothesis_file_concatenate_segments_temp.txt", "r")

	# find the end of file
	line_number = 0
	for line in hypothesis_file:
		line_number = line_number + 1

	eof = line_number
	hypothesis_file.close()

	# concatenate the nearest segments
	# svm_diarization
	hypothesis_file = open("../../visual_diarization_svm/visual_hypothesis_files/" + video + "/hypothesis_file_concatenate_segments_temp.txt", "r")

	line_number = 0
	first_start_speaker = True
	for line in hypothesis_file:
		line_number = line_number + 1
		words = line.rstrip().split(' ')

		if line_number == 1:
			words_prvs = words
			speaker_id_prvs = int(words[2])
			start_speaker = int(words[0])
			end_speaker = int(words[1])
		else:

			if (int(words[0]) - int(words_prvs[1])) <= 100 and (int(words_prvs[2]) == int(words[2])):
				if first_start_speaker:
					start_speaker = int(words_prvs[0])
					first_start_speaker = False
			else:

				hypothesis_visual[PyannSegment(start=start_speaker * 0.04, end=int(words_prvs[1]) * 0.04)] = int(words_prvs[2])
				hypothesis_file_concatenate_segments.write(str(start_speaker) + ' ' + str(int(words_prvs[1])) + ' ' + str(int(words_prvs[2])) + '\n')
				start_speaker = int(words[0])
				first_start_speaker = True

		words_prvs = words

	hypothesis_visual[PyannSegment(start=start_speaker * 0.04, end=int(words[1]) * 0.04)] = int(words[2])
	hypothesis_file_concatenate_segments.write(str(start_speaker) + ' ' + str(int(words[1])) + ' ' + str(int(words[2])) + '\n')


metric = DiarizationErrorRate()

value = metric(reference, hypothesis_visual)
print("optical flow - svm der =", value)
