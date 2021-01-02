# Visual diarization with support vector machines

## What is this?
This is a python project for visual speaker diarization. 
The goal is to find out who speaks and when in videos with multi-speaker conversations in Broadcast News of Greek television, 
using lip motion features and Support Vector Machines.

## Requirements
You need python 3.6.9 version and conda 4.6.11 environment.

## Dataset and ground truth
The gridnews directory contains the videos that were used for training and testing as well the corresponding transcriptions files.
[dataset](https://drive.google.com/drive/u/0/folders/1TO72-uN6_vexSOJdIr3HG-Hws4gyPCwT)

## Step for running the project
### Training
In visual_diarization_svm directory run the following scripts:

1. dlib_face_detection_svm_train.py (creates the optical flow features)
2. clear_opt_flow_features_train.py (ignores the incidental speakers)
	* Should change manually the ids of each main speaker to a unique id at optical_flow_features_train.txt
3. svm_train.py (trains an svm classifier)

### Testing
In visual_diarization_svm directory run the following scripts:
1. dlib_face_detection_svm_test.py (creates the optical flow features)
2. svm_test.py (predicts speech or not speech for each speaker in a frame)
	* Should change manually the ids of each main speaker to a unique id at predictions_file.txt
3. svm_test_window_N (predicts speech or not speech for each speaker in a window of N frames)

### Evaluation
In pyannote-parser-develop/tests directory
run the trs_file_parse_visual_svm.py (computes the diarization error rate). 
Initialize the video variable with the video name. In reference parser write the corresponding transcription file.

	
	video = 'NET20070331_thlep_1_2'

	reference = parser.read("../gridnews/NET20070331/NET20070331.trs")


Alongside, in trs.py initialize the video variable with the video name and set the time limits (start, end)


	video = 'NET20070331_thlep_1_2'

	start = 1505.931
    	end = 1952.862


Time limits for the 4 testing videos:
* Net20070329_thelep_1_1: start = 1705.173, end = 2455.011
* Net20070330_thelep_1_1: start = 1180.084, end = 1540.335
* Net20070330_thelep_1_4: start = 2246.728, end = 2615.844
* Net20070331_thelep_1_2: start = 1505.931, end = 1952.862

## Author
Charalampos Vossos
* [linkedin](https://www.linkedin.com/in/charalampos-vossos-6bbb78185/)
* email: <vossosx96@gmail.com>

## License
Copyright © 2021, Charalampos Vossos. Licensed under [MIT License](LICENSE)
