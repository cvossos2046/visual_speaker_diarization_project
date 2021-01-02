import numpy as np
import pickle
import os
import errno


def main():
	video = "NET20070330_thlep_1_1"

	optical_flow_features_test = open("optical_flow_features_test/" + video + "/optical_flow_features_test.txt", "r")

	file_path = "predictions_files/" + video + "/predictions_file.txt"
	if not os.path.exists(os.path.dirname(file_path)):
		try:
			os.makedirs(os.path.dirname(file_path))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	predictions_file = open(file_path, "w")

	filename = '../model/svm_model.sav'
	clf = pickle.load(open(filename, 'rb'))

	x_test = np.zeros((256), dtype="float")

	line_number = 0
	for line in optical_flow_features_test:
		words = line.rstrip().split(' ')
		if len(words) == 256:
			line_number = line_number + 1

	optical_flow_features_test.close()

	predictions = np.zeros((line_number), dtype="int")

	optical_flow_features_test = open("optical_flow_features_test/" + video + "/optical_flow_features_test.txt", "r")

	line_number = -1
	for line in optical_flow_features_test:
		print(line_number)

		words = line.rstrip().split(' ')
		if line_number == -1:
			words_prvs = words
			line_number = line_number + 1
		if len(words) == 256:
			for i in range(len(words)):
				x_test[i] = words[i]
			predictions[line_number] = clf.predict([x_test])
			predictions_file.write(words_prvs[0] + ' ' + words_prvs[1] + ' ' + str(predictions[line_number]) + '\n')
			line_number = line_number + 1
		words_prvs = words

	optical_flow_features_test.close()
	predictions_file.close()


if __name__ == "__main__":
	main()
