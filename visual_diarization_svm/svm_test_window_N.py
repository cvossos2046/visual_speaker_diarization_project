import numpy as np
import os
import errno


def main():
	video = "NET20070331_thlep_1_2"
	file_path = "predictions_files/" + video + "/predictions_lium_ids.txt"

	predictions_lium_ids = open(file_path, "r")

	file_path = "visual_hypothesis_files/" + video + "/visual_hypothesis_file.txt"
	if not os.path.exists(os.path.dirname(file_path)):
		try:
			os.makedirs(os.path.dirname(file_path))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	visual_hypothesis_file_svm = open(file_path, "w")

	window_size = 150
	ids_list = []

	for line in predictions_lium_ids:
		words = line.rstrip().split(' ')
		if words[1] in ids_list:
			continue
		else:
			ids_list.append(words[1])

	predictions_lium_ids.close()

	predictions_lium_ids = open("predictions_files/" + video + "/predictions_lium_ids.txt", "r")

	id_counts = np.zeros((len(ids_list), 2), dtype='int')

	for i in range(len(ids_list)):
		id_counts[i][0] = int(ids_list[i])

	line_number = 0
	for line in predictions_lium_ids:
		words = line.rstrip().split(' ')
		frame_number = int(words[0])
		speaker_lium_id = int(words[1])
		predictions = int(words[2])

		if line_number == 0:
			window_start = int(words[0])

		if frame_number == (window_start + window_size):
			index = np.argmax(id_counts[:, 1])
			speaker = id_counts[index, 0]
			visual_hypothesis_file_svm.write(str(window_start) + ' ' + str(frame_number) + ' ' + str(speaker) + '\n')
			id_counts[:, 1] = 0
			window_start = frame_number

		for i in range(len(id_counts)):
			if speaker_lium_id == id_counts[i][0]:
				id_counts[i][1] = id_counts[i][1] + predictions
		line_number = line_number + 1

	predictions_lium_ids.close()
	visual_hypothesis_file_svm.close()


if __name__ == "__main__":
	main()
