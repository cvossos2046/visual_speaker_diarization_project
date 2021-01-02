from __future__ import division
from imutils import face_utils
import dlib
import cv2
import numpy as np
import imutils
from pyimagesearch import CentroidTracker
import os
import errno


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


def main():
	video = "NET20070330_thlep_1_2"

	camera = cv2.VideoCapture('../gridnews/visual_svm_training_set/' + video + '.mkv')
	print("fps", camera.get(cv2.CAP_PROP_FPS))
	number_of_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
	first_frame_with_faces = number_of_frames

	file_path = "optical_flow_features_train/" + video + "/optical_flow_features_train.txt"

	if not os.path.exists(os.path.dirname(file_path)):
		try:
			os.makedirs(os.path.dirname(file_path))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	optical_flow_features_train = open(file_path, "w")

	file_path = "visual_hypothsesis_files/" + video + "/visual_hypothsesis_file.txt"
	if not os.path.exists(os.path.dirname(file_path)):
		try:
			os.makedirs(os.path.dirname(file_path))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	visual_hypothesis_file = open(file_path, "w")

	predictor_path = 'shape_predictor_68_face_landmarks.dat'

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	frame_number = 0

	ct = CentroidTracker()

	mag_reduced_dim = np.zeros((16, 16), dtype="float")

	prvs = []
	next = []
	objectid_index_list_prvs = []

	first_time = True

	while True:
		objectid_num_in_frame = 0
		objectid_index_list_next = []
		rects = []
		ret, frame = camera.read()
		frame_number = frame_number + 1
		print(frame_number)

		if first_time is False:
			if frame_number >= first_frame_with_faces + 2:
				prvs = next
				objectid_index_list_prvs = objectid_index_list_next

		next = []
		if ret is False:
			print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
			break

		frame = imutils.resize(frame, width=500)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		dets = detector(frame_gray, 1)

		if len(dets) > 0:
			if first_time:
				first_frame_with_faces = frame_number
				first_time = False
			for k, d in enumerate(dets):
				shape = predictor(frame_gray, d)
				shape_np = shape_to_np(shape)
				(x, y, w, h) = face_utils.rect_to_bb(d)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

				startx = x
				starty = y
				endx = x + w
				endy = y + h
				rect = (startx, starty, endx, endy)

				rects.append(rect)
				objects = ct.update(rects)

				# loop over the tracked objects
				for (objectID, centroid) in objects.items():
					# draw both the ID of the object and the centroid of the
					# object on the output frame
					text = "ID {}".format(objectID)
					cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
					if ((startx <= centroid[0] <= endx) and (starty <= centroid[1] <= endy)):
						for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
							if (name == "mouth"):
								objectid_num_in_frame = objectid_num_in_frame + 1

								cv2.putText(frame, "Face #{}".format(k + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
								cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

								x_max, y_max = shape_np[i:j].max(axis=0)
								x_min, y_min = shape_np[i:j].min(axis=0)

								mouth_center_x = int((x_min + x_max) / 2)
								x_min = int((mouth_center_x + x_min) / 2)
								x_max = int((mouth_center_x + x_max) / 2)

								x_min = int((mouth_center_x + x_min) / 2)
								x_max = int((mouth_center_x + x_max) / 2)

								cv2.rectangle(frame, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), (255, 0, 255), 2)

								mouth_region = frame[y_min:y_max, x_min:x_max]
								mouth_region = cv2.resize(mouth_region, (32, 32))

								norm_mouth_region = np.zeros((32, 32), dtype="float64")
								norm_mouth_region = cv2.normalize(mouth_region, norm_mouth_region, 0, 255, norm_type=cv2.NORM_MINMAX)

								cv2.imshow('flow', norm_mouth_region)

								if frame_number == first_frame_with_faces:
									frame1 = norm_mouth_region
									prvs.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
									objectid_index_list_prvs.append(objectID)

								else:
									frame2 = norm_mouth_region
									next.append(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY))
									objectid_index_list_next.append(objectID)
									if (objectid_num_in_frame <= len(prvs)):
										flow = cv2.calcOpticalFlowFarneback(prvs[objectid_index_list_prvs.index(objectID)], next[len(next) - 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
										mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
										for i in range(int(mag.shape[0] / 2)):
											for j in range(int(mag.shape[1] / 2)):
												mag_reduced_dim[i, j] = (mag[i, j] + mag[i, j + 1] + mag[i + 1, j] + mag[i + 1, j + 1]) / 4

									optical_flow_features_train.write(str(frame_number) + ' ' + str(objectID) + '\n')

									for el in mag_reduced_dim.flatten():
										optical_flow_features_train.write(str(el) + ' ')
									optical_flow_features_train.write('\n')

		cv2.imshow("image", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			camera.release()
			break

	visual_hypothesis_file.close()
	optical_flow_features_train.close()


if __name__ == "__main__":
	main()
