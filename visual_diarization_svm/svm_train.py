import numpy as np
from sklearn import svm
import pickle
import os
import errno


def main():
    video = "NET20070330_thlep_1_2"
    file_path = "optical_flow_features_train/" + video
    optical_flow_features_train = open(file_path + "/clear_opt_flow_features_train.txt", "r")
    y_train_file = open("targets/y_train_files/" + video + "/y_train_file.txt", "r")

    optical_flow_features_list = []
    optical_flow_frame_id_list = []
    y_train_svm_list = []
    x_train_list = []
    y_train_list = []
    visual_ids = []
    optical_flow_features = np.zeros((256), dtype="float")

    line_number = 0
    for line in optical_flow_features_train:
        words = line.rstrip().split(' ')
        if len(words) == 2:
            optical_flow_frame_id_list.append([int(words[0]), int(words[1])])
            line_number = line_number + 1
            if not int(words[1]) in visual_ids:
                visual_ids.append(int(words[1]))
        else:
            for i, el in enumerate(words):
                optical_flow_features[i] = el
            optical_flow_features_list.append(tuple(optical_flow_features))

    print("visual_ids", visual_ids)

    for line in y_train_file:
        words = line.rstrip().split(' ')
        frame = int(words[0])
        speakers_id = int(words[1])
        target = int(words[2])
        y_train_svm_list.append([frame, speakers_id, target])

    optical_flow_features_train.close()
    y_train_file.close()
    last_pos = 0
    k = 0
    for el_x1, el_x2 in zip(optical_flow_frame_id_list, optical_flow_features_list):
        i = last_pos
        while True:
            if el_x1[0] != y_train_svm_list[i][0]:
                i = i + 1
            else:
                j = i
                while el_x1[0] == y_train_svm_list[j][0] and el_x1[1] != y_train_svm_list[j][1]:
                    j = j + 1
                if el_x1[1] == y_train_svm_list[j][1]:
                    k = k + 1
                    x_train_list.append(el_x2)
                    y_train_list.append(y_train_svm_list[j][2])
                last_pos = i
                break

    x_train = np.asarray(x_train_list)
    y_train = np.asarray(y_train_list)
    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)

    file_path = '../model/svm_model.sav'
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    clf = svm.SVC(kernel='rbf', C=1.0, gamma='auto')

    clf.fit(x_train, y_train)
    print("training accuracy", clf.score(x_train, y_train))

    pickle.dump(clf, open(file_path, 'wb'))


if __name__ == "__main__":
    main()
