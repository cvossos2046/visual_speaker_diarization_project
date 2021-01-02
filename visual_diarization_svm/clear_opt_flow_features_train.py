def main():
    video = "NET20070330_thlep_1_2"
    file_path = "optical_flow_features_train/" + video
    optical_flow_features_train = open(file_path + "/optical_flow_features_train.txt", "r")

    clear_of_features_train = open(file_path + "/clear_opt_flow_features_train.txt", "w")
    visual_ids = []

    for line in optical_flow_features_train:
        words = line.rstrip().split(' ')
        if len(words) == 2:
            if not int(words[1]) in visual_ids:
                visual_ids.append(int(words[1]))

    print("visual_ids", visual_ids)
    optical_flow_features_train.close()
    optical_flow_features_train = open(file_path + "/optical_flow_features_train.txt", "r")

    for line in optical_flow_features_train:
        words = line.rstrip().split(' ')
        if len(words) == 2:
            frame = int(words[0])
            speaker_id = int(words[1])
        else:
            # every id that each speaker gets during the video
            if speaker_id == 0 or speaker_id == 6 or speaker_id == 13 or speaker_id == 15 or speaker_id == 5 or speaker_id == 16 or speaker_id == 4 or speaker_id == 17 or speaker_id == 3 or speaker_id == 18:
                clear_of_features_train.write(str(frame) + ' ' + str(speaker_id) + '\n')
                for features in words:
                    clear_of_features_train.write(str(features) + ' ')

                clear_of_features_train.write('\n')

    optical_flow_features_train.close()
    clear_of_features_train.close()


if __name__ == "__main__":
    main()
