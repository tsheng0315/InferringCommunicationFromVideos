import numpy as np

# videos = ["1", "2", "3", "4", "5", "6", "7"]
videos = ["1"]
#  training batch
# i th folder 
def k_fold_training(i, size):
    return videos[:i * size] + videos[(i + 1) * size:]
# testing batch
def k_fold_testing(i, size):
    return videos[i * size:(i + 1) * size]

def transform_target(row):
    result = np.zeros(4 * len(row), dtype='float32')
    for x, value in enumerate(row):
        result[int(value + (x * 4))] = 1
    return result


def load_video_data(video_id):
    # with open(resultJson, 'w') as infile:
    #     finalResult = json.load(infile)
    # return finalResult
    return np.genfromtxt(f'./data/{video_id}.csv', delimiter=',', dtype='float32')
    # return np.genfromtxt(f'./data/{video_id}_missing_label.csv', delimiter=',', dtype='float32')

def prepare(video_id):
    video_data = load_video_data(video_id)
    # class 
    target = video_data[:, 0:2]
    # ["pose_keypoints_2d","hand_left_keypoints_2d", "hand_right_keypoints_2d","gaze"]
    features = video_data[:, 2:]
    # csv to np
    target = np.apply_along_axis(transform_target, 1, video_data[:, 0:2])
    target = target.reshape(target.shape[0], 2, 4)
    features = features.reshape(features.shape[0], 2, 39 + 63 + 63+6)

    return features, target   


def load_prepared(video_id):
    return np.load(f"data/{video_id}-features.npy"), np.load(f"data/{video_id}-target.npy")

def load_prepared_normalized(video_id):
    return np.load(f"data/{video_id}-features-normalized.npy"), np.load(f"data/{video_id}-target.npy")

def load_original_data(video_id):
    return  np.genfromtxt(f'./videosLabelled/{video_id}/data.csv', delimiter=',',  names=True)

def load_10fps(video_id):
    return np.load(f"data/{video_id}-features-10fps.npy"), np.load(f"data/{video_id}-target-10fps.npy")

def load_10fps_normalized(video_id):
    return np.load(f"data/{video_id}-features-10fps-normalized.npy"), np.load(f"data/{video_id}-target-10fps.npy")


def video_fps(video_id):
    data = load_original_data(video_id)
    return data["fps"][0]

def center_position(features):
    # BODY_NOSE = 0
    BODY_NECT = 1

    for frame in range(features.shape[0]):
        for person in range(features.shape[1]):
            # neck in x axis, 
            neck_x = features[frame, person, BODY_NECT * 3]
            neck_y = features[frame, person, BODY_NECT * 3 + 1]

            for p in range(int(features.shape[2] / 3.0)):
                # each keypoint from openpose normalised with neck_x,neck_y
                features[frame, person, p * 3] -= neck_x
                features[frame, person, p * 3 + 1] -= neck_y

