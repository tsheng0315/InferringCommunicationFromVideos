# generate normalised data and save in files
import data
import numpy as np

for video in data.videos:
    print(f"normalizing video {video}")
    
    # load prepared features, features dimensions: frame, person, features
    features, _ = data.load_prepared(video)
    # features -> centerised keypoints features
    data.center_position(features)
    np.save(f"data/{video}-features-normalized.npy", features)
