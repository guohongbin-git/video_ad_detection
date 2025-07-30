
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from video_ad_detector import database
from video_ad_detector import config

def find_matching_ad(recorded_video_features: np.ndarray):
    """
    Finds the best matching ad from the database.

    Args:
        recorded_video_features (np.ndarray): The feature vector of the recorded video.

    Returns:
        tuple: A tuple containing the matched ad's filename and the similarity score, or (None, 0) if no match is found.
    """
    material_features = database.get_all_material_features()
    if not material_features:
        return None, 0

    best_match = None
    max_similarity = 0

    for filename, db_features in material_features:
        similarity = cosine_similarity(recorded_video_features, db_features)[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = filename

    if max_similarity >= config.SIMILARITY_THRESHOLD:
        return best_match, max_similarity
    else:
        return None, 0
