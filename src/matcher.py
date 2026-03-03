import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FACE_WEIGHT = 0.6
CLOTHES_WEIGHT = 0.2
HELMET_WEIGHT = 0.1
PLATE_WEIGHT = 0.1

THRESHOLD = 0.65


def safe_cosine(a, b):

    if a is None or b is None:
        return 0

    if np.isnan(a).any() or np.isnan(b).any():
        return 0

    a = np.asarray(a).reshape(1, -1)
    b = np.asarray(b).reshape(1, -1)

    return cosine_similarity(a, b)[0][0]


def find_match(query_data, database):

    best_id = None
    best_score = 0

    for ticket_id, stored in database.items():

        face_sim = safe_cosine(query_data["face"], stored["face"])
        clothes_sim = safe_cosine(query_data["clothes"], stored["clothes"])

        helmet_match = 1 if query_data["helmet"] == stored["helmet"] else 0
        plate_match = 1 if query_data["plate"] == stored["plate"] else 0

        weights_sum = 0
        final_score = 0

        if face_sim > 0:
            final_score += FACE_WEIGHT * face_sim
            weights_sum += FACE_WEIGHT

        if clothes_sim > 0:
            final_score += CLOTHES_WEIGHT * clothes_sim
            weights_sum += CLOTHES_WEIGHT

        final_score += HELMET_WEIGHT * helmet_match
        final_score += PLATE_WEIGHT * plate_match
        weights_sum += HELMET_WEIGHT + PLATE_WEIGHT

        if weights_sum > 0:
            final_score = final_score / weights_sum

        print(f"""
ID {ticket_id}
Face: {face_sim:.3f}
Clothes: {clothes_sim:.3f}
Helmet: {helmet_match}
Plate: {plate_match}
FinalScore: {final_score:.3f}
""")

        if final_score > best_score:
            best_score = final_score
            best_id = ticket_id

    if best_score >= THRESHOLD:
        print(f"Match ID {best_id} | Score = {best_score:.3f}")
        return best_id

    print("No match above threshold")
    return None