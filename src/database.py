import csv
import os
import numpy as np

DB_PATH = "parking_db.csv"
EMB_DIR = "embeddings"

os.makedirs(EMB_DIR, exist_ok=True)


def init_db():
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ticket_id",
                "plate",
                "helmet",
                "helmet_color",
                "face_embedding_path",
                "clothes_embedding_path",
                "checkin_image",
                "checkout_image",
                "status"
            ])


def add_checkin(ticket_id, plate, helmet, helmet_color,
                face_emb, clothes_emb, image_path):

    face_emb = np.asarray(face_emb, dtype=np.float32)
    clothes_emb = np.asarray(clothes_emb, dtype=np.float32) if clothes_emb is not None else None

    face_path = os.path.join(EMB_DIR, f"{ticket_id}_face.npy")
    np.save(face_path, face_emb)

    if clothes_emb is not None:
        clothes_path = os.path.join(EMB_DIR, f"{ticket_id}_clothes.npy")
        np.save(clothes_path, clothes_emb)
    else:
        clothes_path = ""

    with open(DB_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ticket_id,
            plate,
            helmet,
            helmet_color,
            face_path,
            clothes_path,
            image_path,
            "",
            "ACTIVE"
        ])


def load_active_tickets():

    tickets = {}

    if not os.path.exists(DB_PATH):
        return tickets

    with open(DB_PATH, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "ACTIVE":

                face_emb = np.load(row["face_embedding_path"])

                if row["clothes_embedding_path"] != "":
                    clothes_emb = np.load(row["clothes_embedding_path"])
                else:
                    clothes_emb = None

                tickets[int(row["ticket_id"])] = {
                    "face": face_emb,
                    "clothes": clothes_emb,
                    "helmet": row["helmet"],
                    "plate": row["plate"]
                }

    return tickets


def update_checkout(ticket_id, image_path):

    rows = []
    with open(DB_PATH, mode="r") as f:
        rows = list(csv.reader(f))

    for row in rows[1:]:
        if row[0] == str(ticket_id) and row[8] == "ACTIVE":
            row[7] = image_path
            row[8] = "DONE"

    with open(DB_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)