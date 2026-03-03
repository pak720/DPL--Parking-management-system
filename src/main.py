import cv2
import os
import csv
from extractor import extract_all
from matcher import find_match
from database import *

os.makedirs("captures", exist_ok=True)
init_db()

mode = input("Chọn mode (1 = Check-in, 2 = Check-out): ")
cap = cv2.VideoCapture(0)


if mode == "1":

    print("=== CHECK-IN MODE ===")

    ticket_counter = 1
    if os.path.exists("parking_db.csv"):
        with open("parking_db.csv") as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                ticket_counter = int(rows[-1][0]) + 1

    while True:

        ret, frame = cap.read()
        cv2.imshow("Check-in", frame)
        key = cv2.waitKey(1)

        if key == ord('i'):

            data = extract_all(frame)
            if data is None:
                print("Không detect được mặt")
                continue

            image_path = f"captures/in_{ticket_counter}.jpg"
            cv2.imwrite(image_path, frame)

            add_checkin(
                ticket_counter,
                data["plate"],
                data["helmet"],
                data["helmet_color"],
                data["face_emb"],
                data["clothes_emb"],
                image_path
            )

            print(f"Check-in ID = {ticket_counter}")
            ticket_counter += 1

        elif key == ord('q'):
            break


elif mode == "2":

    print("=== CHECK-OUT MODE ===")

    database_embeddings = load_active_tickets()

    while True:

        ret, frame = cap.read()
        cv2.imshow("Check-out", frame)
        key = cv2.waitKey(1)

        if key == ord('o'):

            data = extract_all(frame)
            if data is None:
                print("Không detect được mặt")
                continue

            query = {
                "face": data["face_emb"],
                "clothes": data["clothes_emb"],
                "helmet": data["helmet"],
                "plate": data["plate"]
            }

            match_id = find_match(query, database_embeddings)

            if match_id is None:
                print("Không tìm thấy vé phù hợp")
                continue

            image_path = f"captures/out_{match_id}.jpg"
            cv2.imwrite(image_path, frame)

            update_checkout(match_id, image_path)

            print(f"Check-out thành công ID = {match_id}")

        elif key == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()