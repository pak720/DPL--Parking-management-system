import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Trọng số: Prioritize plate matching (xác thực chính)
FACE_WEIGHT = 0.4
CLOTHES_WEIGHT = 0.15
HELMET_WEIGHT = 0.15
PLATE_WEIGHT = 0.3  # Tăng từ 0.1 lên 0.3 vì biển số là xác thực chính

# Ngưỡng chính + ngưỡng yếu hơn nếu biển số trùng
FACE_THRESHOLD = 0.70  # Face phải tốt
THRESHOLD_NORMAL = 0.72  # Ngưỡng tiêu chuẩn cao hơn
THRESHOLD_PLATE_MATCH = 0.65  # Nếu biển số trùng, có thể chấp nhận thấp hơn một chút


def safe_cosine(a, b):
    if a is None or b is None:
        return 0.0

    try:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        
        if np.isnan(a).any() or np.isnan(b).any():
            return 0.0
        
        # Normalize vectors
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
            
        a = a / a_norm
        b = b / b_norm
        
        return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])
    except:
        return 0.0


def validate_embedding(emb):
    """Kiểm tra embedding có hợp lệ không"""
    if emb is None:
        return False
    try:
        arr = np.asarray(emb, dtype=np.float32)
        return not np.isnan(arr).any() and arr.size > 0
    except:
        return False


def find_match(query_data, database):
    """
    Matching logic:
    1. Face phải >= 0.70 (bắt buộc)
    2. Nếu biển số trùng: threshold = 0.65
    3. Nếu biển số khác: threshold = 0.72
    4. Trả về best match hoặc None
    """
    
    if not database:
        print("❌ Database trống")
        return None

    best_id = None
    best_score = 0
    best_candidate = None
    
    plate_match_candidates = []  # Lưu candidates có biển số trùng

    # Validate query embeddings
    query_face_valid = validate_embedding(query_data.get("face"))
    query_clothes_valid = validate_embedding(query_data.get("clothes"))
    
    if not query_face_valid:
        print("❌ Face embedding không hợp lệ")
        return None

    print("\n📊 MATCHING ANALYSIS:")
    print(f"Query Plate: {query_data.get('plate', 'unknown')}")
    print(f"Query Helmet: {query_data.get('helmet', 'unknown')}\n")

    for ticket_id, stored in database.items():
        
        # Validate stored embeddings
        if not validate_embedding(stored.get("face")):
            print(f"⚠ ID {ticket_id}: Face embedding không hợp lệ, bỏ qua")
            continue

        # Calculate similarities
        face_sim = safe_cosine(query_data["face"], stored["face"])
        clothes_sim = safe_cosine(query_data["clothes"], stored["clothes"]) if query_clothes_valid else 0
        
        # Exact matches for discrete attributes
        helmet_match = 1 if query_data.get("helmet") == stored.get("helmet") else 0
        plate_match = 1 if query_data.get("plate") == stored.get("plate") else 0

        # **RULE 1: Face phải >= 0.70**
        if face_sim < FACE_THRESHOLD:
            print(f"ID {ticket_id}: Face {face_sim:.3f} < {FACE_THRESHOLD} ❌ (bỏ qua)")
            continue

        # Calculate weighted score
        final_score = (
            FACE_WEIGHT * face_sim +
            CLOTHES_WEIGHT * clothes_sim +
            HELMET_WEIGHT * helmet_match +
            PLATE_WEIGHT * plate_match
        )

        print(f"""ID {ticket_id}:
  Face: {face_sim:.3f} ✓
  Clothes: {clothes_sim:.3f}
  Helmet: {helmet_match} ({stored.get('helmet')})
  Plate: {plate_match} ({stored.get('plate')})
  Score: {final_score:.3f}""")

        # **RULE 2: Nếu biển số trùng, lưu vào candidates riêng**
        if plate_match == 1:
            plate_match_candidates.append((ticket_id, final_score, "plate_match"))
            print(f"  → Biển số TRÙNG (plate priority)")
        
        # **RULE 3: Track best score theo category**
        if final_score > best_score:
            best_score = final_score
            best_id = ticket_id
            best_candidate = ("normal_match", final_score)
            print(f"  → Best so far")
        print()

    # **Decision logic:**
    if plate_match_candidates:
        # Ưu tiên plate match
        plate_match_candidates.sort(key=lambda x: x[1], reverse=True)
        best_plate_id, best_plate_score, _ = plate_match_candidates[0]
        
        print(f"✅ PLATE MATCH FOUND: ID {best_plate_id} | Score = {best_plate_score:.3f}")
        print(f"   (Biển số trùng, sử dụng ngưỡng {THRESHOLD_PLATE_MATCH})")
        
        if best_plate_score >= THRESHOLD_PLATE_MATCH:
            return best_plate_id
        else:
            print(f"⚠ Score {best_plate_score:.3f} < {THRESHOLD_PLATE_MATCH}, không xác nhận")
            return None
    
    # Nếu không có plate match, dùng best overall score
    if best_id is not None and best_score >= THRESHOLD_NORMAL:
        print(f"✅ FACE MATCH: ID {best_id} | Score = {best_score:.3f}")
        return best_id

    print(f"❌ Không tìm thấy match (best score = {best_score:.3f} < {THRESHOLD_NORMAL})")
    return None


def verify_checkout(checkin_data, checkout_data):
    """
    Xác thực checkout bằng cách so sánh check-in vs checkout
    - Face similarity >= 0.75
    - Plate phải trùng hoàn toàn
    - Helmet phải trùng hoàn toàn
    
    Return: (is_valid, score, details)
    """
    
    if not validate_embedding(checkin_data.get("face")) or not validate_embedding(checkout_data.get("face")):
        return False, 0.0, "❌ Face embedding không hợp lệ"
    
    # So sánh face
    face_sim = safe_cosine(checkin_data["face"], checkout_data["face"])
    
    # So sánh clothes
    clothes_sim = safe_cosine(checkin_data["clothes"], checkout_data["clothes"]) if (
        validate_embedding(checkin_data.get("clothes")) and 
        validate_embedding(checkout_data.get("clothes"))
    ) else 0.5
    
    # So sánh plate
    plate_match = checkin_data.get("plate") == checkout_data.get("plate")
    
    # So sánh helmet
    helmet_match = checkin_data.get("helmet") == checkout_data.get("helmet")
    
    # Tính điểm xác thực
    verify_score = (
        FACE_WEIGHT * face_sim +
        CLOTHES_WEIGHT * clothes_sim +
        HELMET_WEIGHT * (1.0 if helmet_match else 0.0) +
        PLATE_WEIGHT * (1.0 if plate_match else 0.0)
    )
    
    print("\n🔐 VERIFICATION CHECK-IN vs CHECK-OUT:")
    print(f"  Face similarity: {face_sim:.3f}")
    print(f"  Clothes similarity: {clothes_sim:.3f}")
    print(f"  Helmet match: {'✓' if helmet_match else '✗'} (Check-in: {checkin_data.get('helmet')} vs Check-out: {checkout_data.get('helmet')})")
    print(f"  Plate match: {'✓' if plate_match else '✗'} (Check-in: {checkin_data.get('plate')} vs Check-out: {checkout_data.get('plate')})")
    print(f"  Verification Score: {verify_score:.3f}")
    
    # **Điều kiện bắt buộc**
    if not plate_match:
        return False, verify_score, "❌ Biển số khác - từ chối checkout"
    
    if not helmet_match:
        return False, verify_score, "❌ Mũ bảo hiểm khác - từ chối checkout"
    
    if face_sim < 0.70:
        return False, verify_score, f"❌ Face không khớp ({face_sim:.3f} < 0.70) - từ chối checkout"
    
    # Nếu tất cả điều kiện thỏa
    if verify_score >= 0.75:
        return True, verify_score, "✅ Xác thực thành công - cho phép checkout"
    else:
        return False, verify_score, f"⚠ Điểm xác thực thấp ({verify_score:.3f}) - từ chối checkout"