import os
import shutil
import joblib
import json
import hashlib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from deepface import DeepFace
from numpy.linalg import norm
from datetime import datetime

from audit_chain import (
    add_block,
    load_chain,
    verify_chain,
    verify_dead_matches,
    verify_missing_search_cases,
    verify_missing_dataset,
    verify_child_cases,
    verify_child_dataset,
    verify_child_matches,
    sha256_file,
    sha256
)

# Configuration
MISSING_PERSON_FOLDER = "missing_dataset"
MISSING_THUMB_FOLDER = "missing_dataset/thumbs"
MISSING_SEARCH_CASE_FOLDER = "missing_search_cases"
DEAD_CONFIRM_FOLDER = "storage/confirmed_dead_matches"
CHILD_CONFIRM_FOLDER = "storage/confirmed_child_matches"
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.jfif', '.bmp')
CHILD_CASES_FOLDER = "child_cases"
CHILD_DATASET_FOLDER = "child_dataset"
CHILD_THUMB_FOLDER = "child_dataset/thumbs"

os.makedirs(MISSING_THUMB_FOLDER, exist_ok=True)
os.makedirs(CHILD_CASES_FOLDER, exist_ok=True)
os.makedirs(DEAD_CONFIRM_FOLDER, exist_ok=True)
os.makedirs(CHILD_CONFIRM_FOLDER, exist_ok=True)
os.makedirs(CHILD_DATASET_FOLDER, exist_ok=True)
os.makedirs(CHILD_THUMB_FOLDER, exist_ok=True)
os.makedirs(MISSING_SEARCH_CASE_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = MISSING_PERSON_FOLDER

# Global DEAD data structures
embeddings_data = {}
filename_map = {}
metadata_map = {}

#child data structures
child_embeddings_data = {}
child_filename_map = {}
child_metadata_map = {}

def load_dataset():
    """Initializes the dataset by loading embeddings and mapping thumbnails."""
    for person_id in os.listdir(MISSING_PERSON_FOLDER):
        person_folder = os.path.join(MISSING_PERSON_FOLDER, person_id)
        if not os.path.isdir(person_folder):
            continue

        metadata_path = os.path.join(person_folder, "metadata.json")
        embeddings = []
        first_image = None

        for file in sorted(os.listdir(person_folder)):
            file_lower = file.lower()
            # ---- EMBEDDINGS ----
            if file_lower.endswith(".npy"):
                emb_path = os.path.join(person_folder, file)
                emb = np.load(emb_path)
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)

            # ---- MULTI-FORMAT IMAGE SUPPORT ----
            elif file_lower.endswith(VALID_EXTENSIONS) and first_image is None:
                first_image = f"{person_id}/{file}"

        if embeddings:
            embeddings_data[person_id] = embeddings
            if first_image:
                filename_map[person_id] = first_image
            
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_map[person_id] = json.load(f)
            else:
                metadata_map[person_id] = {}

def load_child_dataset():
    """
    Loads child dataset embeddings + metadata.
    Structure identical to missing dataset.
    """

    for child_id in os.listdir(CHILD_DATASET_FOLDER):

        person_folder = os.path.join(
            CHILD_DATASET_FOLDER,
            child_id
        )

        if not os.path.isdir(person_folder):
            continue

        metadata_path = os.path.join(
            person_folder,
            "metadata.json"
        )

        embeddings = []

        first_image = None

        for file in sorted(os.listdir(person_folder)):

            file_lower = file.lower()

            # EMBEDDINGS
            if file_lower.endswith(".npy"):

                emb_path = os.path.join(
                    person_folder,
                    file
                )

                emb = np.load(emb_path)

                emb = emb / np.linalg.norm(emb)

                embeddings.append(emb)

            # IMAGES
            elif file_lower.endswith(VALID_EXTENSIONS) and first_image is None:

                first_image = f"{child_id}/{file}"

        if embeddings:

            child_embeddings_data[child_id] = embeddings

            if first_image:
                child_filename_map[child_id] = first_image

            if os.path.exists(metadata_path):

                with open(metadata_path, "r", encoding="utf-8") as f:
                    child_metadata_map[child_id] = json.load(f)

            else:
                child_metadata_map[child_id] = {}


def load_confirmed_dead_matches():
    """
    Loads all immutable confirmed-match evidence files.

    Returns:
        [
            {
                "match_id": "...",
                "case_id": "...",
                "person_id": "...",
                "name": "...",
                "similarity": "...",
                "timestamp": "..."
            }
        ]
    """

    records = []

    if not os.path.exists(DEAD_CONFIRM_FOLDER):
        return records

    try:

        for file in os.listdir(DEAD_CONFIRM_FOLDER):

            # --------------------------------------------
            # Only JSON evidence files
            # --------------------------------------------

            if not file.lower().endswith(".json"):
                continue

            path = os.path.join(
                DEAD_CONFIRM_FOLDER,
                file
            )

            if not os.path.isfile(path):
                continue

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                # ----------------------------------------
                # Optional validation
                # ----------------------------------------

                required_fields = [
                    "match_id",
                    "case_id",
                    "person_id",
                    "name",
                    "similarity",
                    "timestamp"
                ]

                if not all(
                    field in data
                    for field in required_fields
                ):
                    continue

                records.append(data)

            except Exception as e:

                print(
                    f"Failed loading confirmed "
                    f"match file {file}: {e}"
                )

        # ------------------------------------------------
        # Sort newest first
        # ------------------------------------------------

        records.sort(
            key=lambda x: x.get(
                "timestamp",
                ""
            ),
            reverse=True
        )

        return records

    except Exception as e:

        print(
            "load_confirmed_dead_matches error:",
            e
        )

        return []

def load_confirmed_child_matches():

    matches = []

    if not os.path.exists(CHILD_CONFIRM_FOLDER):
        return matches

    for file in os.listdir(CHILD_CONFIRM_FOLDER):

        if not file.endswith(".json"):
            continue

        path = os.path.join(CHILD_CONFIRM_FOLDER, file)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            match = {
                "match_id": data.get("match_id"),
                "child_id": data.get("child_id"),
                "similarity": float(data.get("similarity", 0)),
                "timestamp": data.get("timestamp"),
            }

            child_name = ""
            child_data = os.path.join(
                        CHILD_DATASET_FOLDER,
                        data.get("child_id"),
                    )
            
            if os.path.exists(child_data):
                with open(os.path.join(child_data, "metadata.json"), "r", encoding="utf-8") as f:
                    child_meta = json.load(f)

                child_name = child_meta.get("name")


            match["name"] = child_name
            
            # =====================================================
            # CASE TYPE 1: DIRECT UPLOAD (-1)
            # =====================================================
            if data.get("image_id"):

                # find saved image
                base = data["image_id"]

                possible_exts = [".jpg", ".jpeg"]

                img_url = None

                for ext in possible_exts:

                    candidate = os.path.join(
                        CHILD_CONFIRM_FOLDER,
                        base + ext
                    )

                    if os.path.exists(candidate):
                        img_url = "/" + candidate.replace("\\", "/")
                        break

                match["image"] = img_url

                match["case_id"] = "-1"

            # =====================================================
            # CASE TYPE 2: NORMAL CASE
            # =====================================================
            elif data.get("case_id"):

                case_id = data["case_id"]

                case_dir = os.path.join(
                    CHILD_CASES_FOLDER,
                    case_id
                )

                img_url = None

                if os.path.exists(case_dir):

                    images = [
                        f for f in os.listdir(case_dir)
                        if f.lower().endswith((".jpg", ".jpeg"))
                    ]

                    if images:

                        first_image = images[0]

                        img_url = f"/child_cases/{case_id}/{first_image}"

                match["image"] = img_url
                match["case_id"] = case_id

            matches.append(match)

        except Exception as e:
            print(f"[LOAD MATCH ERROR] {file}: {e}")

    # newest first
    matches.sort(
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )

    return matches

def bootstrap_existing_data():
    """
    Bootstrap existing filesystem evidence into the audit chain.

    IMPORTANT:
    - Only lightweight manifests are stored in-chain
    - Metadata files are NOT embedded in blocks
    - Metadata integrity is enforced via SHA256
    - Images are verified via SHA256
    """

    chain = load_chain()

    # Prevent duplicate bootstrapping
    if len(chain) > 1:
        return

    # ========================================================
    # DATASET PERSONS
    # ========================================================

    for person_id in os.listdir(MISSING_PERSON_FOLDER):

        person_folder = os.path.join(
            MISSING_PERSON_FOLDER,
            person_id
        )

        if not os.path.isdir(person_folder):
            continue

        metadata_path = os.path.join(
            person_folder,
            "metadata.json"
        )

        if not os.path.exists(metadata_path):
            continue

        try:

            metadata_hash = sha256_file(metadata_path)

            image_hashes = []
            embedding_hashes = []

            for file in sorted(os.listdir(person_folder)):

                file_lower = file.lower()

                full_path = os.path.join(
                    person_folder,
                    file
                )

                if file_lower.endswith(".jpg"):

                    image_hashes.append({
                        "file": file,
                        "sha256": sha256_file(full_path)
                    })

                elif file_lower.endswith(".npy"):

                    embedding_hashes.append({
                        "file": file,
                        "sha256": sha256_file(full_path)
                    })

            manifest = {
                "person_id": person_id,
                "metadata_sha256": metadata_hash,
                "images": image_hashes,
                "embeddings": embedding_hashes
            }

            add_block(
                "BOOTSTRAP_MISSING_PERSON",
                manifest
            )

        except Exception as e:

            print(
                f"Bootstrap failed for dataset person "
                f"{person_id}: {e}"
            )

    # ========================================================
    # CONFIRMED DEAD MATCHES
    # ========================================================

    if os.path.exists(DEAD_CONFIRM_FOLDER):

        for file in os.listdir(DEAD_CONFIRM_FOLDER):

            if not file.lower().endswith(".json"):
                continue

            evidence_path = os.path.join(
                DEAD_CONFIRM_FOLDER,
                file
            )

            if not os.path.isfile(evidence_path):
                continue

            try:

                # Match ID from filename
                match_id = os.path.splitext(file)[0]

                manifest = {
                    "match_id": match_id,
                    "sha256": sha256_file(
                        evidence_path
                    )
                }

                add_block(
                    "BOOTSTRAP_CONFIRMED_DEAD_MATCH",
                    manifest
                )

            except Exception as e:

                print(
                    f"Bootstrap failed for confirmed "
                    f"match evidence {file}: {e}"
                )

    # ========================================================
    # DEAD BODY SEARCH CASES
    # ========================================================

    if os.path.exists(MISSING_SEARCH_CASE_FOLDER):

        for case_id in os.listdir(
            MISSING_SEARCH_CASE_FOLDER
        ):

            case_folder = os.path.join(
                MISSING_SEARCH_CASE_FOLDER,
                case_id
            )

            if not os.path.isdir(case_folder):
                continue

            image_path = os.path.join(
                case_folder,
                "corpse.jpg"
            )

            if not os.path.exists(image_path):
                continue

            try:

                manifest = {
                    "case_id": case_id,
                    "images": [
                        {
                            "file": "corpse.jpg",
                            "sha256": sha256_file(
                                image_path
                            )
                        }
                    ]
                }

                add_block(
                    "BOOTSTRAP_MISSING_SEARCH_CASE",
                    manifest
                )

            except Exception as e:

                print(
                    f"Bootstrap failed for "
                    f"search case {case_id}: {e}"
                )

    # ========================================================
    # CHILD WITNESS CASES
    # ========================================================

    if os.path.exists(CHILD_CASES_FOLDER):

        for case_id in os.listdir(CHILD_CASES_FOLDER):

            case_folder = os.path.join(
                CHILD_CASES_FOLDER,
                case_id
            )

            if not os.path.isdir(case_folder):
                continue

            metadata_path = os.path.join(
                case_folder,
                "metadata.json"
            )

            status_path = os.path.join(
                case_folder,
                "status.json"
            )

            if not os.path.exists(metadata_path):
                continue

            if not os.path.exists(status_path):
                continue

            try:

                # ----------------------------------------
                # Metadata hash
                # ----------------------------------------

                metadata_hash = sha256_file(
                    metadata_path
                )

                # ----------------------------------------
                # Image hashes
                # ----------------------------------------

                image_hashes = []

                for file in sorted(os.listdir(case_folder)):

                    if not file.lower().endswith(".jpg"):
                        continue

                    image_path = os.path.join(
                        case_folder,
                        file
                    )

                    image_hashes.append({
                        "file": file,
                        "sha256": sha256_file(
                            image_path
                        )
                    })

                # ----------------------------------------
                # Original case manifest
                # ----------------------------------------

                manifest = {
                    "case_id": case_id,
                    "metadata_sha256": metadata_hash,
                    "images": image_hashes
                }

                add_block(
                    "BOOTSTRAP_CHILD_WITNESS_CASE",
                    manifest
                )

                # ----------------------------------------
                # Bootstrap status manifest
                # ----------------------------------------

                with open(
                    status_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    status_data = json.load(f)

                status_manifest = {
                    "case_id": case_id,
                    "status": status_data["status"],
                    "status_sha256": sha256_file(
                        status_path
                    )
                }

                add_block(
                    "CHILD_CASE_STATUS_INITIAL",
                    status_manifest
                )

            except Exception as e:

                print(
                    f"Bootstrap failed for child "
                    f"case {case_id}: {e}"
                )

    # ========================================================
    # CHILD DATASET
    # ========================================================

    if os.path.exists(CHILD_DATASET_FOLDER):

        for child_id in os.listdir(CHILD_DATASET_FOLDER):

            child_folder = os.path.join(
                CHILD_DATASET_FOLDER,
                child_id
            )

            if not os.path.isdir(child_folder):
                continue

            metadata_path = os.path.join(
                child_folder,
                "metadata.json"
            )

            if not os.path.exists(metadata_path):
                continue

            try:

                metadata_hash = sha256_file(
                    metadata_path
                )

                image_hashes = []
                embedding_hashes = []

                for file in sorted(os.listdir(child_folder)):

                    file_lower = file.lower()

                    full_path = os.path.join(
                        child_folder,
                        file
                    )

                    if file_lower.endswith(".jpg"):

                        image_hashes.append({
                            "file": file,
                            "sha256": sha256_file(
                                full_path
                            )
                        })

                    elif file_lower.endswith(".npy"):

                        embedding_hashes.append({
                            "file": file,
                            "sha256": sha256_file(
                                full_path
                            )
                        })

                manifest = {
                    "child_id": child_id,
                    "metadata_sha256": metadata_hash,
                    "images": image_hashes,
                    "embeddings": embedding_hashes
                }

                add_block(
                    "BOOTSTRAP_MISSING_CHILD",
                    manifest
                )

            except Exception as e:

                print(
                    f"Bootstrap failed for child "
                    f"dataset person {child_id}: {e}"
                )
    # ========================================================
    # CONFIRMED CHILD MATCHES
    # ========================================================

    if os.path.exists(CHILD_CONFIRM_FOLDER):

        for file in os.listdir(CHILD_CONFIRM_FOLDER):

            if not file.lower().endswith(".json"):
                continue

            evidence_path = os.path.join(
                CHILD_CONFIRM_FOLDER,
                file
            )

            if not os.path.isfile(evidence_path):
                continue

            try:

                match_id = os.path.splitext(
                    file
                )[0]

                with open(
                    evidence_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    record = json.load(f)

                manifest = {

                    "match_id": match_id,

                    "evidence_sha256": sha256_file(
                        evidence_path
                    )
                }

                # =====================================
                # DIRECT IMAGE SEARCH MATCH
                # =====================================

                if "image_id" in record:

                    uploaded_image_path = os.path.join(

                        CHILD_CONFIRM_FOLDER,

                        f'{record["image_id"]}.jpg'
                    )

                    if os.path.exists(
                        uploaded_image_path
                    ):

                        manifest[
                            "uploaded_image_sha256"
                        ] = sha256_file(
                            uploaded_image_path
                        )

                add_block(

                    "BOOTSTRAP_CONFIRMED_CHILD_MATCH",

                    manifest
                )

            except Exception as e:

                print(
                    f"Bootstrap failed for confirmed "
                    f"child match evidence {file}: {e}"
                )

# Initial Load
load_dataset()
load_child_dataset()
bootstrap_existing_data()

# def save_confirmed_match(record):
#     """
#     Saves immutable evidence artifact for a confirmed match.
#     Returns:
#         match_id,
#         evidence_path
#     """

#     timestamp = datetime.utcnow().strftime(
#         "%Y%m%d%H%M%S"
#     )

#     random_suffix = hashlib.sha256(
#         os.urandom(32)
#     ).hexdigest()[:10]

#     match_id = f"{timestamp}_{random_suffix}"

#     record["match_id"] = match_id

#     evidence_path = os.path.join(
#         DEAD_CONFIRM_FOLDER,
#         f"{match_id}.json"
#     )

#     with open(
#         evidence_path,
#         "w",
#         encoding="utf-8"
#     ) as f:

#         json.dump(
#             record,
#             f,
#             indent=4,
#             ensure_ascii=False
#         )

#     return match_id, evidence_path

# Load Adapter Pipeline
try:
    pipeline = joblib.load("arcface_adapter.pkl")
    adapter = pipeline["model"]
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    print("Adapter pipeline loaded")
except Exception as e:
    print("Adapter load failed:", e)
    adapter = scaler = pca = None

def get_image_path(base_name):
    if base_name in filename_map:
        return os.path.join(MISSING_PERSON_FOLDER, filename_map[base_name])
    raise FileNotFoundError(f"No image found for {base_name}")

def make_dead_thumb(base_name):
    img_path = get_image_path(base_name)
    relative = filename_map[base_name]
    thumb_path = os.path.join(MISSING_THUMB_FOLDER, *relative.split("/"))
    
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    if not os.path.exists(thumb_path):
        with Image.open(img_path) as img:
            img = img.convert("RGB") # Ensure consistency
            img.thumbnail((100, 100), Image.Resampling.LANCZOS)
            img.save(thumb_path, "JPEG")
    return thumb_path

def make_child_thumb(src_path, filename):

    try:
        img = Image.open(src_path).convert("RGB")
        img.thumbnail((300, 300))
        thumb_path = os.path.join(
            CHILD_THUMB_FOLDER,
            os.path.splitext(filename)[0] + ".jpg"
        )
        os.makedirs(
            os.path.dirname(thumb_path),
            exist_ok=True
        )
        img.save(
            thumb_path,
            "JPEG",
            quality=90
        )
        return thumb_path

    except Exception as e:
        print(f"[CHILD THUMB ERROR] {e}")
        return None

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def get_child_pending_cases():

    cases = []

    if not os.path.exists(CHILD_CASES_FOLDER):
        print("no folder")
        return cases

    for case_id in os.listdir(CHILD_CASES_FOLDER):

        case_dir = os.path.join(
            CHILD_CASES_FOLDER,
            case_id
        )

        if not os.path.isdir(case_dir):
            print("not dir")
            continue

        metadata_path = os.path.join(
            case_dir,
            "metadata.json"
        )

        status_path = os.path.join(
            case_dir,
            "status.json"
        )

        if not os.path.exists(metadata_path):
            print("no metadata")
            continue

        try:

            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)

            if status.get("status") != "pending":
                continue

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            child_img = None

            for img_name in metadata.get("images", []):

                img_path = os.path.join(
                    case_dir,
                    img_name
                )

                if os.path.exists(img_path):

                    child_img = img_path
                    break

            if child_img is None:
                continue

            cases.append({
                "case_id": case_id,

                "name": metadata.get("name"),

                "phone": metadata.get("phone"),

                "address": metadata.get("address"),

                "last_seen": (
                    f"{metadata.get('last_seen', {}).get('area', '')}, "
                    f"{metadata.get('last_seen', {}).get('city', '')}, "
                    f"{metadata.get('last_seen', {}).get('state', '')} "
                    f"({metadata.get('last_seen', {}).get('date', '')})"
                ),

                "event_description": metadata.get("event_description"),

                "created_at": metadata.get(
                    "created_at"
                ),

                "image": "/"+ child_img.replace("\\", "/")
            })

        except Exception as e:
            print(
                f"[PENDING CASE LOAD ERROR] {case_id}: {e}"
            )

    return cases

def get_confirmed_children():

    matches = []

    if not os.path.exists(CHILD_CONFIRM_FOLDER):
        return matches

    for file in os.listdir(CHILD_CONFIRM_FOLDER):

        if not file.endswith(".json"):
            continue

        path = os.path.join(CHILD_CONFIRM_FOLDER, file)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # normalize fields (safe defaults)
            matches.append({
                "match_id": data.get("match_id"),
                "child_id": data.get("child_id"),
                "similarity": float(data.get("similarity", 0)),
                "timestamp": data.get("timestamp")
            })

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # newest first
    matches.sort(
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )

    return matches

@app.route("/", methods=["GET","POST"])
def index():
    active_tab = "search"
    uploaded_image = None
    uploaded_metadata = None
    sorted_dataset_images = list(embeddings_data.keys())
    sims = {}
    case_id = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            case_id = datetime.now().strftime("%Y%m%d%H%M%S")
            case_folder = os.path.join(MISSING_SEARCH_CASE_FOLDER, case_id)
            os.makedirs(case_folder, exist_ok=True)
            
            upload_path = os.path.join(case_folder, "corpse.jpg")
            
            # Convert any uploaded format to standard JPEG
            img = Image.open(file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(upload_path, "JPEG", quality=95)

            try:

                manifest = {
                    "case_id": case_id,
                    "images": [
                        {
                            "file": "corpse.jpg",
                            "sha256": sha256_file(
                                upload_path
                            )
                        }
                    ]
                }

                add_block(
                    "MISSING_SEARCH_CASE",
                    manifest
                )

            except Exception as e:

                print(
                    f"Blockchain hashing failed for "
                    f"search case {case_id}: {e}"
                )

            uploaded_image = url_for('case_file', filename=f"{case_id}/corpse.jpg")

            rep = DeepFace.represent(
                img_path=upload_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            upload_emb = np.array(rep[0]["embedding"])
            upload_emb = upload_emb / np.linalg.norm(upload_emb)

            # Analysis for metadata
            face = rep[0]["facial_area"]
            x,y,w,h = face["x"], face["y"], face["w"], face["h"]
            face_crop = np.array(img.crop((x,y,x+w,y+h)))

            analysis = DeepFace.analyze(
                img_path=face_crop,
                actions=["age","gender","race"],
                detector_backend="retinaface",
                enforce_detection=False
            )[0]

            # Normalization logic
            gender_map = {"man": "male", "woman": "female"}
            gender = gender_map.get(analysis["dominant_gender"].lower(), analysis["dominant_gender"].lower())
            
            race_to_skin = {
                "white": "fair", "latino hispanic": "fair",
                "middle eastern": "medium", "asian": "medium",
                "indian": "medium", "black": "dark"
            }

            uploaded_metadata = {
                "age": analysis["age"],
                "gender": gender,
                "race": analysis["dominant_race"],
                "skin_tone": race_to_skin.get(analysis["dominant_race"], "unknown")
            }

            # Similarity Computation
            for pid, emb_list in embeddings_data.items():
                scores = [cosine_sim(upload_emb, e) for e in emb_list]
                scores.sort(reverse=True)
                cos_score = np.mean(scores[:3]) if len(scores) >= 3 else np.mean(scores)
                cos_score = (cos_score + 1) / 2 # Scale 0 to 1

                if adapter:
                    fv = np.concatenate([upload_emb, upload_emb**2, np.abs(upload_emb)])
                    fv = scaler.transform([fv])
                    fv = pca.transform(fv)
                    probs = adapter.predict_proba(fv)[0]
                    labels = list(adapter.classes_)
                    prob = probs[labels.index(pid)] if pid in labels else 0
                    
                    combined = 0.6 * cos_score + 0.4 * prob
                    sims[pid] = combined / (0.5 + 0.5 * combined)
                else:
                    sims[pid] = cos_score

            sorted_dataset_images = sorted(sims, key=lambda x: sims[x], reverse=True)

    for f in sorted_dataset_images:
        try:
            make_dead_thumb(f)
        except Exception as e:
            print("Thumbnail error:", e)

    matches = load_confirmed_dead_matches()

    childmatches = load_confirmed_child_matches()


    return render_template(
        "index.html",
        uploaded_image=uploaded_image,
        child_pending_cases=get_child_pending_cases(),
        child_matches=childmatches,
        child_filename_map=child_filename_map,
        images=sorted_dataset_images,
        case_id=case_id,
        filename_map=filename_map,
        sims=sims,
        metadata_map=metadata_map,
        uploaded_metadata=uploaded_metadata,
        active_tab=active_tab,
        matches=matches,
        child_case_id=-1
    )

@app.route("/add_missing", methods=["POST"])
def add_missing():
    files = request.files.getlist("person_images")
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    person_folder = os.path.join(MISSING_PERSON_FOLDER, unique_id)
    os.makedirs(person_folder, exist_ok=True)

    try:
        embeddings = []
        first_valid_image = None

        for idx, file in enumerate(files):
            if not file or file.filename == "": continue
            
            img_filename = f"{unique_id}_{idx}.jpg"
            img_path = os.path.join(person_folder, img_filename)

            # --- KEY FIX: Standardize to RGB JPEG ---
            image = Image.open(file)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path, "JPEG", quality=95)

            rep = DeepFace.represent(
                img_path=img_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            emb = np.array(rep[0]["embedding"])
            emb = emb / np.linalg.norm(emb)
            np.save(os.path.join(person_folder, f"{unique_id}_{idx}.npy"), emb)
            embeddings.append(emb)

            if first_valid_image is None:
                first_valid_image = f"{unique_id}/{img_filename}"

        if not embeddings:
            raise Exception("No valid faces detected")

        embeddings_data[unique_id] = embeddings
        filename_map[unique_id] = first_valid_image

        # Process metadata
        mark_types = request.form.getlist("mark_type[]")
        body_parts = request.form.getlist("body_part[]")
        descriptions = request.form.getlist("description[]")

        height_min = request.form.get("height_min")
        height_max = request.form.get("height_max")
        skin_tone = request.form.get("skin_tone")
        hair_color = request.form.get("hair_color")
        facial_hair = request.form.get("facial_hair")
        last_state = request.form.get("last_state")
        last_city = request.form.get("last_city")
        last_area = request.form.get("last_area")
        last_seen_date = request.form.get("last_seen_date")
        
        metadata = {
            "unique_id": unique_id,
            "name": request.form.get("name"),
            "dob": request.form.get("dob"),
            "gender": request.form.get("gender"),
            "height_range_cm": {
                "min": height_min,
                "max": height_max
            },
            "skin_tone": skin_tone,
            "hair_color": hair_color,
            "facial_hair": facial_hair,
            "marks": [{"type": t, "body_part": b, "description": d} for t, b, d in zip(mark_types, body_parts, descriptions)],
            "last_seen": {
                "state": last_state,
                "city": last_city,
                "area": last_area,
                "date": last_seen_date
            },
            "created_at": datetime.now().isoformat()
        }

        metadata_path = os.path.join(person_folder, "metadata.json")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        try:

            metadata_hash = sha256_file(metadata_path)

            image_hashes = []
            embedding_hashes = []

            for file in sorted(os.listdir(person_folder)):

                file_lower = file.lower()

                full_path = os.path.join(
                    person_folder,
                    file
                )


                # IMAGE HASHES
                if file_lower.endswith(".jpg"):

                    image_hashes.append({
                        "file": file,
                        "sha256": sha256_file(full_path)
                    })

                # EMBEDDING HASHES
                elif file_lower.endswith(".npy"):

                    embedding_hashes.append({
                        "file": file,
                        "sha256": sha256_file(full_path)
                    })

            manifest = {
                "person_id": unique_id,
                "metadata_sha256": metadata_hash,
                "images": image_hashes,
                "embeddings": embedding_hashes
            }

            add_block(
                "MISSING_PERSON",
                manifest
            )

        except Exception as e:

            print(
                f"Blockchain hashing failed for "
                f"missing dataset person {unique_id}: {e}"
            )

        metadata_map[unique_id] = metadata
        os.system("py train_adapter.py")

    except Exception as e:
        print("Error in add_missing:", e)

    return redirect(url_for("index"))

@app.route("/submit_child_witness", methods=["POST"])
def submit_child_witness():

    files = request.files.getlist("child_images")
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    case_folder = os.path.join(CHILD_CASES_FOLDER, unique_id)
    os.makedirs(case_folder, exist_ok=True)

    try:

        saved_images = []
        image_hashes = []

        for idx, file in enumerate(files):

            if not file or file.filename == "":
                continue

            filename = f"{unique_id}_{idx}.jpg"

            image_path = os.path.join(case_folder, filename)

            image = Image.open(file)

            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(image_path, "JPEG", quality=90)

            saved_images.append(filename)

            image_hashes.append({
                "file": filename,
                "sha256": sha256_file(
                    image_path
                )
            })

        metadata = {
            "case_id": unique_id,
            "name": request.form.get("name"),
            "phone": request.form.get("phone"),
            "address": request.form.get("address"),
            "event_description": request.form.get("event_description"),

            "last_seen": {
                "state": request.form.get("last_state"),
                "city": request.form.get("last_city"),
                "area": request.form.get("last_area"),
                "date": request.form.get("last_seen_date")
            },

            "created_at": datetime.now().isoformat(),

            "images": saved_images,
        }

        metadata_path = os.path.join(case_folder, "metadata.json")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        status_data = {

            "status": "pending",

            "updated_at": datetime.now().isoformat()

        }

        status_path = os.path.join(
            case_folder,
            "status.json"
        )

        with open(
            status_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                status_data,
                f,
                indent=4,
                ensure_ascii=False
            )

        
        metadata_hash = sha256_file(metadata_path)
    
        manifest = {
            "case_id": unique_id,
            "metadata_sha256": metadata_hash,
            "images": image_hashes
        }

        add_block(
            "CHILD_WITNESS_CASE",
            manifest
        )

        add_block(
            "CHILD_CASE_STATUS_INITIAL",
            {

                "case_id": unique_id,

                "status": "pending",

                "status_sha256": sha256_file(
                    status_path
                )

            }
        )

        return redirect(url_for("index"))

    except Exception as e:
        print("Child witness submission error:", e)
        return "Upload failed", 500

# --- Static File Serving Routes ---
@app.route("/dataset/<path:filename>")
def dataset_file(filename): return send_from_directory(MISSING_PERSON_FOLDER, filename)

@app.route('/cases/<path:filename>')
def case_file(filename): return send_from_directory(MISSING_SEARCH_CASE_FOLDER, filename)

@app.route('/child_cases/<path:filename>')
def child_case_file(filename): return send_from_directory(CHILD_CASES_FOLDER, filename)

@app.route("/dataset/thumbs/<path:filename>")
def thumb_file(filename): return send_from_directory(MISSING_THUMB_FOLDER, filename)

@app.route("/child_dataset/<path:filename>")
def child_dataset_file(filename):

    return send_from_directory(
        CHILD_DATASET_FOLDER,
        filename
    )

@app.route("/storage/confirmed_child_matches/<path:filename>")
def confirmed_child_matches_file(filename):
    filename = os.path.basename(filename)

    if not filename.lower().endswith((".jpg", ".jpeg")):
        abort(404)
    return send_from_directory(
        "storage/confirmed_child_matches",
        filename
    )

@app.route("/child_thumbnails/<path:filename>")
def child_thumb_file(filename):

    return send_from_directory(
        CHILD_THUMB_FOLDER,
        filename
    )

@app.route("/confirm_dead_match", methods=["POST"])
def confirm_dead_match():

    data = request.get_json()
    matches = data.get("matches", [])
    case_id = data.get("case_id")
    

    if not case_id:
        return {
            "status": "error",
            "message": "Missing case_id"
        }, 400

    saved = []
    try:

        for m in matches:
            person_id = m.get("person_id")

            if not person_id:
                continue

            meta = metadata_map.get(person_id,{})

            # ============================================
            # BUILD EVIDENCE RECORD
            # ============================================

            match_id = (
                datetime.utcnow().strftime(
                    "%Y%m%d%H%M%S"
                )
                + "_"
                + hashlib.sha256(
                    os.urandom(32)
                ).hexdigest()[:8]
            )

            record = {
                "match_id": match_id,
                "case_id": case_id,
                "person_id": person_id,
                "name": meta.get(
                    "name",
                    "Unknown"
                ),
                "similarity": m.get(
                    "similarity"
                ),
                "timestamp": datetime.utcnow().isoformat()
            }

            # ============================================
            # SAVE EVIDENCE FILE
            # ============================================

            evidence_path = os.path.join(
                DEAD_CONFIRM_FOLDER,
                f"{match_id}.json"
            )

            with open(
                evidence_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    record,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

            # ============================================
            # HASH EVIDENCE
            # ============================================

            evidence_hash = sha256_file(
                evidence_path
            )

            # ============================================
            # STORE MINIMAL MANIFEST IN CHAIN
            # ============================================

            add_block(
                "CONFIRMED_DEAD_MATCH",
                {
                    "match_id": match_id,
                    "sha256": evidence_hash
                }
            )

            saved.append(match_id)

        return {
            "status": "ok",
            "matches": saved
        }

    except Exception as e:

        print(
            "confirm_dead_match error:",
            e
        )

        return {
            "status": "error",
            "message": str(e)
        }, 500

@app.route("/confirm_child_match", methods=["POST"])
def confirm_child_match():

    data = request.get_json()
    matches = data.get("matches", [])
    case_id = data.get("case_id")

    image_id = None
    uploaded_image = data.get("uploaded_image")
    

    if case_id is None:
        return {
            "status": "error",
            "message": "Missing case_id"
        }, 400


    if str(case_id) == "-1" and uploaded_image:

        image_id = (
            datetime.utcnow().strftime("%Y%m%d%H%M%S")
            + "_"
            + hashlib.sha256(os.urandom(32)).hexdigest()[:8]
        )

        try:
            filename = os.path.basename(uploaded_image)

            clean_path = uploaded_image.replace("/static/", "").lstrip("/")
            src_path = os.path.join("static", clean_path)

            # open image
            image = Image.open(src_path)

            # force RGB (important for PNG/alpha images)
            image = image.convert("RGB")

            dest_path = os.path.join(
                CHILD_CONFIRM_FOLDER,
                f"{image_id}.jpg"
            )

            # save as JPG
            image.save(dest_path, format="JPEG", quality=95)

        except Exception as img_err:
            print("Image save error:", img_err)

    saved = []

    try:

        for m in matches:

            child_id = m.get("child_id")

            if not child_id:
                continue

            meta = child_metadata_map.get(
                child_id,
                {}
            )

            # ============================================
            # BUILD EVIDENCE RECORD
            # ============================================

            match_id = (
                datetime.utcnow().strftime(
                    "%Y%m%d%H%M%S"
                )
                + "_"
                + hashlib.sha256(
                    os.urandom(32)
                ).hexdigest()[:8]
            )

            record = {
                "match_id": match_id,
                "child_id": child_id,
                "similarity": m.get(
                    "similarity"
                ),
                "timestamp": datetime.utcnow().isoformat()
            }

            if str(case_id) == "-1" and image_id:
                record["image_id"] = image_id

            else:
                record["case_id"] = case_id

            # ============================================
            # SAVE EVIDENCE FILE
            # ============================================

            evidence_path = os.path.join(
                CHILD_CONFIRM_FOLDER,
                f"{match_id}.json"
            )

            with open(
                evidence_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    record,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

            # ============================================
            # HASH EVIDENCE
            # ============================================

            evidence_hash = sha256_file(
                evidence_path
            )

            # ============================================
            # STORE MANIFEST IN CHAIN
            # ============================================

            manifest = {
                "match_id": match_id,
                "evidence_sha256": evidence_hash
            }

            if image_id:

                uploaded_image_path = os.path.join(
                    CHILD_CONFIRM_FOLDER,
                    f"{image_id}.jpg"
                )

                manifest["uploaded_image_sha256"] = sha256_file(
                    uploaded_image_path
                )

            add_block(
                "CONFIRMED_CHILD_MATCH",
                manifest
            )

            saved.append(match_id)
            
        # =====================================================
        # UPDATE STATUS FILE
        # =====================================================
        if str(case_id) != "-1":

            case_folder = os.path.join(
                CHILD_CASES_FOLDER,
                case_id
            )

            status_path = os.path.join(
                case_folder,
                "status.json"
            )

            status_data = {

                "status": "matched",

                "updated_at": (
                    datetime.utcnow().isoformat()
                )

            }

            with open(

                status_path,
                "w",
                encoding="utf-8"

            ) as f:

                json.dump(

                    status_data,
                    f,
                    indent=4,
                    ensure_ascii=False

                )

            # =====================================================
            # STORE STATUS UPDATE BLOCK
            # =====================================================

            add_block(

                "CHILD_CASE_STATUS_UPDATED",

                {

                    "case_id": case_id,

                    "status": "matched",

                    "status_sha256": sha256_file(
                        status_path
                    )

                }

            )
        return jsonify({
            "status": "ok",
            "matches": saved
        })

    except Exception as e:

        print(
            "confirm_child_match error:",
            e
        )

        return {
            "status": "error",
            "message": str(e)
        }, 500

@app.route("/get_confirmed_dead_matches")
def get_confirmed_dead_matches():

    return {
        "matches": load_confirmed_dead_matches(),
        "filename_map": filename_map
    }

@app.route("/get_confirmed_child_matches")
def get_confirmed_child_matches():

    return {
        "child_matches": load_confirmed_child_matches(),
        "child_filename_map": child_filename_map
    }
@app.route("/get_pending_child_cases")
def send_pending_child_matches():

    return {
        "child_pending_cases": get_child_pending_cases()
    }

@app.route("/child_upload")
def child_upload():
    return render_template("child_upload.html")

@app.route("/add_child_missing", methods=["POST"])
def add_child_missing():

    files = request.files.getlist("child_images")

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    child_folder = os.path.join(CHILD_DATASET_FOLDER,unique_id)
    os.makedirs(child_folder,exist_ok=True)

    try:
        embeddings = []
        first_valid_image = None

        for idx, file in enumerate(files):

            if not file or file.filename == "":
                continue

            img_filename = f"{unique_id}_{idx}.jpg"
            img_path = os.path.join(child_folder,img_filename)

            image = Image.open(file)

            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path,"JPEG",quality=95)

            rep = DeepFace.represent(
                img_path=img_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            emb = np.array(rep[0]["embedding"])
            emb = emb / np.linalg.norm(emb)

            np.save(os.path.join(child_folder,f"{unique_id}_{idx}.npy"),emb)

            embeddings.append(emb)

            if first_valid_image is None:
                first_valid_image = (f"{unique_id}/{img_filename}")

        if not embeddings:
            raise Exception("No valid faces detected")

        child_embeddings_data[unique_id] = embeddings
        child_filename_map[unique_id] = first_valid_image

        # ==========================================
        # CHILD METADATA
        # ==========================================

        mark_types = request.form.getlist("mark_type[]")
        body_parts = request.form.getlist("body_part[]")
        descriptions = request.form.getlist("description[]") 

        metadata = {
            "unique_id": unique_id,
            "name": request.form.get("name"),
            "dob": request.form.get("dob"),
            "gender": request.form.get("gender"),
            "skin_tone": request.form.get("skin_tone"),
            "hair_color":  request.form.get("hair_color"),
            "marks": [{"type": t, "body_part": b, "description": d} for t, b, d in zip(mark_types, body_parts, descriptions)],
            "last_seen": {
                "state": request.form.get("last_state"),
                "city": request.form.get("last_city"),
                "area": request.form.get("last_area"),
                "date": request.form.get("last_seen_date")
            },
            "created_at": datetime.now().isoformat()
        }

        metadata_path = os.path.join(child_folder,"metadata.json")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata,f,indent=4)

        try:

            metadata_hash = sha256_file(metadata_path)

            image_hashes = []
            embedding_hashes = []

            for file in sorted(os.listdir(child_folder)):

                file_lower = file.lower()

                full_path = os.path.join(
                    child_folder,
                    file
                )

                if file_lower.endswith(".jpg"):

                    image_hashes.append({
                        "file": file,
                        "sha256": sha256_file(full_path)
                    })

                elif file_lower.endswith(".npy"):

                    embedding_hashes.append({
                        "file": file,
                        "sha256": sha256_file(full_path)
                    })

            manifest = {
                "child_id": unique_id,
                "metadata_sha256": metadata_hash,
                "images": image_hashes,
                "embeddings": embedding_hashes
            }

            add_block(
                "MISSING_CHILD",
                manifest
            )

        except Exception as e:

            print(
                f"Blockchain hashing failed for "
                f"missing child {unique_id}: {e}"
            )
    
        child_metadata_map[unique_id] = metadata
        return redirect(url_for("index"))

    except Exception as e:
        print("add_child_missing error:",e)
        return "Upload failed", 500

def run_child_search(image_source):

    # =========================================
    # LOAD IMAGE
    # =========================================

    if isinstance(image_source, str):

        image = Image.open(image_source)

    else:

        image = image_source

    if image.mode != "RGB":
        image = image.convert("RGB")

    # =========================================
    # EMBEDDING
    # =========================================

    rep = DeepFace.represent(
        img_path=np.array(image),
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=False
    )

    upload_emb = np.array(
        rep[0]["embedding"]
    )

    upload_emb = (
        upload_emb /
        np.linalg.norm(upload_emb)
    )

    # =========================================
    # SIMILARITY SEARCH
    # =========================================

    sims = {}

    for child_id, emb_list in child_embeddings_data.items():

        scores = [
            cosine_sim(upload_emb, e)
            for e in emb_list
        ]

        scores.sort(reverse=True)

        sim = (
            np.mean(scores[:3])
            if len(scores) >= 3
            else np.mean(scores)
        )

        sim = (sim + 1) / 2

        sims[child_id] = float(sim)

    # =========================================
    # SORT
    # =========================================

    sorted_children = sorted(
        sims,
        key=lambda x: sims[x],
        reverse=True
    )

    # =========================================
    # THUMBNAILS
    # =========================================

    for cid in sorted_children:

        try:

            src = os.path.join(
                CHILD_DATASET_FOLDER,
                child_filename_map[cid]
            )

            filename = child_filename_map[cid]

            make_child_thumb(
                src,
                filename
            )

        except Exception as e:

            print(
                "Child thumbnail error:",
                e
            )

    return sims, sorted_children

@app.route("/search_child", methods=["POST"])
def search_child():

    active_tab = "IDchild"

    file = request.files.get("child_file")

    if not file or file.filename == "":
        return redirect(url_for("index"))

    try:

        image = Image.open(file)

        sims, sorted_children = run_child_search(image)

        temp_filename = secure_filename(file.filename)

        temp_path = os.path.join(
            "static",
            "temp",
            temp_filename
        )

        os.makedirs("static/temp", exist_ok=True)

        image.save(temp_path)

        return render_template(
            "index.html",
            active_tab=active_tab,
            child_pending_cases=get_child_pending_cases(),
            child_images=sorted_children,
            child_uploaded_image=url_for("static", filename=f"temp/{temp_filename}"),
            child_filename_map=child_filename_map,
            child_metadata_map=child_metadata_map,
            child_sims=sims,
            child_case_id=-1
        )

    except Exception as e:

        print(
            "search_child error:",
            e
        )

        return "Search failed", 500

@app.route("/search_child_case/<case_id>")
def search_child_case(case_id):

    try:

        case_dir = os.path.join(
            CHILD_CASES_FOLDER,
            case_id
        )

        metadata_path = os.path.join(
            case_dir,
            "metadata.json"
        )

        if not os.path.exists(metadata_path):
            return "Metadata not found", 404

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        image_filename = None
        image_path = None

        for img_name in metadata.get("images", []):

            candidate = os.path.join(
                case_dir,
                img_name
            )

            if os.path.exists(candidate):

                image_filename = img_name
                image_path = candidate
                break

        if image_path is None:
            return "Child image not found", 404

        sims, sorted_children = run_child_search(image_path)

        return render_template(
            "index.html",
            active_tab="IDchild",
            child_pending_cases=get_child_pending_cases(),
            child_images=sorted_children,
            child_uploaded_image=f"/child_cases/{case_id}/{image_filename}",
            child_filename_map=child_filename_map,
            child_metadata_map=child_metadata_map,
            child_sims=sims,
            child_case_id=case_id
        )

    except Exception as e:

        print("search_child_case error:", e)

        return "Search failed", 500

# @app.route("/search_child_case/<case_id>")
# def search_child_case(case_id):

#     try:

#         case_dir = os.path.join(
#             CHILD_CASES_FOLDER,
#             case_id
#         )

#         metadata_path = os.path.join(
#             case_dir,
#             "metadata.json"
#         )

#         if not os.path.exists(metadata_path):
#             return "Metadata not found", 404

#         with open(metadata_path, "r") as f:
#             metadata = json.load(f)

#         image_filename = None

#         for img_name in metadata.get("images", []):

#             img_path = os.path.join(
#                 case_dir,
#                 img_name
#             )

#             if os.path.exists(img_path):

#                 image_filename = img_name
#                 image_path = img_path
#                 break

#         if image_filename is None:
#             return "Child image not found", 404

#         sims = search_child(image_path)

#         child_images = list(sims.keys())

#         return render_template(
#             "index.html",

#             child_sims=sims,

#             child_images=child_images,

#             child_uploaded_image=
#                 f"/child_cases/{case_id}/{image_filename}",

#             child_case_id=case_id,

#             active_tab="IDchild",

#             child_pending_cases=
#                 get_child_pending_cases(),

#             child_filename_map=
#                 load_child_filename_map(),

#             child_metadata_map=
#                 load_child_metadata_map()
#         )

#     except Exception as e:
#         print("search_child_case error:", e)
#         return str(e), 500
    

@app.route("/verify_audit")
def verify_audit():
    return {
        "chain_valid": verify_chain(),
        "confirmed_dead_valid": verify_dead_matches(),
        "missing_search_cases_valid": verify_missing_search_cases(),
        "missing_dataset_valid": verify_missing_dataset(),
        "child_cases_valid": verify_child_cases(),
        "child_dataset_valid": verify_child_dataset(),
        "child_matches_valid": verify_child_matches()
    }

if __name__ == "__main__":
    app.run(debug=True)