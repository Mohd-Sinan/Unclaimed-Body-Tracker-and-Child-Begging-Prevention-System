import json
import hashlib
import os
from datetime import datetime


BLOCKCHAIN_FILE = "storage/audit_chain.json"
MISSING_CHILD_FOLDER = "child_dataset"
MISSING_PERSON_FOLDER = "dataset"
MISSING_SEARCH_CASE_FOLDER = "missing_search_cases"
CHILD_CASES_FOLDER = "child_cases"
DEAD_CONFIRM_FOLDER = "storage/confirmed_dead_matches"
CHILD_CONFIRM_FOLDER = "storage/confirmed_child_matches"
os.makedirs("storage", exist_ok=True)


def sha256(data):

    return hashlib.sha256(
        data.encode()
    ).hexdigest()


def sha256_file(path):

    with open(path, "rb") as f:

        return hashlib.sha256(
            f.read()
        ).hexdigest()


def load_chain():

    if not os.path.exists(BLOCKCHAIN_FILE):

        genesis = [{
            "index": 0,
            "timestamp": datetime.now().isoformat(),
            "action": "GENESIS",
            "data_hash": "0",
            "prev_hash": "0",
            "hash": "0"
        }]

        with open(
            BLOCKCHAIN_FILE,
            "w"
        ) as f:

            json.dump(
                genesis,
                f,
                indent=4
            )

    with open(BLOCKCHAIN_FILE) as f:
        return json.load(f)


def add_block(action, data):

    chain = load_chain()

    prev_block = chain[-1]

    data_string = json.dumps(
        data,
        sort_keys=True
    )

    data_hash = sha256(data_string)

    block = {
        "index": len(chain),
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "data_hash": data_hash,
        "prev_hash": prev_block["hash"]
    }

    block_string = json.dumps(
        block,
        sort_keys=True
    )

    block["hash"] = sha256(block_string)

    chain.append(block)

    with open(
        BLOCKCHAIN_FILE,
        "w"
    ) as f:

        json.dump(
            chain,
            f,
            indent=4
        )


def verify_chain():

    chain = load_chain()

    for i in range(1, len(chain)):

        current = chain[i]

        previous = chain[i - 1]

        test_block = {
            "index": current["index"],
            "timestamp": current["timestamp"],
            "action": current["action"],
            "data_hash": current["data_hash"],
            "prev_hash": current["prev_hash"]
        }

        recalculated_hash = sha256(
            json.dumps(
                test_block,
                sort_keys=True
            )
        )

        if current["hash"] != recalculated_hash:
            return False

        if current["prev_hash"] != previous["hash"]:
            return False

    return True


# =========================================================
# CONFIRMED DEAD MATCHES
# =========================================================

def verify_dead_matches():

    chain = load_chain()

    blockchain_hashes = set()

    for block in chain:

        if block["action"] in [
            "CONFIRMED_DEAD_MATCH",
            "BOOTSTRAP_CONFIRMED_DEAD_MATCH"
        ]:

            blockchain_hashes.add(
                block["data_hash"]
            )

    filesystem_hashes = set()

    if not os.path.exists(DEAD_CONFIRM_FOLDER):
        return False

    for file in os.listdir(DEAD_CONFIRM_FOLDER):

        if not file.lower().endswith(".json"):
            continue

        path = os.path.join(
            DEAD_CONFIRM_FOLDER,
            file
        )

        if not os.path.isfile(path):
            continue

        match_id = os.path.splitext(file)[0]

        manifest = {
            "match_id": match_id,
            "sha256": sha256_file(path)
        }

        manifest_hash = sha256(
            json.dumps(
                manifest,
                sort_keys=True
            )
        )

        filesystem_hashes.add(
            manifest_hash
        )

    return blockchain_hashes.issubset(
        filesystem_hashes
    )


# =========================================================
# MISSING SEARCH CASES
# =========================================================

def verify_missing_search_cases():

    chain = load_chain()

    blockchain_hashes = set()

    for block in chain:

        if block["action"] in [
            "MISSING_SEARCH_CASE",
            "BOOTSTRAP_MISSING_SEARCH_CASE"
        ]:

            blockchain_hashes.add(
                block["data_hash"]
            )

    filesystem_hashes = set()

    if not os.path.exists(
        MISSING_SEARCH_CASE_FOLDER
    ):
        return False

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

        manifest_hash = sha256(
            json.dumps(
                manifest,
                sort_keys=True
            )
        )

        filesystem_hashes.add(
            manifest_hash
        )

    return blockchain_hashes.issubset(
        filesystem_hashes
    )


# =========================================================
# DATASET PERSONS
# =========================================================

def verify_missing_dataset():

    chain = load_chain()

    blockchain_hashes = set()

    for block in chain:

        if block["action"] in [
            "MISSING_PERSON",
            "BOOTSTRAP_MISSING_PERSON"
        ]:

            blockchain_hashes.add(
                block["data_hash"]
            )

    filesystem_hashes = set()

    if not os.path.exists(
        MISSING_PERSON_FOLDER
    ):
        return False

    for person_id in os.listdir(
        MISSING_PERSON_FOLDER
    ):

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

        metadata_hash = sha256_file(
            metadata_path
        )

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
            "person_id": person_id,
            "metadata_sha256": metadata_hash,
            "images": image_hashes,
            "embeddings": embedding_hashes
        }

        manifest_hash = sha256(
            json.dumps(
                manifest,
                sort_keys=True
            )
        )

        filesystem_hashes.add(
            manifest_hash
        )

    return blockchain_hashes.issubset(
        filesystem_hashes
    )

# =========================================================
# CHILD WITNESS CASES
# =========================================================

def verify_child_cases():

    chain = load_chain()

    blockchain_hashes = {}

    # =====================================================
    # EXTRACT ORIGINAL CASE MANIFEST HASHES
    # =====================================================

    for block in chain:

        action = block.get("action")

        if action not in [
            "CHILD_WITNESS_CASE",
            "BOOTSTRAP_CHILD_WITNESS_CASE"
        ]:
            continue

        data_hash = block.get("data_hash")

        # ---------------------------------------------
        # Match filesystem manifest against block hash
        # ---------------------------------------------

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

            if not os.path.exists(metadata_path):
                continue

            metadata_hash = sha256_file(
                metadata_path
            )

            image_hashes = []

            for file in sorted(os.listdir(case_folder)):

                if not file.lower().endswith(".jpg"):
                    continue

                full_path = os.path.join(
                    case_folder,
                    file
                )

                image_hashes.append({
                    "file": file,
                    "sha256": sha256_file(
                        full_path
                    )
                })

            manifest = {
                "case_id": case_id,
                "metadata_sha256": metadata_hash,
                "images": image_hashes
            }

            manifest_hash = sha256(
                json.dumps(
                    manifest,
                    sort_keys=True
                )
            )

            if manifest_hash == data_hash:

                blockchain_hashes[
                    case_id
                ] = manifest_hash

    # =====================================================
    # VERIFY FILESYSTEM AGAINST STORED HASHES
    # =====================================================

    for case_id in os.listdir(CHILD_CASES_FOLDER):

        if case_id not in blockchain_hashes:
            return False

    for case_id, expected_hash in (
        blockchain_hashes.items()
    ):

        case_folder = os.path.join(
            CHILD_CASES_FOLDER,
            case_id
        )

        if not os.path.isdir(case_folder):
            return False

        metadata_path = os.path.join(
            case_folder,
            "metadata.json"
        )

        if not os.path.exists(metadata_path):
            return False

        metadata_hash = sha256_file(
            metadata_path
        )

        image_hashes = []

        for file in sorted(os.listdir(case_folder)):

            if not file.lower().endswith(".jpg"):
                continue

            full_path = os.path.join(
                case_folder,
                file
            )

            image_hashes.append({
                "file": file,
                "sha256": sha256_file(
                    full_path
                )
            })

        manifest = {
            "case_id": case_id,
            "metadata_sha256": metadata_hash,
            "images": image_hashes
        }

        current_hash = sha256(
            json.dumps(
                manifest,
                sort_keys=True
            )
        )

        if current_hash != expected_hash:
            return False

        # =================================================
        # VERIFY STATUS FILE
        # =================================================

        status_path = os.path.join(
            case_folder,
            "status.json"
        )

        if not os.path.exists(status_path):
            return False

        status_hash = sha256_file(
            status_path
        )

        status_verified = False

        for block in chain:

            if block.get("action") not in [
                "CHILD_CASE_STATUS_INITIAL",
                "CHILD_CASE_STATUS_UPDATED"
            ]:
                continue

            block_data_hash = block.get(
                "data_hash"
            )

            with open(status_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)

            manifest = {
                "case_id": case_id,
                "status": status_data["status"],
                "status_sha256": status_hash
            }

            manifest_hash = sha256(
                json.dumps(
                    manifest,
                    sort_keys=True
                )
            )

            if manifest_hash == block_data_hash:
                status_verified = True
                break

        if not status_verified:
            return False

    return True

def verify_child_dataset():

    chain = load_chain()

    blockchain_hashes = set()

    for block in chain:

        if block["action"] in [
            "MISSING_CHILD",
            "BOOTSTRAP_MISSING_CHILD"
        ]:

            blockchain_hashes.add(
                block["data_hash"]
            )

    filesystem_hashes = set()

    if not os.path.exists(
        MISSING_CHILD_FOLDER
    ):
        return False

    for child_id in os.listdir(
        MISSING_CHILD_FOLDER
    ):

        person_folder = os.path.join(
            MISSING_CHILD_FOLDER,
            child_id
        )

        if not os.path.isdir(person_folder):
            continue

        metadata_path = os.path.join(
            person_folder,
            "metadata.json"
        )

        if not os.path.exists(metadata_path):
            continue

        metadata_hash = sha256_file(
            metadata_path
        )

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

        manifest_hash = sha256(
            json.dumps(
                manifest,
                sort_keys=True
            )
        )

        filesystem_hashes.add(
            manifest_hash
        )

    return blockchain_hashes.issubset(
        filesystem_hashes
    )

# =========================================================
# CONFIRMED CHILD MATCHES
# =========================================================

def verify_child_matches():

    chain = load_chain()

    blockchain_hashes = set()

    # =====================================================
    # EXTRACT BLOCKCHAIN HASHES
    # =====================================================

    for block in chain:

        if block.get("action") not in [
            "CONFIRMED_CHILD_MATCH",
            "BOOTSTRAP_CONFIRMED_CHILD_MATCH"
        ]:
            continue

        blockchain_hashes.add(
            block.get("data_hash")
        )

    filesystem_hashes = set()

    if not os.path.exists(CHILD_CONFIRM_FOLDER):
        return False

    # =====================================================
    # VERIFY MATCH FILES
    # =====================================================

    for file in os.listdir(CHILD_CONFIRM_FOLDER):

        if not file.lower().endswith(".json"):
            continue

        path = os.path.join(
            CHILD_CONFIRM_FOLDER,
            file
        )

        if not os.path.isfile(path):
            continue

        match_id = os.path.splitext(file)[0]

        # =================================================
        # LOAD RECORD
        # =================================================

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                record = json.load(f)

        except:
            return False

        # =================================================
        # VERIFY CHILD EXISTS
        # =================================================

        child_id = str(
            record.get("child_id", "")
        )

        child_folder = os.path.join(
            MISSING_CHILD_FOLDER,
            child_id
        )

        if not os.path.isdir(child_folder):
            return False

        # =================================================
        # HASH EVIDENCE FILE
        # =================================================

        evidence_hash = sha256_file(path)

        manifest = {
            "match_id": match_id,
            "evidence_sha256": evidence_hash
        }

        # =================================================
        # VERIFY CASE-BASED MATCH
        # =================================================

        if "case_id" in record:

            case_folder = os.path.join(
                CHILD_CASES_FOLDER,
                str(record["case_id"])
            )

            if not os.path.isdir(case_folder):
                return False

        # =================================================
        # VERIFY DIRECT IMAGE MATCH
        # =================================================

        elif "image_id" in record:

            uploaded_image_path = os.path.join(
                CHILD_CONFIRM_FOLDER,
                f'{record["image_id"]}.jpg'
            )

            if not os.path.exists(
                uploaded_image_path
            ):
                return False

            manifest[
                "uploaded_image_sha256"
            ] = sha256_file(
                uploaded_image_path
            )

        else:

            return False

        # =================================================
        # HASH MANIFEST
        # =================================================

        manifest_hash = sha256(
            json.dumps(
                manifest,
                sort_keys=True
            )
        )

        filesystem_hashes.add(
            manifest_hash
        )

    # =====================================================
    # FINAL BLOCKCHAIN CHECK
    # =====================================================

    return blockchain_hashes.issubset(
        filesystem_hashes
    )