import os
import hashlib

def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_hash(file_path, hash_file_path):
    with open(hash_file_path, "r") as hash_file:
        expected_hash = hash_file.read().strip()
    actual_hash = compute_sha256(file_path)
    return expected_hash == actual_hash


def process_dir(data_dir, hash_dir):
    status_entries = {}
    max_filename_length = 60 # Ensures static table width
    max_status_length = 7 # Ensures static table width
    print("====================== Dataset Integrity Check ======================")
    
    files_to_check = set(f[:-7] for f in os.listdir(hash_dir) if f.endswith(".sha256"))
    
    for file_name in files_to_check:
        file_path = os.path.join(data_dir, file_name)
        hash_file_path = os.path.join(hash_dir, file_name + ".sha256")
        
        if os.path.isfile(file_path):
            is_valid = validate_hash(file_path, hash_file_path)
            status_entries[file_name] = "VALID" if is_valid else "INVALID"
        else:
            status_entries[file_name] = "MISSING"
    
    for file_name, status in status_entries.items():
        print(f"{str(file_name).ljust(max_filename_length)}  {str(status).rjust(max_status_length)}")
    
    return all(status == "VALID" for status in status_entries.values())

