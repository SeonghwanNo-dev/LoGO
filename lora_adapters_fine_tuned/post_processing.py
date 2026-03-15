import os
import io
import zipfile
import shutil
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# 1. Path and Configuration Variables
TOKEN_FILE = 'token.json'              # Name of the authentication token file
CRED_FILE = 'credentials.json'         # Name of the credential file from Google Console
TARGET_FILE = './backup_6_3_10.zip'    # File to be uploaded
FOLDER_ID = '1MnADBZTRqtblNiFxQWJjS3AMJMID1qHw' # Google Drive Folder ID


def get_credentials():
    creds = None
    
    # 1. Check if the token.json file already exists
    if os.path.exists(TOKEN_FILE):
        # Important: Use the Credentials function instead of pickle to avoid '{' errors.
        creds = Credentials.from_authorized_user_file(TOKEN_FILE)
    
    # 2. Handle cases where credentials are missing or invalid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Token expired. Refreshing...")
            creds.refresh(Request())
        else:
            # Requires a local browser or manual auth logic to execute this block.
            print(" No valid credentials found. Please check token.json.")
            return None
            
        # Save the refreshed token back to the file
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            
    return creds

def process_sequentially():
    creds = get_credentials()
    if not creds: return
    service = build('drive', 'v3', credentials=creds)

    save_path = Path("./weights")
    temp_dl = Path("./temp_dl")
    temp_ext = Path("./temp_ext")
    
    # Create necessary directories
    for p in [save_path, temp_dl, temp_ext]: 
        p.mkdir(parents=True, exist_ok=True)

    # 1. List of already processed ZIP files
    completed_zips = {
        # 6_1 and etc
        "backup5.zip", "backup6_1.zip", "backup6_1_1.zip", "chunk1.zip",
        # 6_2
        "backup_6_2_24.zip", "backup_6_2_23.zip", "backup_6_2_22.zip",
        "backup_6_2_21.zip", "backup_6_2_20.zip", "backup_6_2_19.zip",
        "backup_6_2_18.zip", "backup_6_2_17.zip", "backup_6_2_16.zip",
        "backup_6_2_15.zip", "backup_6_2_14.zip", "backup_6_2_13.zip",
        "backup_6_2_12.zip", "backup_6_2_11.zip", "backup_6_2_10.zip",
        "backup_6_2_9.zip", "backup_6_2_8.zip", "backup_6_2_7.zip",
        "backup_6_2_6.zip", "backup_6_2_5.zip", "backup_6_2_4.zip",
        "backup_6_2_3.zip", "backup_6_2_2.zip", "backup_6_2_1.zip",
        # 6_3
        "backup_6_4_12.zip", "backup_6_3_11.zip", "backup_6_3_10.zip", 
        # "backup_6_3_10.zip",
        # "backup_6_3_9.zip", "backup_6_3_8.zip", "backup_6_3_7.zip",
        # "backup_6_3_6.zip", "backup_6_3_5.zip", "backup_6_3_4.zip",
        # "backup_6_3_3.zip", "backup_6_3_2.zip", "backup_6_3_1.zip",
        # 6_4
        # "backup_6_4_11.zip", "backup_6_4_10.zip",
        # "backup_6_4_9.zip", "backup_6_4_8.zip", "backup_6_4_7.zip",
        # "backup_6_4_6.zip", "backup_6_4_5.zip", "backup_6_4_4.zip",
        # "backup_6_4_3.zip", "backup_6_4_2.zip", "backup_6_4_1.zip",
    }

    # 2. Fetch file list from Drive
    query = f"'{FOLDER_ID}' in parents and mimeType = 'application/zip'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print(" No ZIP files found or permission denied.")
        return

    for item in items:
        file_id = item['id']
        file_name = item['name']

        if file_name in completed_zips:
            print(f"⏩ Skipping (already completed): {file_name}")
            continue

        print(f"\n🚀 Processing: {file_name}")

        # 3. Download ZIP
        zip_file_path = temp_dl / file_name
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(zip_file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"   Download {int(status.progress() * 100)}%", end='\r')

        # 4. Extract and Cleanup Checkpoints
        extract_to = temp_ext / zip_file_path.stem
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        # Remove unnecessary checkpoint folders
        for checkpoint_dir in list(extract_to.rglob('checkpoint-*')):
            if checkpoint_dir.is_dir():
                print(f" 🧹 Removing checkpoint: {checkpoint_dir.name}")
                shutil.rmtree(checkpoint_dir)

        # 5. Move task folders to designated chunk directories
        for task_dir in [d for d in extract_to.iterdir() if d.is_dir()]:
            first_char = task_dir.name[0].upper()
            
            # Determine the target chunk folder based on the first character
            if 'A' <= first_char <= 'P':
                target_chunk = save_path / "chunk_1"
            elif 'Q' <= first_char <= 'Z':
                target_chunk = save_path / "chunk_2"
            else:
                target_chunk = save_path / "others" # For numbers or special characters
            
            target_chunk.mkdir(parents=True, exist_ok=True)
            
            dest_path = target_chunk / task_dir.name
            
            # Handle duplicate folder names within the same chunk
            counter = 2
            original_dest = dest_path
            while dest_path.exists():
                dest_path = target_chunk / f"{original_dest.name}_{counter}"
                counter += 1
            
            shutil.move(str(task_dir), dest_path)
            
            # Log the result including the chunk info
            log_msg = f" Extracted to {target_chunk.name}: {dest_path.name}"
            print(f"   {log_msg}")
            
            with open("post_processing_result.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[{file_name}] {log_msg}\n")

        # 6. Temporary Files Cleanup
        if zip_file_path.exists(): os.remove(zip_file_path)
        if extract_to.exists(): shutil.rmtree(extract_to)

    print("\n All weight extractions completed successfully!")

if __name__ == '__main__':
    process_sequentially()