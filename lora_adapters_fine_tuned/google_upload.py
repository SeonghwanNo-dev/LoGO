import os
import pickle
import time
import random
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

def upload_to_drive(file_path, folder_id, max_retries=5):
    creds = None
    # Load credentials from token.json
    if os.path.exists('token.json'):
        with open('token.json', 'rb') as token:
            creds = pickle.load(token)
    
    # Refresh token if it's expired or invalid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open('token.json', 'wb') as token:
                pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }

    # Retry loop using exponential backoff
    for attempt in range(max_retries):
        try:
            # chunksize set to 5MB for stable large file handling
            media = MediaFileUpload(file_path, resumable=True, chunksize=5*1024*1024)
            request = service.files().create(body=file_metadata, media_body=media, fields='id')
            
            print(f"🚀 Uploading '{os.path.basename(file_path)}'... (Attempt {attempt + 1}/{max_retries})")
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"📤 Progress: {int(status.progress() * 100)}%")
            
            print(f"✅ Success! File ID: {response.get('id')}")
            return True # Return True on successful upload

        except HttpError as e:
            # Retry on transient server errors (500, 502, 503, 504)
            if e.resp.status in [500, 502, 503, 504]:
                wait_time = (2 ** attempt) + random.random() # Exponentially increase wait time
                print(f"⚠️ Google server error ({e.resp.status}). Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # Do not retry for non-transient errors (e.g., 403 Permission, 404 Not Found)
                raise e
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise e

    print("🚫 Maximum retry attempts exceeded. Upload failed.")
    return False

if __name__ == "__main__":
    GOOGLE_DRIVE_FOLDER_ID = '1MnADBZTRqtblNiFxQWJjS3AMJMID1qHw'
    LOCAL_FILE = './backup6_1.zip'
    upload_to_drive(LOCAL_FILE, GOOGLE_DRIVE_FOLDER_ID)