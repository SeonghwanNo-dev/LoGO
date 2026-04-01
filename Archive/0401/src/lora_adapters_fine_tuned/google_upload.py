import os
import time
import random
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError


# 1. Path and Configuration Variables
TOKEN_FILE = 'token.json'              # Name of the authentication token file
CRED_FILE = 'credentials.json'         # Name of the credential file from Google Console
TARGET_FILE = './chunk_selected_10.zip'    # File to be uploaded
FOLDER_ID = '1_8AxnNX_EMO7YqnbftCvFrNcrbTW0izm' # Google Drive Folder ID


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
            print("❌ No valid credentials found. Please check token.json.")
            return None
            
        # Save the refreshed token back to the file
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            
    return creds

def upload_to_drive(file_path, folder_id):
    try:
        creds = get_credentials()
        if not creds: return

        service = build('drive', 'v3', credentials=creds)
        
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }

        # Transfer logic (Retry up to 5 times)
        for attempt in range(5):
            try:
                media = MediaFileUpload(file_path, resumable=True, chunksize=5*1024*1024)
                request = service.files().create(body=file_metadata, media_body=media, fields='id')
                
                print(f"🚀 Uploading {os.path.basename(file_path)}... (Attempt {attempt+1})")
                
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        print(f"📤 Progress: {int(status.progress() * 100)}%")
                
                print(f"✅ Success! File ID: {response.get('id')}")
                return

            except HttpError as e:
                # Exponential backoff for server-side errors
                if e.resp.status in [500, 502, 503, 504]:
                    time.sleep((2 ** attempt) + random.random())
                else:
                    raise e
                    
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    upload_to_drive(file_path=TARGET_FILE, folder_id=FOLDER_ID)