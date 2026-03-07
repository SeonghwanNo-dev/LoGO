import os
import time
import shutil
import google_upload # Requires google_upload.py and token.json in the same directory

# [Configuration]
GOOGLE_DRIVE_FOLDER_ID = '1MnADBZTRqtblNiFxQWJjS3AMJMID1qHw'
WATCH_PATH = "./6_1/"
BATCH_SIZE = 5
# Minimum time (seconds) after last modification to consider a folder "stable" (e.g., 300s = 5m)
STABLE_TIME_THRESHOLD = 300 

def monitor_and_upload_all():
    print(f"🚀 Monitoring {WATCH_PATH}... (Collecting {BATCH_SIZE} stable folders for compression)")
    
    # Initialize batch count
    count = 1
    
    while True:
        # Ensure watch path exists
        if not os.path.exists(WATCH_PATH):
            os.makedirs(WATCH_PATH, exist_ok=True)

        # 1. Check all entries in the watch path
        all_entries = os.listdir(WATCH_PATH)
        ready_folders = []

        now = time.time()
        for entry in all_entries:
            full_path = os.path.join(WATCH_PATH, entry)
            
            # Verify if the entry is a directory
            if os.path.isdir(full_path):
                # Check folder's mtime (updates when internal files are created/modified)
                mtime = os.path.getmtime(full_path)
                
                # Consider it "complete" only if it hasn't been modified for the threshold time
                if (now - mtime) > STABLE_TIME_THRESHOLD:
                    ready_folders.append(entry)
        
        ready_folders.sort()
        print(f"👀 Ready folders: {len(ready_folders)} / Total: {len(all_entries)}")

        # 2. Compress and upload when batch size is reached
        if len(ready_folders) >= BATCH_SIZE:
            targets = ready_folders[:BATCH_SIZE]
            timestamp = time.strftime("%m%d_%H%M%S")
            
            zip_base_name = f"backup_6_2_{count}"
            zip_filename = f"{zip_base_name}.zip"
            temp_batch_dir = f"./temp_batch_{timestamp}_{count}"
            
            print(f"📦 [Session {count}] Starting compression of {BATCH_SIZE} folders...")
            os.makedirs(temp_batch_dir, exist_ok=True)

            try:
                # Move target folders to the temporary working directory
                for folder_name in targets:
                    src = os.path.join(WATCH_PATH, folder_name)
                    dst = os.path.join(temp_batch_dir, folder_name)
                    shutil.move(src, dst)

                # Archive the temporary directory into a single zip file
                shutil.make_archive(zip_base_name, 'zip', temp_batch_dir)

                print(f"📤 Uploading to Google Drive: {zip_filename}")
                google_upload.upload_to_drive(zip_filename, GOOGLE_DRIVE_FOLDER_ID)

                # Clean up: Remove temporary directory and zip file upon success
                shutil.rmtree(temp_batch_dir)
                os.remove(zip_filename)
                
                print(f"✨ [Session {count}] Upload and local cleanup completed!")
                
                # Increment count only on successful upload
                count += 1

            except Exception as e:
                print(f"❌ Error occurred: {e}")
                # Rollback: Move folders back to the watch path if something fails
                if os.path.exists(temp_batch_dir):
                    for folder in os.listdir(temp_batch_dir):
                        shutil.move(os.path.join(temp_batch_dir, folder), os.path.join(WATCH_PATH, folder))
                    shutil.rmtree(temp_batch_dir)
                if os.path.exists(zip_filename):
                    os.remove(zip_filename)
        
        # Wait for 1 minute before next check
        time.sleep(60)

if __name__ == "__main__":
    monitor_and_upload_all()