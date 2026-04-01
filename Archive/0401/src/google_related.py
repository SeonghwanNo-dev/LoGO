# while(True):
#     print("test")
#     print("ing")

import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# 권한 범위 (업로드 권한)
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_new_token():
    creds = None
    # 1. 기존 토큰 로드 시도
    if os.path.exists('token.json'):
        print("1")
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # 2. 토큰이 없거나 만료되었다면 재인증
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            print("2")
        else:
            # 브라우저를 띄워 새로 인증받는 핵심 로직
            flow = InstalledAppFlow.from_client_secrets_file('./client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
            print("3")

        # 3. 새로 만든 토큰 저장
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

if __name__ == "__main__":
    get_new_token()
