import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

if "GOOGLE_DRIVE_CREDENTIALS" not in os.environ:
    print("No Google credentials, skip upload")
    exit(0)

creds = json.loads(os.environ["GOOGLE_DRIVE_CREDENTIALS"])

credentials = service_account.Credentials.from_service_account_info(
    creds,
    scopes=["https://www.googleapis.com/auth/drive"]
)

service = build('drive', 'v3', credentials=credentials)

file_metadata = {'name': 'run_id.txt'}
media = MediaFileUpload('run_id.txt')

file = service.files().create(
    body=file_metadata,
    media_body=media
).execute()

print("Uploaded:", file.get('id'))