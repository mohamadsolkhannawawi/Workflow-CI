import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

credentials_info = json.loads(os.environ["GOOGLE_DRIVE_CREDENTIALS"])

credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/drive"]
)

service = build('drive', 'v3', credentials=credentials)

file_metadata = {
    'name': 'model_artifact.json'
}

media = MediaFileUpload('MLProject/artifacts/model_summary.json')

file = service.files().create(
    body=file_metadata,
    media_body=media
).execute()

print("Uploaded:", file.get('id'))