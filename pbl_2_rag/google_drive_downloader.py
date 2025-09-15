import os
import io
import logging
from typing import List, Dict, Any
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

# Define the scopes for read-only access to Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveDownloader:
    """Downloads PDF files from Google Drive folders."""
    
    def __init__(self, credentials_path: str = 'credentials.json', token_path: str = 'token.pickle'):
        self.creds = None
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Google Drive API."""
        # Check if token.pickle exists (contains user access token)
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)
        
        # If no valid credentials are available, request user to log in
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Credentials file '{self.credentials_path}' not found. "
                        "Please download it from Google Cloud Console."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # Save the credentials for future use
            with open(self.token_path, 'wb') as token:
                pickle.dump(self.creds, token)
        
        self.service = build('drive', 'v3', credentials=self.creds)
        logger.info("Successfully authenticated with Google Drive API")
    
    def extract_folder_id(self, drive_url: str) -> str:
        """Extract folder ID from Google Drive URL."""
        if '/folders/' in drive_url:
            return drive_url.split('/folders/')[1].split('?').split('/')
        else:
            # If it's just an ID
            return drive_url
    
    def list_pdf_files(self, folder_id: str) -> List[Dict[str, str]]:
        """List all PDF files in the specified Google Drive folder."""
        try:
            # Query to find all PDF files in the specified folder
            query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, size, modifiedTime)",
                pageSize=1000
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"Found {len(files)} PDF files in Google Drive folder")
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files from Google Drive: {str(e)}")
            return []
    
    def download_file(self, file_id: str, file_name: str, local_path: str) -> bool:
        """Download a single file from Google Drive."""
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            
            downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024)  # 1MB chunks
            done = False
            
            logger.info(f"Downloading: {file_name}")
            while not done:
                status, done = downloader.next_chunk()
                progress = int(status.progress() * 100)
                if progress % 25 == 0:  # Log every 25%
                    logger.info(f"  Progress: {progress}%")
            
            # Write the file to local storage
            fh.seek(0)
            with open(local_path, 'wb') as f:
                f.write(fh.read())
            
            logger.info(f"Successfully downloaded: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {file_name}: {str(e)}")
            return False
    
    def download_pdfs_from_folder(self, folder_id_or_url: str, local_dir: str) -> List[str]:
        """
        Download all PDF files from Google Drive folder to local directory.
        
        Args:
            folder_id_or_url: Google Drive folder ID or full URL
            local_dir: Local directory to save PDF files
        
        Returns:
            List of successfully downloaded file paths
        """
        # Extract folder ID from URL if needed
        folder_id = self.extract_folder_id(folder_id_or_url)
        
        # Create local directory if it doesn't exist
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            logger.info(f"Created directory: {local_dir}")
        
        # List all PDF files in the folder
        files = self.list_pdf_files(folder_id)
        
        if not files:
            logger.warning("No PDF files found in the specified folder")
            return []
        
        downloaded_files = []
        
        # Download each file
        for file_info in files:
            file_id = file_info['id']
            file_name = file_info['name']
            local_path = os.path.join(local_dir, file_name)
            
            # Skip if file already exists locally
            if os.path.exists(local_path):
                logger.info(f"File already exists, skipping: {file_name}")
                downloaded_files.append(local_path)
                continue
            
            # Download the file
            if self.download_file(file_id, file_name, local_path):
                downloaded_files.append(local_path)
            else:
                logger.warning(f"Failed to download: {file_name}")
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} PDF files")
        return downloaded_files
    
    def get_folder_info(self, folder_id_or_url: str) -> Dict[str, Any]:
        """Get information about the Google Drive folder."""
        folder_id = self.extract_folder_id(folder_id_or_url)
        
        try:
            folder_info = self.service.files().get(fileId=folder_id).execute()
            files = self.list_pdf_files(folder_id)
            
            return {
                "folder_name": folder_info.get('name', 'Unknown'),
                "folder_id": folder_id,
                "pdf_count": len(files),
                "total_size": sum(int(f.get('size', 0)) for f in files),
                "files": [f['name'] for f in files]
            }
            
        except Exception as e:
            logger.error(f"Error getting folder info: {str(e)}")
            return {}
