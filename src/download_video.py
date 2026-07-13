''' Install all video from metadata '''
import json
from pathlib import Path
import yt_dlp
import os

'''Set up paths for metadata and video storage'''
BASE_DIR = Path(__file__).resolve().parent

''' 
Choose the folder containing metadata JSON files and the folder to save downloaded videos:
    info_folder_path: folder containing metadata JSON files
    video_folder_path: folder to save downloaded videos
'''
info_folder_path = BASE_DIR / ".." / "data" / "hcmc2023" / "data-batch-1" / "metadata"  # Example
video_folder_path = BASE_DIR / ".." / "data" / "hcmc2023" / "original_videos"  # Example

if not info_folder_path.exists():
    print(f"Metadata folder {info_folder_path} does not exist. Please check the path.")
    exit(1)
if not video_folder_path.exists():
    os.makedirs(video_folder_path)

filenames = [f for f in os.listdir(info_folder_path) if f.endswith('.json')]
for filename in filenames:
    file_path = os.path.join(info_folder_path, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        if data:
            data = json.loads(data)

            url = data.get('watch_url')
            try:
                ydl_opts = {
                    'format': 'best',
                    'outtmpl': os.path.join(video_folder_path, f'{filename.replace(".json", "")}.%(ext)s'),
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                except Exception as e:
                    print(f"Error occurred while downloading {filename}: {e}")
            except Exception as e:
                print(f"Error occurred while processing {filename}: {e}")
        else:
            print(f"Empty data in {filename}, skipping download.")