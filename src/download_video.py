''' Install all video from metadata '''
import json
import yt_dlp
import os
''' Choose the folder containing metadata JSON files and the folder
to save downloaded videos '''
# folder_path = os.path.join('data', 'hcmc2023', 'metadata')
info_folder_path = os.path.join('..','data', 'metadata') # Example
video_folder_path = os.path.join('..','data', 'video')  # Example

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