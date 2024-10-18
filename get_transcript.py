import os
import re
import pandas as pd
from pytube import Playlist, YouTube
import youtube_dl
import youtube_transcript_api

def extract_timestamp(image_name):
    match = re.search(r'\d+_(\d+\.\d+)\.jpg', image_name)
    if match:
        return float(match.group(1))
    else:
        return None

def get_transcript_with_time(video_id, start_time_ms, end_time_ms):
    try:
        transcript_list = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id)
    
        filtered_transcript = [line['text'] for line in transcript_list if line['start'] * 1000 >= start_time_ms and line['start'] * 1000 <= end_time_ms]

        return ' '.join(filtered_transcript)

    except youtube_transcript_api.TranscriptNotFoundError:
        pass

    try:
        ydl_opts = {
            'writesubtitles': True,
            'subtitleslangs': ['en'], 
            'skip_download': True,  
            'quiet': True  
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_id, download=False)
            automatic_captions = info_dict.get('automatic_captions', {}).get('en', [])
            
            filtered_captions = [caption['text'] for caption in automatic_captions if caption['start'] * 1000 >= start_time_ms and caption['start'] * 1000 <= end_time_ms]

            return ' '.join(filtered_captions)

    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_image_transcripts(video_id, folder_path):
    start_time = 0
    image_names = sorted(os.listdir(folder_path))
    data = []

    for image_name in image_names:
        end_time = extract_timestamp(image_name)
        transcript = get_transcript_with_time(video_id, start_time, end_time)
        data.append({'Image Name': image_name, 'Transcript': transcript})
        start_time = end_time

    df = pd.DataFrame(data)
    return df

def process_playlist(playlist_url, output_folder):
    playlist_url = "https://www.youtube.com/watch?v=XvdLKUOldkE&list=PLAOUn-KLSAVOqj5TG8E1HTb8Txwxe6OtV"
    playlist = Playlist(playlist_url)

    folder_path = "images_playlist/"
    # playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")
    for index, video_url in enumerate(playlist.video_urls, start=1):
        match = re.search(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})", video_url)
        video_id = match.group(1)
        print(video_id)
        img_path = folder_path+'v'+video_id
        transcript_df = extract_image_transcripts(video_id, img_path)
        output_csv_path = os.path.join(output_folder, f"{video_id}.csv")
        transcript_df.to_csv(output_csv_path, index=False)
        print(f"Transcript saved for video ID: {video_id}")

def main():
    playlist_url = "https://www.youtube.com/watch?v=XvdLKUOldkE&list=PLAOUn-KLSAVOqj5TG8E1HTb8Txwxe6OtV"
    output_folder = "video_transcripts"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    process_playlist(playlist_url, output_folder)

if __name__ == "__main__":
    main()
