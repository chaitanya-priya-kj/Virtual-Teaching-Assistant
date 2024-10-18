import os
import re
import cv2
from pytube import YouTube
from pytube import Playlist

def is_slide_change(frame1, frame2, threshold=0.2):
    # Compute absolute difference between frames
    frame_diff = cv2.absdiff(frame1, frame2)

    # Calculate percentage of pixels that are different
    num_pixels_different = cv2.countNonZero(cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY))
    total_pixels = frame_diff.size // 3  # Total number of pixels in the frame
    percentage_different = num_pixels_different / total_pixels

    # Check if the percentage of different pixels exceeds the threshold
    return percentage_different > threshold

def extract_slide_frames(video_url, output_dir):
    # Fetch video stream from YouTube using pytube
    yt = YouTube(video_url)
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video_stream_url = video_stream.url

    # Initialize variables to track the previous frame and timestamp
    prev_frame = None
    prev_timestamp = 0

    # Initialize OpenCV video capture from the video stream URL
    cap = cv2.VideoCapture(video_stream_url)

    # Loop through the frames and save the ones where the slide changes
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the current timestamp
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Check if the slide has changed
        if prev_frame is not None and is_slide_change(prev_frame, frame):
            # Save the frame with its timestamp
            frame_filename = f"{frame_count:06d}_{timestamp:.1f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        # Update the previous frame and timestamp
        prev_frame = frame.copy()
        prev_timestamp = timestamp

    # Release the video capture object
    cap.release()

    return frame_count

def extract_slide_frames_from_playlist(playlist_url, output_dir):
    # Fetch the playlist
    playlist = Playlist(playlist_url)
    # playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")
    
    # Loop through the videos in the playlist
    for index, video_url in enumerate(playlist.video_urls, start=1):
        match = re.search(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})", video_url)
        v_id = match.group(1)
        video_output_dir = os.path.join(output_dir, f"{v_id}")
        os.makedirs(video_output_dir, exist_ok=True)
        num_images = extract_slide_frames(video_url, video_output_dir)
        print(f"Done with video {index} with {num_images} images.")

def main():
    # Example usage
    playlist_url = "https://www.youtube.com/watch?v=XvdLKUOldkE&list=PLAOUn-KLSAVOqj5TG8E1HTb8Txwxe6OtV"
    output_dir = "images_playlist1"
    extract_slide_frames_from_playlist(playlist_url, output_dir)

if __name__ == "__main__":
    main()
