import cv2
import os

# folder containing frames
image_folder = "fall_frames"

# output video name
video_name = "fall_video.mp4"

# frames per second
fps = 20

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# read first image to get size
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape

# create video writer
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()

print("Video created successfully!")