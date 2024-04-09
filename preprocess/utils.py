"""
Utility functions for working with video files.
"""

import cv2
import numpy as np
from tqdm import tqdm

def video_playback_fromframes(frames, fps=None):
    playback_delay = 1000//fps if fps else 1
    for f in frames:
        cv2.imshow('Video', f)
        if cv2.waitKey(playback_delay) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def video_playback_fromfile(filename, fps=None):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")
    
    if not fps:
        fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000//fps if fps > 1 else 1)
    print(delay, fps)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
    
        cv2.imshow('Video', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def vid_to_ndarray(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def ndarray_to_vid(frames, output_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Set codec (e.g., MP4)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()

def extract_nth_frame(video_path, n, output_path=None, flip=False):
    cap = cv2.VideoCapture(video_path)
    if n < 1 or n > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)): return None

    for count in range(n-1):
       ret, _ = cap.read()
       if not ret or not cap.isOpened(): break

    ret, frame = cap.read()
    cap.release()
    if not ret: return None

    if flip: frame = cv2.flip(frame, 0)
    if output_path: cv2.imwrite(output_path, frame)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def crop_video(input_path, output='array', output_path=None,
               bottom_left=None, top_right=None,
               time_window=None,
               new_width=None, new_height=None, **kwargs):

    if output not in ['array', 'file']:
        raise ValueError("Invalid output type. Must be 'array' or 'file'.")
    if output == 'file' and not output_path:
        raise ValueError("Output path must be specified if output type is 'file'.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise Exception(f"Could not open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Set codec (e.g., MP4)
    print(f'original video dimensions: width:{width}, height:{height}')

    # crop video dims
    if bottom_left is None or top_right is None:
        top_right = (width, height)
        bottom_left = (0, 0)
    crop_width = top_right[0] - bottom_left[0]
    crop_height = top_right[1] - bottom_left[1]
    if (crop_width < 1 or crop_height < 1 or 
        crop_width > width or crop_height > height or
        bottom_left[0] < 0 or bottom_left[1] < 0 or
        top_right[0] > width or top_right[1] > height):
        raise Exception(f'invalid dims: width:{crop_width}, height:{crop_height}')
    print(f'new cropped video dimensions: width:{crop_width}, height:{crop_height}')

    # crop time
    if time_window is None: time_window = (0, cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps)
    frame_start = int(time_window[0] * fps)
    frame_end = int(time_window[1] * fps)

    # resize video
    if new_height is None or new_width is None:
        new_height = crop_height
        new_width = crop_width
    print(f'final video dimensions: width:{new_width}, height:{new_height}')


    output_frames = []
    if output == 'file':
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    for frame_num in tqdm(range(frame_end)):
        if frame_num < frame_start: continue
        ret, frame = cap.read()
        if not ret: break

        cropped_frame = frame[bottom_left[1]:bottom_left[1]+crop_height, 
                                bottom_left[0]:bottom_left[0]+crop_width]
        resized_frame = cv2.resize(cropped_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        if output == 'array': output_frames.append(resized_frame)
        else: out.write(resized_frame)

    cap.release()
    if output == 'file': out.release()
    cv2.destroyAllWindows()

    if output == 'array': return np.array(output_frames)
    else: return output_path

def smear(arr, mean=True):
    if mean:
        return np.mean(arr, axis=0).astype(int)[:,:,[2,1,0]]
    else:
        return np.std(arr, axis=0).mean(axis=-1).astype(int)

def resize_func(f, array, size):
    # example usage: resize_func(np.mean, arr, 3)
    t, h, w = array.shape
    h_new, w_new = h//size, w//size
    flattened = np.reshape(array, (t, h_new, size, w_new, size)).swapaxes(2, 3).reshape(t, h_new, w_new, size**2)
    return f(flattened, axis=-1)
