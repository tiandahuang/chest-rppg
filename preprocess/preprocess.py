"""
Preprocess video -- stabilize and crop.
"""

import cv2
import numpy as np
from .utils import ndarray_to_vid, crop_video
from tqdm import tqdm
import scipy


def preprocess(video_path, *args, **kwargs):
    p = PreprocessPipeline(video_path, *args, **kwargs)
    return p.run()


class PreprocessPipeline():

    def __init__(self, video_path, *args, **kwargs):
        self.video_path = video_path
        self.output = kwargs.get('output', 'array')
        if self.output not in ['array', 'file']:
            raise ValueError("Invalid output type. Must be 'array' or 'file'.")
        self.output_path = kwargs.get('output_path', None)
        if self.output == 'file' and not self.output_path:
            raise ValueError("Output path must be specified if output type is 'file'.")
        
        self.crop = kwargs.get('crop', False)
        # crop_params should be dictionary with keys 
        # 'bottom_left', 'top_right', 'time_window', 'new_width', 'new_height'
        self.crop_params = kwargs.get('crop_params', dict())

        self.smoothing_window = kwargs.get('smoothing_window', 1)

        self.playback = kwargs.get('playback', False)

    def run(self):
        if self.crop:
            print('cropping video', self.crop_params)
            cropped_frames = crop_video(
                    self.video_path, 
                    output='array', 
                    **self.crop_params)
        else:
            cropped_frames = None

        print('stabilizing output')
        stabilized_frames = self._stabilize_video(cropped_frames)
        print('smoothing output')
        smoothed_frames = self._smooth_video(stabilized_frames, self.smoothing_window)

        if self.output == 'file':
            ndarray_to_vid(smoothed_frames, self.output_path)
            return self.output_path
        else:   # output == 'array'
            return smoothed_frames

    @staticmethod
    def _kp_preprocess(frame):
        return frame[:, :, 0] - np.min(frame, axis=-1)
    
    @staticmethod
    def _get_kp(frame):
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, 
                                            maxCorners=100,
                                            qualityLevel=0.1,
                                            minDistance=5)
        return keypoints
    
    def _stabilize_video(self, frames):
        from_video = frames is None
        playback = self.playback
        video_path = self.video_path

        # Open the video capture
        if from_video:
            cap = cv2.VideoCapture(video_path)
        else: cap = None

        # Define variables for tracking
        if from_video:
            ret, prev_frame = cap.read()
        else: prev_frame = frames[0]

        prev_kp_frame = self._kp_preprocess(prev_frame)
        prev_pts = self._get_kp(prev_kp_frame)

        # Lists to store tracked points and motion vectors
        mv = [np.eye(3)]
        stabilized_frames = []
        last_transform = np.eye(3)

        num_frames = len(frames) if not from_video else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in tqdm(range(num_frames)):
            if from_video:
                ret, frame = cap.read()
                if not ret:
                    break
            else:   # read from array
                frame = frames[frame_num]

            kp_frame = self._kp_preprocess(frame)

            # Calculate optical flow
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_kp_frame, kp_frame, prev_pts, 
                np.array([]))

            # Filter good features
            good_old = prev_pts[status == 1]
            good_new = curr_pts[status == 1]

            # Estimate motion from good features
            if len(good_old) >= 3:
                M, _ = cv2.estimateAffine2D(good_old, good_new)
                if M is not None:
                    M = np.vstack([M, [0, 0, 1]]) # Convert to 3x3 matrix
                else:
                    M = mv[-1] # Use last motion vector
            else:
                M = mv[-1]
                mv.append(M)

            # Update for next iteration
            prev_kp_frame = kp_frame.copy()
            prev_pts = good_new.reshape(-1, 1, 2)

            # Stabilize frame (reverse motion)
            h, w = frame.shape[:2]
            # avg_mv = np.mean(mv, axis=0)
            transform = M @ last_transform
            last_transform = transform
            inv_M = cv2.invertAffineTransform(transform[:2])
            stabilized_frame = cv2.warpAffine(frame, inv_M, (w, h))

            stabilized_frames.append(stabilized_frame)
            if playback:
                # Display original and stabilized frame side-by-side
                cv2.imshow('Video', np.hstack([frame, stabilized_frame, cv2.absdiff(frame, stabilized_frame)]))

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        if from_video: cap.release()
        cv2.destroyAllWindows()

        return np.array(stabilized_frames)

    def _smooth_video(self, frames, window):
        return scipy.ndimage.uniform_filter1d(
            frames, 
            size=window, 
            axis=0, 
            mode='nearest')
