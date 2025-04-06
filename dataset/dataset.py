import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class UCFVideoDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, stride=8, transform=None):
        self.root_dir = root_dir  # e.g., 'path/to/Train'
        self.clip_length = clip_length  # Number of frames per clip (T)
        self.stride = stride  # Step between clip starting points
        self.transform = transform
        self.clips = []  # List of clips (each clip is a list of frame paths)
        self.labels = []  # Corresponding labels for each clip

        # Binary labels: 0 for normal, 1 for anomalous
        self.class_to_label = {
            'NormalVideos': 0,
            'Abuse': 1, 
            'Arrest': 1, 
            'Arson': 1, 
            'Assault': 1, 
            'Burglary': 1, 
            'Explosion': 1, 
            'Fighting': 1, 
            'RoadAccidents': 1, 
            'Robbery': 1, 
            'Shooting': 1, 
            'Shoplifting': 1, 
            'Stealing': 1, 
            'Vandalism': 1
        }

        # Collect all frame paths
        frame_paths = glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True)

        # Group frames by video_id
        video_frames = {}
        for frame_path in frame_paths:
            class_name = os.path.basename(os.path.dirname(frame_path))  # e.g., 'Abuse'
            filename = os.path.basename(frame_path)  # e.g., 'Abuse028_x264_0.png'
            parts = filename.split('_')
            video_id = '_'.join(parts[:-1])  # e.g., 'Abuse028_x264'
            frame_number = int(parts[-1].split('.')[0])  # e.g., 0
            if video_id not in video_frames:
                video_frames[video_id] = {'class_name': class_name, 'frames': []}
            video_frames[video_id]['frames'].append((frame_path, frame_number))

        # Create clips for each video
        for video_id, data in video_frames.items():
            class_name = data['class_name']
            label = self.class_to_label[class_name]
            # Sort frames by frame number
            frames = sorted(data['frames'], key=lambda x: x[1])
            frame_paths = [f[0] for f in frames]
            num_frames = len(frame_paths)
            # Generate clips with stride
            for start in range(0, num_frames - clip_length + 1, stride):
                end = start + clip_length
                clip_frames = frame_paths[start:end]
                self.clips.append(clip_frames)
                self.labels.append(label)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_frames = self.clips[idx]
        label = self.labels[idx]
        # Load and transform frames
        frames = []
        for frame_path in clip_frames:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        video_tensor = torch.stack(frames)  # Shape: [T, 3, H, W]
        return video_tensor, label

