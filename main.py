import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection as detection
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model.model import VideoAnomalyDetector, create_yolo_masks
from dataset.dataset import UCFVideoDataset
from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLO model (e.g., YOLOv5 from ultralytics)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Hyperparameters
num_frames = 16
height, width = 64, 64
batch_size = 64
epochs = 1
num_workers = 4
print(f"device: {device}")
print(torch.cuda.get_device_name(0))
# Model, loss, optimizer
model = VideoAnomalyDetector(num_frames, height, width).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lists to store losses
train_losses = []
test_losses = []

# Define image transformations
transform = transforms.Compose([
    # Resize to 256x256 if model expects it; comment out if using 64x64
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and data loaders
train_dataset = UCFVideoDataset(root_dir='archive/Train', clip_length=16, stride=8, transform=transform)
test_dataset = UCFVideoDataset(root_dir='archive/Test', clip_length=16, stride=8, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def visualize_attention(attn_weights, frame_idx, height, width, title):
    print(f"{title} attention shape: {attn_weights.shape}")
    
    if isinstance(height, tuple) or isinstance(width, tuple):
        raise ValueError(f"height and width must be integers, got height={height}, width={width}")
    h4, w4 = (height // 4, width // 4) if 'spatial' in title.lower() else (height, width)
    print(f"{title} grid size: {h4}x{w4}")

    if attn_weights.dim() == 4:
        weights = attn_weights[0, 0]
    elif attn_weights.dim() == 3:
        weights = attn_weights[0]
    else:
        raise ValueError(f"Expected attention weights, got shape {attn_weights.shape}")

    if 'spatial' in title.lower():
        cls_attn = weights[0].cpu().detach().numpy()
        print(f"{title} CLS attention shape: {cls_attn.shape}")
        attn_values = cls_attn[1:]
        print(f"{title} CLS attention values (excluding CLS): {attn_values}")
        print(f"{title} attn_values shape: {attn_values.shape}, expected length: {h4 * w4}")

        try:
            attn_map = attn_values.reshape(h4, w4)
        except Exception as e:
            print(f"Error reshaping attn_values: {e}")
            return

        print(f"{title} attention map shape: {attn_map.shape}")
        print(f"{title} attention map min: {attn_map.min()}, max: {attn_map.max()}")
        if np.any(np.isnan(attn_map)) or np.any(np.isinf(attn_map)):
            print(f"Warning: {title} attention map contains NaN or Inf values")
            attn_map = np.nan_to_num(attn_map, nan=0.0, posinf=1.0, neginf=0.0)

        try:
            attn_map = F.interpolate(
                torch.tensor(attn_map).unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode='bilinear'
            ).squeeze().numpy()
        except Exception as e:
            print(f"Error during interpolation: {e}")
            return

        print(f"{title} interpolated attention map shape: {attn_map.shape}")
        print(f"{title} interpolated attention map min: {attn_map.min()}, max: {attn_map.max()}")
        if np.any(np.isnan(attn_map)) or np.any(np.isinf(attn_map)):
            print(f"Warning: {title} interpolated attention map contains NaN or Inf values")
            attn_map = np.nan_to_num(attn_map, nan=0.0, posinf=1.0, neginf=0.0)

        # Ensure figure is closed before creating a new one
        plt.close('all')
        plt.figure(figsize=(8, 6))
        plt.imshow(attn_map, cmap='hot')
        plt.title(f'{title} Attention Map for Frame {frame_idx}')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Width')
        plt.ylabel('Height')
        try:
            plt.savefig(f'{title}_attention_frame_{frame_idx}.png')
            print(f"Saved {title} plot successfully")
        except Exception as e:
            print(f"Error saving {title} plot: {e}")
        plt.show()
        plt.close()

    else:
        frame_attn = weights[1:, 1:]
        print(f"{title} frame-to-frame attention shape: {frame_attn.shape}")
        print(f"{title} frame-to-frame attention values:\n{frame_attn}")

        attn_map = frame_attn.cpu().detach().numpy()
        if np.any(np.isnan(attn_map)) or np.any(np.isinf(attn_map)):
            print(f"Warning: {title} attention map contains NaN or Inf values")
            attn_map = np.nan_to_num(attn_map, nan=0.0, posinf=1.0, neginf=0.0)

        plt.close('all')
        plt.figure(figsize=(8, 6))
        plt.imshow(attn_map, cmap='viridis')
        plt.title(f'{title} Attention Matrix')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Frame Index')
        plt.ylabel('Query Frame Index')
        try:
            plt.savefig(f'{title}_attention_frame_{frame_idx}.png')
            print(f"Saved {title} plot successfully")
        except Exception as e:
            print(f"Error saving {title} plot: {e}")
        plt.show()
        plt.close()

def main(train=True):

    if train:
        # Training loop with progress bars
        for epoch in tqdm(range(epochs), desc='Epochs'):
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
            for batch_idx, (video, labels) in enumerate(train_bar):
                video = video.to(device)  # [batch, T, 3, H, W]
                labels = labels.to(device)  # [batch]

                # Generate masks with YOLO for the entire batch
                masks = create_yolo_masks(video, yolo_model)

                # Concatenate masks to input
                input_video = torch.cat([video, masks], dim=2)  # [batch, T, 4, H, W]

                # Forward pass
                output, spatial_attn, temporal_attn = model(input_video)
                loss = criterion(output, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            test_loss = 0.0
            test_bar = tqdm(test_loader, desc=f'Epoch {epoch} Testing')
            with torch.no_grad():
                for video, labels in test_bar:
                    video = video.to(device)
                    labels = labels.to(device)

                    masks = create_yolo_masks(video, yolo_model)
                    input_video = torch.cat([video, masks], dim=2)
                    output, _, _ = model(input_video)
                    loss = criterion(output, labels)
                    test_loss += loss.item()
                    test_bar.set_postfix(loss=loss.item())

            test_loss /= len(test_loader)
            test_losses.append(test_loss)

            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # Save model
        torch.save(model.state_dict(), 'anomaly_detector.pth')

        # Plotting the loss curves
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Curves')
        plt.legend()
        plt.savefig(f'Training_Test_Loss.png')
        plt.show()

    else:
        # Load model parameters
        model.load_state_dict(torch.load('anomaly_detector.pth'))
        model.eval()

        # Get a random batch from the test loader
        test_iter = iter(test_loader)
        video, labels = next(test_iter)  # Take one batch
        video = video.to(device)  # [batch, T, 3, H, W]

        # Generate masks
        with torch.no_grad():
            masks = create_yolo_masks(video, yolo_model)
            input_video = torch.cat([video, masks], dim=2)  # [batch, T, 4, H, W]

            # Forward pass to get attention weights
            output, spatial_attn, temporal_attn = model(input_video)

        # Visualize attention for the first video in the batch
        visualize_attention(spatial_attn, frame_idx=0, height=height, width=width,title="Spatial")
        visualize_attention(temporal_attn, frame_idx=0, height=num_frames, width=num_frames,title="Temporel")
    plt.ioff()  # Turn off interactive mode after execution

if __name__ == '__main__':
    main(train=False)  # Set to False to visualize attention instead of training