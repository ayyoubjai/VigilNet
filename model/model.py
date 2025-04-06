import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection as detection



class VideoAnomalyDetector(nn.Module):
    def __init__(self, num_frames=16, height=256, width=256):
        super(VideoAnomalyDetector, self).__init__()
        self.num_frames = num_frames
        self.height = height
        self.width = width

        # Convolutional layers (input: RGB + mask = 4 channels)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample: H/2, W/2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample: H/4, W/4
        )
        # Feature map size after conv: [batch, 128, H/4, W/4]

        # Spatial attention setup
        self.spatial_seq_len = (height // 4) * (width // 4)  # e.g., 64 * 64 = 4096
        self.feature_dim = 128
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.spatial_seq_len + 1, self.feature_dim))
        self.spatial_attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=8)
        self.spatial_layer_norm = nn.LayerNorm(self.feature_dim)

        # Temporal attention setup
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames + 1, self.feature_dim))
        self.temporal_attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=8)
        self.temporal_layer_norm = nn.LayerNorm(self.feature_dim)

        # Classifier
        self.classifier = nn.Linear(self.feature_dim, 2)  # Anomaly or normal

    def forward(self, x):
        # Input: [batch, T, 4, H, W]
        batch_size = x.size(0)

        # Process all frames with conv layers
        x = x.view(batch_size * self.num_frames, 4, self.height, self.width)
        x = self.conv(x)  # [batch * T, 128, H/4, W/4]
        x = x.view(batch_size, self.num_frames, 128, self.height // 4, self.width // 4)

        # Spatial attention per frame
        spatial_features = []
        for t in range(self.num_frames):
            frame_x = x[:, t, :, :, :]  # [batch, 128, H/4, W/4]
            frame_x = frame_x.flatten(2).transpose(1, 2)  # [batch, (H/4)*(W/4), 128]
            
            # Add [CLS] token
            cls_token = self.spatial_cls_token.expand(batch_size, -1, -1)
            frame_x = torch.cat([cls_token, frame_x], dim=1)  # [batch, 1 + (H/4)*(W/4), 128]
            frame_x = frame_x + self.spatial_pos_embed 

            # Apply spatial attention
            frame_x = frame_x.transpose(0, 1)  # [1 + (H/4)*(W/4), batch, 128]
            attn_output, attn_weights = self.spatial_attention(frame_x, frame_x, frame_x)
            attn_output = attn_output.transpose(0, 1)  # [batch, 1 + (H/4)*(W/4), 128]
            frame_x = self.spatial_layer_norm(attn_output)

            # Extract [CLS] token output
            cls_output = frame_x[:, 0, :]  # [batch, 128]
            spatial_features.append(cls_output)

        # Stack spatial features: [batch, T, 128]
        spatial_features = torch.stack(spatial_features, dim=1)

        # Temporal attention
        temporal_cls_token = self.temporal_cls_token.expand(batch_size, -1, -1)
        temporal_x = torch.cat([temporal_cls_token, spatial_features], dim=1)  # [batch, 1 + T, 128]
        temporal_x = temporal_x + self.temporal_pos_embed

        # Apply temporal attention
        temporal_x = temporal_x.transpose(0, 1)  # [1 + T, batch, 128]
        temporal_attn_output, temporal_attn_weights = self.temporal_attention(temporal_x, temporal_x, temporal_x)
        temporal_attn_output = temporal_attn_output.transpose(0, 1)  # [batch, 1 + T, 128]
        temporal_x = self.temporal_layer_norm(temporal_attn_output)

        # Extract temporal [CLS] token output
        temporal_cls_output = temporal_x[:, 0, :]  # [batch, 128]

        # Classify
        output = self.classifier(temporal_cls_output)  # [batch, 2]

        return output, attn_weights, temporal_attn_weights  # Return attention weights for visualization
    
def create_yolo_masks(video_batch, yolo_model):
    # video_batch: [batch, T, 3, H, W]
    batch_size, T, _, H, W = video_batch.shape
    
    # Reshape to [batch*T, 3, H, W] for batched YOLO inference
    video_flat = video_batch.view(batch_size * T, 3, H, W)  # [batch*T, 3, H, W]
    
    # Run YOLO inference on all frames at once
    with torch.no_grad():
        results = yolo_model(video_flat)  # Should return a list or tensor
    
    # Initialize masks tensor
    masks = torch.zeros(batch_size * T, 1, H, W, device=video_batch.device)  # [batch*T, 1, H, W]
    
    # Check the type of results to handle different outputs
    if isinstance(results, torch.Tensor):
        # Case 1: Raw tensor output [batch*T, num_boxes, 6] (x1, y1, x2, y2, conf, cls)
        preds = results
        if preds.dim() == 3:  # Ensure it’s [batch*T, num_boxes, 6]
            for i in range(batch_size * T):
                frame_preds = preds[i]  # Predictions for frame i
                person_boxes = frame_preds[frame_preds[:, 5] == 0]  # Filter 'person' (class 0)
                mask = torch.zeros(H, W, device=video_batch.device)
                for box in person_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    mask[y1:y2, x1:x2] = 1
                masks[i, 0] = mask
    else:
        # Case 2: Results object with .pred attribute (list of [num_boxes, 6] per frame)
        for i, pred in enumerate(results.pred):  # results.pred is a list of predictions per frame
            person_boxes = pred[pred[:, 5] == 0]  # Filter 'person' (class 0)
            mask = torch.zeros(H, W, device=video_batch.device)
            for box in person_boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                mask[y1:y2, x1:x2] = 1
            masks[i, 0] = mask
    
    # Reshape masks back to [batch, T, 1, H, W]
    masks = masks.view(batch_size, T, 1, H, W)
    
    return masks  # [batch, T, 1, H, W]

