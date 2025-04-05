![Navy Minimal Company Logo](https://github.com/ayyoubjai/VigilNet/assets/58954552/2d45e707-f872-44bd-8c8b-13fb996135f6)

# VigilNet
VigilNet is an advanced AI system that leverages cutting-edge computer vision and deep learning techniques to enhance video surveillance. It's designed to analyze video feeds from surveillance cameras and provide real-time detection of various anomalies, including fights, robberies, and criminal activities.

graph TD
    A[UCF-Crime Dataset<br>[N, T, 3, 64, 64]] --> B[Normalize]
    B --> C[YOLO]
    C --> D[YOLO Masks<br>[N, T, 1, 64, 64]]
    B --> E[Concatenate]
    D --> E
    E --> F[Concatenate Output<br>[N, T, 4, 64, 64]]
    F -->|VigiNet| G[CNN<br>[N, T, 128, 16, 16]]
    G --> H[Flatten Feature Map<br>[N, T, 256, 128]]
    H --> I[Add Spatial [CLS] Token<br>[N, T, 257, 128]]
    I --> J[Spatial Multi-Head Attention<br>[N, T, 257, 128]]
    J --> K[Extract Spatial [CLS] Token<br>[N, T, 128]]
    J --> L[Visualize Attention Heatmap]
    K --> M[Add Temporal [CLS] Token<br>[N, 1+T, 128]]
    M --> N[Temporal Multi-Head Attention<br>[N, 1+T, 128]]
    N --> O[Extract Temporal [CLS] Token<br>[N, 128]]
    O --> P[Classifier<br>[N, 2]]
    P --> Q[Binary Classification<br>(Normal/Anomalous)<br>[N, 2]]
