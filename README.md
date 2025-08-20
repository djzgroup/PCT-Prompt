# PCT-Prompt

PCT-Prompt is a novel framework designed to enhance point cloud segmentation tasks by leveraging the power of Transformer models. Built upon the foundation of standard Transformer architectures, PCT-Prompt introduces a **prompt-guided feature branch** to effectively capture multi-scale geometric features and improve the adaptability of standard Transformers for dense prediction tasks like point cloud segmentation.

## Key Features:
- **Standard Transformer Backbone**: Utilizes a pre-trained standard Transformer model as the backbone for global feature extraction, making it robust for large-scale point cloud data processing.
- **Fine-grained Feature Extraction (FFE) Block**: A hierarchical feature extraction block that captures multi-scale geometric features using a geometry-sensitive abstraction layer and a PnP-3D layer to integrate local context with global regularization.
- **Prompt-refined Feature Learning (PFL) Block**: Generates dynamic prompt tokens from multi-scale geometric features, which are then iteratively refined using cross-attention mechanisms, integrating both local and global information.
- **Prompt Drop Mechanism**: Introduces a progressive prompt removal strategy across Transformer layers, balancing the preservation of local details with the global consistency of the segmented objects.
- **Training with Fine-tuning**: PCT-Prompt is designed for fine-tuning with pre-trained models, allowing for efficient domain adaptation and improved segmentation accuracy for specific datasets.

## Results:
PCT-Prompt has demonstrated superior performance over existing Transformer-based methods, particularly in dense prediction tasks such as point cloud segmentation. It has been evaluated on popular datasets such as **ShapeNetPart**, **S3DIS**, and **DALES**, showing significant improvements in segmentation accuracy and robustness.

## Installation:
You can clone this repository and install the required dependencies:

## Code:
The full code for the PCT-Prompt framework, along with pre-trained models and usage instructions, will be made available soon.

## Usage:
The framework can be used for point cloud segmentation tasks, leveraging pre-trained models to process and segment 3D point clouds. More usage details will be provided after the full code release.

