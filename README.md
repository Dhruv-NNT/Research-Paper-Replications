# 🖼️ An Image is Worth 16x16 Words: Vision Transformers in PyTorch

This project replicates the famous research paper "An Image is Worth 16x16 Words" using PyTorch. The paper introduces Vision Transformers (ViTs), which apply the Transformer architecture, originally designed for natural language processing, to image classification tasks. This implementation demonstrates how ViTs can achieve state-of-the-art performance on image classification benchmarks.

## 📚 Project Overview

### Research Paper Summary

The research paper "An Image is Worth 16x16 Words" proposes the Vision Transformer (ViT), which splits an image into patches and processes them as a sequence of words using a standard Transformer encoder. This method leverages the strengths of Transformers in capturing long-range dependencies and has shown to outperform traditional convolutional neural networks (CNNs) on image classification tasks.

### Key Contributions

- **Patch Embeddings**: Splitting images into fixed-size patches and linearly embedding each patch.
- **Transformer Encoder**: Applying a Transformer encoder to the sequence of embedded patches.
- **Classification Head**: Using the output of the Transformer encoder for image classification.

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/Research-Paper-Replications.git
cd Research-Paper-Replications
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

## 📂 Code Structure

```
📦Research-Paper-Replications
 ┣ 📜An image is worth 16x16 words.pdf       # The actual research paper
 ┣ 📜vision-transformer-pytorch.ipynb        # Jupyter notebook with code and experiments
 ┗ 📜vision-transformer-pytorch.html         # HTML export of the Jupyter notebook
```

## 🚀 Running the Code

1. **Open the Jupyter Notebook:**

```bash
jupyter notebook vision-transformer-pytorch.ipynb
```

2. **Run the cells in the notebook to execute the experiments and view the results.**

## 📝 Implementation Details

### Vision Transformer (ViT) Architecture

The Vision Transformer (ViT) architecture consists of the following main components:

1. **Patch Embedding Layer**:
Converts an image $x \in \mathbb{R}^{H \times W \times C}$ into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(H, W)$ is the image size, $C$ is the number of channels, $(P, P)$ is the patch size, and $N = \frac{HW}{P^2}$ is the number of patches.

   ```python
   class PatchEmbedding(nn.Module):
    """Turns a 2D input image inta a 1D sequence learnable embedding"""
    def __init__(self,
                in_channels:int=3,
                patch_size:int=16,
                embedding_dim:int=768):
        super().__init__()
        # To turn an image into patches
        self.patcher=nn.Conv2d(in_channels=in_channels,
                              out_channels=embedding_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              padding=0)
        # Create a layer to flatten the patch feature maps into a single dimension
        self.flatten=nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        image_resol = x.shape[-1]
        assert image_resol%patch_size==0, f"Input image size must be divisble by patch size, image shape: {image_resol}, patch size: {patch_size}"
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1)
   ```

3. **Transformer Encoder**: Applies multiple layers of the standard Transformer encoder to the sequence of embedded patches.

![image](https://github.com/user-attachments/assets/4de8e12f-8439-4d12-a4fc-ff9b9ff8dda4)

   
4. **Classification Head**: Uses the output of the Transformer encoder for classification.

### Mathematical Formulation

The Transformer encoder operates on the sequence of embedded patches $x_p$ using multi-head self-attention and feed-forward neural networks:

1. **Multi-Head Self-Attention**:
   $\[
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]$

2. **Feed-Forward Network**:
   $\[
   \text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
   \]$

## 🔍 Insights and Results

### Insights from Implementation

- **Patch Size Impact**: The choice of patch size $(P, P)$ significantly affects the performance and computational efficiency.
- **Data Augmentation**: Effective data augmentation techniques are crucial for training ViTs to prevent overfitting.
- **Transfer Learning**: Fine-tuning pre-trained ViTs on specific datasets can lead to substantial performance improvements.

### Results

- **Accuracy**: The implemented Vision Transformer achieved competitive accuracy on benchmark datasets.
- **Efficiency**: Despite the lack of inductive biases inherent in CNNs, ViTs demonstrated remarkable efficiency in capturing global context.

## 📊 Future Work

- **Experiment with Different Patch Sizes**: Analyze the impact of various patch sizes on model performance.
- **Incorporate Positional Embeddings**: Enhance the model by experimenting with different types of positional embeddings.
- **Explore Hybrid Models**: Combine ViTs with convolutional layers to leverage the benefits of both architectures.

## 📧 Contact

For any questions or feedback, feel free to reach out to:

- **Name**: Aradhya Dhruv
- **Email**: aradhya.dhruv@gmail.com
