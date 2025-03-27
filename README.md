# ShorthandML: Deep Learning for Shorthand Recognition

ShorthandML is an advanced deep learning system for optical recognition and phonetic transcription of shorthand writing. The system employs a sophisticated neural architecture to convert shorthand images into phonetic sequences, enabling automated transcription of this specialized writing system.

## 🔍 Technical Overview

ShorthandML utilizes a hybrid CNN-LSTM-Transformer architecture with the following components:

- **Convolutional Feature Extraction**: Three-layer CNN with batch normalization and spatial pooling to extract visual features from shorthand image inputs.
- **Bidirectional LSTM Encoding**: 3-layer BiLSTM to capture sequential information with 512 hidden units.
- **Multi-Head Attention Mechanism**: 8-head attention layer that allows the model to focus on relevant parts of the sequence.
- **Connectionist Temporal Classification (CTC)**: Specialized loss function for sequence-to-sequence learning without explicit alignment.
- **Weighted Loss Function**: Custom weighting scheme based on phonetic unit frequencies to improve rare phoneme recognition.
- **Dynamic Decoding Strategies**: Multiple decoding approaches including beam search and temperature-controlled sampling.

## 🧠 Model Architecture

```
ShorthandModel
├── Convolutional Layers
│   ├── Conv2d(1, 32, 3) → BatchNorm2d → ReLU → MaxPool2d → Dropout
│   ├── Conv2d(32, 64, 3) → BatchNorm2d → ReLU → MaxPool2d → Dropout
│   └── Conv2d(64, 128, 3) → BatchNorm2d → ReLU → MaxPool2d → Dropout
├── Dense Preparation
│   └── Linear → ReLU → Dropout
├── Sequence Processing
│   └── LSTM(512, 512, num_layers=3, bidirectional=True)
├── Attention Mechanism
│   ├── MultiHeadAttention(hidden_size=1024, num_heads=8)
│   ├── LayerNorm → Dropout
│   ├── FeedForward Network
│   └── LayerNorm → Dropout
└── Output Layer
    └── Linear → LogSoftmax
```

## 📊 Training & Monitoring

The system includes a comprehensive monitoring framework for model diagnostics:

- **Gradient Tracking**: Monitors norm of gradients to detect vanishing/exploding gradients
- **Weight Analysis**: Tracks weight norms and distributions across training
- **Activation Capture**: Hooks to capture layer activations during forward passes
- **Confusion Matrix Generation**: Visual representation of model errors
- **Performance Metrics**: Custom metrics to evaluate phonetic sequence accuracy

## 📋 Phonetic Processing

The system uses an API-driven approach to convert English words to IPA (International Phonetic Alphabet):

- **Batch Processing**: Efficiently processes words in configurable batch sizes
- **Caching Mechanism**: Persistent caching of phonetic translations
- **Error Handling**: Robust fallback strategies for failed API calls
- **Custom Phonetic Unit Set**: Focused set of 28 phonetic units for English

## 📦 Dataset Preparation

Images undergo specialized preprocessing to maintain consistent input quality:

- **Aspect Ratio Preservation**: Custom transformation that maintains the writing's natural proportions
- **Data Augmentation**: Controlled rotations and translations to improve model robustness
- **Normalization**: Standard channel-wise normalization

## 🚀 Usage

```python
# Load the pretrained model
model = ShorthandModel.load('path/to/model.pth')

# Prepare image
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image = Image.open('shorthand_sample.png')
tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(tensor)
    log_probs = F.log_softmax(output, dim=2)
    transcription = model.decode_ctc(log_probs)

print(f"Transcription: {transcription[0]}")
```

## 🔧 Model Optimization

The model incorporates several key optimizations:

- **Temperature Scaling**: Controls prediction confidence to prevent overcommitment
- **Class-Weighted Loss**: Addresses class imbalance in phonetic distributions
- **Gradient Clipping**: Prevents unstable gradients during backpropagation
- **Orthogonal Initialization**: Specialized weight initialization for RNN components

## 🔗 Dependencies

- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- tqdm
- Pillow
