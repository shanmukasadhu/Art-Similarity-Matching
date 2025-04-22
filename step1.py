import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
import open_clip



# Set device first, so we can move everything to it consistently
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create model and move it to the device immediately
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
model = model.to(device)


single_sample = ["1886"]


# Load and preprocess the image
image = preprocess(Image.open("wikiart/Romanticism/ivan-aivazovsky_the-tempest-1886.jpg")).unsqueeze(0)
image = image.to(device)


# Use RN50 Tokenizer:
tokenizer = open_clip.get_tokenizer('RN50')


for year in single_sample:
    # Split 1886 into 18 86
    split_year = year[:2] + " " + year[2:]
    tokens = tokenizer(split_year).to(device)
    print(f"RN50 tokenization of {year} → {split_year} → {tokens}")


with torch.no_grad():
    print(f"Tokens shape: {tokens.shape}")
    
    image_features = model.encode_image(image)
    text_features = model.encode_text(tokens)
    
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    
    inner_product = image_features @ text_features.T
    print(f"Inner product shape: {inner_product.shape}")
    
    
# Output:
# RN50 tokenization of 1886 → 18 86 → tensor([[49406,   272,   279,   279,   277, 49407,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0]], device='mps:0')
# Tokens shape: torch.Size([1, 77])
# Image features shape: torch.Size([1, 1024])
# Text features shape: torch.Size([1, 1024])
# Inner product shape: torch.Size([1, 1])