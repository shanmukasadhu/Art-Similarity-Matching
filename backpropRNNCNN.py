import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Define the CNNBlock and CNNplusRNNBlock classes
class CNNBlock(nn.Module):
    def __init__(self, output_dim):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

class CNNplusRNNBlock(nn.Module):
    def __init__(self, output=128, hidden=128, temperature=1.0):
        super(CNNplusRNNBlock, self).__init__()
        self.cnnblock = CNNBlock(output)
        self.rnn = nn.LSTM(input_size=output, hidden_size=hidden, batch_first=True)
        self.digit_head = nn.Linear(hidden, 10)
        self.temperature = temperature

    def forward(self, x):
        batch_size = x.size(0)
        cnnblock = self.cnnblock(x)
        rnn_input = cnnblock.unsqueeze(1).repeat(1, 4, 1)
        rnn_out, _ = self.rnn(rnn_input)
        logits = self.digit_head(rnn_out)
        gumbel_samples = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        return gumbel_samples, logits

def decode_gumbel_digits(one_hot_vectors):
    preds = one_hot_vectors.argmax(dim=-1)
    output = []
    for pred in preds:
        digits = ''.join(str(d.item()) for d in pred)
        output.append(f"The year: {digits}")
    return output

# Load and preprocess the image
image = Image.open("wikiart/Romanticism/ivan-aivazovsky_the-tempest-1886.jpg").convert('RGB')

# Preprocessing for CNN model
cnn_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Apply CNN transform
cnn_input = cnn_transform(image).unsqueeze(0)  # Shape: (1, 3, 28, 28)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNplusRNNBlock().to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze CLIP model weights
for param in clip_model.parameters():
    param.requires_grad = False

# Set up optimizer for CNNplusRNNBlock
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training step
model.train()
clip_model.eval()

# Move input to device
cnn_input = cnn_input.to(device)

# Forward pass through CNNplusRNNBlock
gumbel_output, logits = model(cnn_input)  # gumbel_output: (batch_size, 4, 10), logits: (batch_size, 4, 10)

# Decode for CLIP
decoded_texts = decode_gumbel_digits(gumbel_output)  # e.g., ["The year: 1886"]

# Prepare inputs for CLIP
inputs = clip_processor(text=decoded_texts, images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Forward pass through CLIP
with torch.no_grad():
    outputs = clip_model(**inputs)
    image_embedding = outputs.image_embeds  # Shape: (batch_size, embed_dim)
    text_embedding = outputs.text_embeds    # Shape: (batch_size, embed_dim)

# Compute cosine similarity
similarity = torch.cosine_similarity(image_embedding, text_embedding)  
# image_embedding @ text_embedding.T




similarity_loss = -similarity.mean()

# Combined loss
loss = similarity_loss


optimizer.zero_grad()
loss.backward()
optimizer.step()

# Print results
for text, sim in zip(decoded_texts, similarity):
    print(f"{text}, Similarity: {sim.item():.4f}")