import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, output_dim):
        super(CNNBlock, self).__init__()
        # Defining a Convolutional Block
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Final Fully Connected Layer
        self.fc = nn.Linear(64 * 7 * 7, output_dim)  # for 28x28 input

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)  # (batch_size, output_dim)

class CNNplusRNNBlock(nn.Module):
    def __init__(self, output=128, hidden=128, temperature=1.0):
        super(CNNplusRNNBlock, self).__init__()
        self.cnnblock = CNNBlock(output)
        self.rnn = nn.LSTM(input_size=output, hidden_size=hidden, batch_first=True)
        self.digit_head = nn.Linear(hidden, 10)
        self.temperature = temperature

    def forward(self, x):
        batch_size = x.size(0)
        cnnblock = self.cnnblock(x)  # (batch_size, cnn_out_dim)

        rnn_input = cnnblock.unsqueeze(1).repeat(1, 4, 1)  # (batch_size, 4, cnn_out_dim)

        rnn_out, _ = self.rnn(rnn_input)  # (batch_size, 4, hidden_dim)
        logits = self.digit_head(rnn_out)  # (batch_size, 4, 10)

        # Apply Gumbel-Softmax 
        gumbel_samples = F.gumbel_softmax(logits, tau=self.temperature, hard=True)  # (batch_size, 4, 10)
        return gumbel_samples  # one-hot digit vectors

def decode_gumbel_digits(one_hot_vectors):
    # one_hot_vectors: (batch_size, 4, 10)
    preds = one_hot_vectors.argmax(dim=-1)  # get index of 1s
    output = []
    for pred in preds:
        digits = ''.join(str(d.item()) for d in pred)
        output.append(f"The year: {digits}")
    return output


from PIL import Image
from torchvision import transforms

# Load the image
image = Image.open("wikiart/Romanticism/ivan-aivazovsky_the-tempest-1886.jpg").convert('RGB') 

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])

# Apply preprocessing
tensor_image = transform(image)  # shape: (1, 28, 28)

# Add batch dimension
input_tensor = tensor_image.unsqueeze(0)  # shape: (1, 1, 28, 28)

model = CNNplusRNNBlock()

gumbel_output = model(input_tensor)
decoded = decode_gumbel_digits(gumbel_output)
for text in decoded:
    print(text)