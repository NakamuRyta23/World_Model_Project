import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Encoder: Structured VAE Encoder
class VAEStructuredEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, grid_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_latent = nn.Linear(128 * grid_size * grid_size, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        z = self.fc_latent(x)
        return z

# Decoder: Structured VAE Decoder
class VAEStructuredDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, grid_size):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * grid_size * grid_size)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 128, 4, 4)  # Assuming grid size of 4x4
        z = torch.relu(self.deconv1(z))
        x_recon = torch.sigmoid(self.deconv2(z))
        return x_recon

# Transformer Decoder
class ObjectCentricTransformer(nn.Module):
    def __init__(self, latent_dim, n_heads, n_layers):
        super().__init__()
        self.transformer = nn.Transformer(d_model=latent_dim, nhead=n_heads, num_encoder_layers=n_layers, batch_first=True)

    def forward(self, z):
        return self.transformer(z, z)

# Hungarian Algorithm (Alignment Step)
def hungarian_alignment(cost_matrix):
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

# Loss Function
class ObjectWiseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_pred, z_true):
        loss = torch.abs(z_pred - z_true).mean()
        return loss

# Main OCVT Model
class OCVT(nn.Module):
    def __init__(self, input_dim, latent_dim, grid_size, n_heads, n_layers):
        super().__init__()
        self.encoder = VAEStructuredEncoder(input_dim, latent_dim, grid_size)
        self.transformer = ObjectCentricTransformer(latent_dim, n_heads, n_layers)
        self.decoder = VAEStructuredDecoder(latent_dim, input_dim, grid_size)

    def forward(self, x):
        z = self.encoder(x)
        z_trans = self.transformer(z.unsqueeze(1)).squeeze(1)  # Add temporal dimension
        x_recon = self.decoder(z_trans)
        return x_recon, z_trans

# Example Usage
if __name__ == "__main__":
    input_dim = 3
    latent_dim = 256
    grid_size = 4
    n_heads = 4
    n_layers = 4

    model = OCVT(input_dim, latent_dim, grid_size, n_heads, n_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = ObjectWiseLoss()

    # Dummy Data
    batch_size = 8
    x = torch.randn(batch_size, input_dim, 64, 64)

    for epoch in range(5):
        optimizer.zero_grad()
        x_recon, z_trans = model(x)
        loss = criterion(x_recon, x)  # Simple reconstruction loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
