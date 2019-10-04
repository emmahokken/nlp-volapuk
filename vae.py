import torch
import torch.nn as nn
from torchvision.utils import make_grid

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784):
        super().__init__()
        self.W3 = nn.Linear(input_dim, hidden_dim)
        self.W4 = nn.Linear(hidden_dim, z_dim)
        self.W5 = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean, std = None, None

        h = torch.tanh(self.W3(input))
        mean = self.W4(h)
        std = torch.exp(self.W5(h))**0.5

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784):
        super().__init__()
        self.W1 = nn.Linear(z_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = None

        y = self.sigmoid(self.W2(torch.tanh(self.W1(input))))

        mean = y

        # mean = torch.sum(input @ torch.log(y) + (1 - input) @ torch.log(1 - y))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784, device='cpu'):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim, input_dim).to(device=device)
        self.decoder = Decoder(hidden_dim, z_dim, input_dim).to(device=device)
        self.device = device

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        average_negative_elbo = None

        mean, std = self.encoder.forward(input)
        # print(self.device)
        # z = Normal(0,1).sample() * std + mean
        z = torch.randn(std.size(0), self.z_dim, device=self.device) * std + mean

        output = self.decoder.forward(z)

        # recon_loss = torch.mean(torch.log(mean_dec))
        # recon_loss = torch.nn.functional.binary_cross_entropy(input, target)
        recon_loss = torch.nn.functional.binary_cross_entropy(output, input, reduction='sum')
        reg_loss = - 0.5 * torch.sum(- std**2 - mean**2 + 1 + torch.log(std**2))
        # reg_loss = - 0.5 * torch.sum(- std**2 - mean**2 + 1 - torch.log(std**2))
        average_negative_elbo = (recon_loss + reg_loss) / input.size(0)

        # print(average_negative_elbo)

        return average_negative_elbo, z, output

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = None, None

        # sampled_ims = Bernoulli().sample()

        z = torch.randn(n_samples, self.z_dim)
        # z = z - torch.min(z)
        # z = z / torch.max(z)


        # z = Bernoulli(z).sample()

        im_means = self.decoder.forward(z)

        sampled_ims = torch.bernoulli(im_means)

        # im_means = torch.mean(sampled_ims,dim=0)

        return sampled_ims, im_means
