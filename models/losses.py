import torch.nn as nn
import torchmetrics

# SSIM Loss Function
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)