import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import gc

class AdversialLoss(nn.Module):
    def __init__(self, cartoon_labels, fake_labels):
        super(AdversialLoss, self).__init__()

         # Etiquetas de las cartoon reales
        self.cartoon_labels = cartoon_labels

        # Etiquetas de las imagenes "cartoon" generadas
        self.fake_labels = fake_labels

        # Binary crossentropy
        self.base_loss = nn.BCEWithLogitsLoss()

    def forward(self, cartoon, generated_f, edge_f):
        # cartoon -> salida del discriminador para cartoons reales
        # generated_f -> salida del discriminador para imágenes generadas
        # edge_f -> salida del discriminador para cartoons smoothed
        
        # loss para cartoons reales (debe clasificarlos como REAL)
        D_cartoon_loss = self.base_loss(cartoon, self.cartoon_labels)

        # loss para imágenes generadas (debe clasificarlas como FAKE)
        D_generated_fake_loss = self.base_loss(generated_f, self.fake_labels)

        # loss para cartoons smoothed (también deben ser FAKE)
        D_edge_fake_loss = self.base_loss(edge_f, self.fake_labels)
        
        # suma total de losses del discriminador
        return D_cartoon_loss + D_generated_fake_loss + D_edge_fake_loss
        

class ContentLoss(nn.Module):
    def __init__(self, omega=10):
        super(ContentLoss, self).__init__()

        # Se usa L1 por que hay menos blur
        self.base_loss = nn.L1Loss()

        # Peso que se le dara a la content loss
        self.omega = omega

        # Usar solo las primeras 25 capas
        perception = list(vgg16(pretrained=True).features)[:25]
        self.perception = nn.Sequential(*perception).eval()

        # Congelar VGG16
        for param in self.perception.parameters():
            param.requires_grad = False

        gc.collect()

    def forward(self, x1, x2):
        # x1 -> imagen generada
        # x2 -> imagen real
        
        # extraer features perceptuales con VGG
        x1 = self.perception(x1)
        x2 = self.perception(x2)
        
        # comparar features en vez de píxeles
        return self.omega * self.base_loss(x1, x2)
        