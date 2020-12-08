import torch
import torch.nn as nn
import torchvision


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



class AlexNetFinetuning(nn.Module):
    
    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(AlexNetFinetuning, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        self.classifier = model.classifier

        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])

        self.classifier[6] = nn.Linear(4096, num_classes)
        init_layer(self.classifier[6])
        
        print('\n<< SER AlexNet Finetuning model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return out

   

class AlexNetGAP(nn.Module):
   
    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(AlexNetGAP, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        
        #Global average pooling layer
        self.classifier = nn.Sequential(
                                        nn.Conv2d(256, num_classes, kernel_size=(1,1)),
                                        nn.AvgPool2d(6)
                                        )
        init_layer(self.classifier[0])


        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])
        
        print('\n<< SER AlexNet GAP model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        out = self.classifier(x).squeeze(-1).squeeze(-1)

        return out
