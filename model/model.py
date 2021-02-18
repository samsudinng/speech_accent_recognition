import torch
import torch.nn as nn
import torchvision
from model.vggm import VGGM
from facenet_pytorch import InceptionResnetV1

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


class VGG16GAP(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(VGG16GAP, self).__init__()

        model = torchvision.models.vgg16(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool

        #Global average pooling layer
        self.classifier = nn.Sequential(
                                        nn.Conv2d(512, num_classes, kernel_size=(1,1)),
                                        nn.AvgPool2d(7)
                                        )
        init_layer(self.classifier[0])


        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
            init_layer(self.features[0])

        print('\n<< VGG16GAP model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        out = self.classifier(x).squeeze(-1).squeeze(-1)

        return out

class VGG16BnGAP(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(VGG16BnGAP, self).__init__()

        model = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool

        #Global average pooling layer
        self.classifier = nn.Sequential(
                                        nn.Conv2d(512, num_classes, kernel_size=(1,1)),
                                        nn.AvgPool2d(7)
                                        )
        init_layer(self.classifier[0])


        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
            init_layer(self.features[0])

        print('\n<< VGG16BnGAP model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        out = self.classifier(x).squeeze(-1).squeeze(-1)

        return out


class Resnet50Accent(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(Resnet50Accent, self).__init__()

        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc=nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        init_layer(model.fc)

        self.model= model
        print('\n<< Resnet50Accent model initialized >>\n')

    def forward(self, x):

        return self.model(x)

class Resnet34Accent(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(Resnet34Accent, self).__init__()
            
        model = torchvision.models.resnet34(pretrained=pretrained)
        model.fc=nn.Linear(in_features=512, out_features=num_classes, bias=True)
        init_layer(model.fc)
                                    
        self.model= model
        print('\n<< Resnet34Accent model initialized >>\n')
                                                    
    def forward(self, x):
                                                        
        return self.model(x)


class Resnet101Accent(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(Resnet101Accent, self).__init__()

        model = torchvision.models.resnet101(pretrained=pretrained)
        model.fc=nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        init_layer(model.fc)

        self.model= model
        print('\n<< Resnet101Accent model initialized >>\n')

    def forward(self, x):

        return self.model(x)

class Resnet152Accent(nn.Module):

    def __init__(self,num_classes=8, in_ch=3, pretrained=True):
        super(Resnet152Accent, self).__init__()

        model = torchvision.models.resnet152(pretrained=pretrained)
        model.fc=nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        init_layer(model.fc)

        self.model= model
        print('\n<< Resnet152Accent model initialized >>\n')

    def forward(self, x):

        return self.model(x)



class VGGMAccent(nn.Module):

    def __init__(self, n_classes = 8, pretrained_path = None):
        super(VGGMAccent, self).__init__()
        
        #initialize network
        model = VGGM()

        #initialize with pretrained weight
        if pretrained_path is not None:
            torch.load(pretrained_path)
            print(f"\nPre-trained weight loaded from {pretrained_path}\n")
        
        #initialize the last linear layer to n_classes
        model.classifier[2] = nn.Linear(1024, n_classes)

        self.features = model.features
        self.classifier = model.classifier

    def forward(self, inp):
        inp = inp[:,0,:,:].unsqueeze(1)
        inp=self.features(inp)
        inp=self.classifier(inp)
        return inp            
