from torchvision import models

# Script to return all pretrained models in torchvision.models module
# TODO: make sure how to use these - set the input and output sizes
def resnet18(pretrained : bool):
    print(f'in resnet18 method, pretrained: {pretrained}')
    return models.resnet18(pretrained=pretrained) # TODO: check if this works - how to set the inputsize? 

def resnet34(pretrained : bool):
    print(f'in resnet34 method, pretrained: {pretrained}')
    return models.resnet34(pretrained=pretrained)


