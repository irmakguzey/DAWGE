from torchvision import models

# Script to return all pretrained models in torchvision.models module
# TODO: make sure how to use these - set the input and output sizes
def resnet18(pretrained : bool):
    if pretrained:
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18()

def resnet34(pretrained : bool):
    if pretrained:
        return models.resnet34(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet34()


