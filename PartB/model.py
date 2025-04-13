from torchvision.models import (
    resnet50, ResNet50_Weights,
    googlenet, GoogLeNet_Weights,
    inception_v3, Inception_V3_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    vit_b_32, ViT_B_32_Weights,
    vgg16, VGG16_Weights
)
import torch.nn as nn

class PretrainedModel(nn.Module):
    def __init__(self, model_name, num_classes=10):
        super(PretrainedModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

    def get_model(self):
        model = None
        if self.model_name == 'vgg16':
            weights = VGG16_Weights.DEFAULT
            model = vgg16(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        elif self.model_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == 'googlenet':
            weights = GoogLeNet_Weights.DEFAULT
            model = googlenet(weights=weights, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == 'inception_v3':
            weights = Inception_V3_Weights.DEFAULT
            model = inception_v3(weights=weights, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == 'efficientnet':
            weights = EfficientNet_V2_S_Weights.DEFAULT
            model = efficientnet_v2_s(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        elif self.model_name == 'vit':
            weights = ViT_B_32_Weights.DEFAULT
            model = vit_b_32(weights=weights)
            model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
        else:
            raise ValueError("Model architecture not supported.")
        
        return model

    def get_trainable_model(self, strategy='last', k=None):
        model = self.get_model()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        if strategy == 'last':
            # Only the final layer remains trainable (already replaced)
            if self.model_name in ['resnet50', 'googlenet', 'inception_v3']:
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif self.model_name == 'vgg16':
                for param in model.classifier[-1].parameters():
                    param.requires_grad = True
            elif self.model_name == 'efficientnet':
                for param in model.classifier[-1].parameters():
                    param.requires_grad = True
            elif self.model_name == 'vit':
                for param in model.heads.head.parameters():
                    param.requires_grad = True

        elif strategy == 'last_k':
            if k is None:
                raise ValueError("Please provide the value of k for 'last_k' strategy.")
            params = list(model.named_parameters())
            for name, param in params[-k:]:
                param.requires_grad = True

        elif strategy == 'full':
            for param in model.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return model
