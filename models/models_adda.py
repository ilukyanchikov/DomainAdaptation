import torch
from torch import nn
from models.models import BaseModel
from models.backbone_model_adda import get_lenet


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class ADDAModel(BaseModel):
    def __init__(self):
        super(ADDAModel, self).__init__()
        self.features, self.pooling, self.class_classifier, \
        _, domain_input_len, self.classifier_before_domain_cnt = get_lenet()

        self.src_cnn = nn.Sequential(self.features,
                                     self.pooling)

        self.trg_cnn = nn.Sequential(self.features,
                                     self.pooling)

        self.discriminator = Discriminator(domain_input_len, 500, 2)

    def forward(self, input_data, domain='src'):
        if domain == 'src':
            cnn = self.src_cnn
        elif domain == 'trg':
            cnn = self.trg_cnn
        else:
            raise ValueError('Wrong domain {}'.format(domain))
        features = cnn(input_data)
        output_classifier = self.class_classifier(features)
        output_domain = self.discriminator(features)

        output = {
            "class": output_classifier,
            "domain": output_domain,
        }

        return output

    def predict_domain(self, src_data, trg_data):
        for (src_images, _), (trg_images, _) in zip(src_data, trg_data):
            features_src = self.model.src_cnn(src_images)
            features_trg = self.model.trg_cnn(trg_images)

            features = torch.cat((features_src, features_trg), 0)

            pred = self.model.discriminator(features.detach())

            source_len = len(src_images)
            target_len = len(trg_images)
            is_target_on_src = torch.ones(source_len, dtype=torch.long, device=self.device)
            is_target_on_trg = torch.zeros(target_len, dtype=torch.long, device=self.device)
            domain_labels = torch.cat((is_target_on_src, is_target_on_trg), 0)

    def predict(self, input_data, domain='src'):
        return self.forward(input_data, domain=domain)["class"]

    def copy_src_to_trg(self):
        self.trg_cnn.load_state_dict(self.src_cnn.state_dict())