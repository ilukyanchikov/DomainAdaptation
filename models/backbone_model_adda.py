from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class LeNet(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNet, self).__init__()

        self.restored = False

        self.features = nn.Sequential(

            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
            nn.ReLU()
        )
        self.pooling = nn.Sequential(
            Flatten(),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU()
        )

    def forward(self, input):
        """Forward the LeNet."""
        features = self.features(input)
        feat = self.pooling(features)
        return feat


def get_lenet():
    lenet = LeNet()
    features = lenet.features
    pooling = lenet.pooling
    classifier = nn.Sequential(
        nn.Dropout2d(),
        nn.Linear(500, 10),
    )
    classifier_layer_ids = [3]
    return features, pooling, classifier, classifier_layer_ids, 500, None