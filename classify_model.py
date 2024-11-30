class ClassifyModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassifyModel, self).__init__()

        # Load the pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)  # Replace the final fully connected layer

    def forward(self, x):
        return self.model(x)
