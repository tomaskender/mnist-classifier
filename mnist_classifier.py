from torch import nn, save, load, argmax, device, cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from os.path import exists
from PIL import Image


MODEL_FILENAME = 'model.pt'
TEST_FILENAME = 'test.jpg'
DEVICE = device('cuda' if cuda.is_available() else 'cpu')
print("Using CUDA:", cuda.is_available())
train = datasets.MNIST(root='./data', download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, X):
        return self.model(X)

classifier = Classifier().to(DEVICE)
optimizer = Adam(classifier.parameters(), 1e-3)
loss = nn.CrossEntropyLoss()

if __name__ == '__main__':
    if not exists(MODEL_FILENAME):
        for epoch in range(10):
            for batch in dataset:
                X, Y = batch
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                Y_hat = classifier(X)
                l = loss(Y_hat, Y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

            print(f"Epoch #{epoch+1}: loss {l.item()}")

        with open(MODEL_FILENAME, 'wb') as f:
            save(classifier.state_dict(), f)
        
    with open(MODEL_FILENAME, 'rb') as f:
        state_dict = load(f)
        classifier.load_state_dict(state_dict)
        img = Image.open(TEST_FILENAME)
        tensor = ToTensor()(img).unsqueeze(0).to(DEVICE)
        print('Classified as: ', argmax(classifier(tensor)))
