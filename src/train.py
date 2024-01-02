from get_input_arg import get_input_args
from architecture_model import get_classifier
from data_loader_trainer import get_data_loader_trainer
from torchvision import models
import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
from torch import optim
import os


def main_loop():
    my_input = get_input_args()
    trainloader, testloader, train_data, _ = get_data_loader_trainer()

    # model = models.densenet121(weights=True)
    match my_input.arch:
        case "densenet201":
            model = torchvision.models.densenet201(
                weights="DEFAULT"
            )  # output 1920 neurons
            classifier = get_classifier(my_input.hidden_units, 1920)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = classifier
        case "alexnet":
            model = torchvision.models.alexnet(
                weights="DEFAULT"
            )  # lr = 0.00006 ,epoch = 7
            classifier = get_classifier(my_input.hidden_units, 9216)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = classifier
        case "efficientnet":
            model = torchvision.models.efficientnet_v2_s(
                weights="DEFAULT"
            )  # lr = 0.00006 ,epoch = 7
            classifier = get_classifier(my_input.hidden_units, 1280)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = classifier

    model.train()

    if my_input.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    else:
        device = torch.device("cpu")

    print(device)
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=my_input.learning_rate,
    )

    model.to(device)
    epochs = my_input.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}"
                )
                running_loss = 0
                model.train()

    model.class_to_idx = train_data.class_to_idx

    if not os.path.exists(my_input.save_dir):
        os.makedirs(my_input.save_dir)
        torch.save(model, my_input.save_dir + "/" + "barak_model.pth")
    else:
        torch.save(model, my_input.save_dir + "/" + "barak_model.pth")


if __name__ == "__main__":
    main_loop()
