import os
import time
import copy
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


# ==========================================
# 1. Zarządzanie warstwami (Zamrażanie/Odmrażanie)
# ==========================================
def set_parameter_requires_grad(model, unfreeze_layer4=False):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True


# ==========================================
# 2. Pętla ucząca
# ==========================================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=5,
                best_acc_start=0.0):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = best_acc_start

    for epoch in range(num_epochs):
        print(f'Epoka {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Obecny Learning Rate: {current_lr:.6f}')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("-> Zapisano nowe najlepsze wagi modelu!")
        print()

    time_elapsed = time.time() - since
    print(f'Faza zakończona w {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Najlepsza dokładność (Val Acc): {best_acc:4f}\n')

    model.load_state_dict(best_model_wts)
    return model, best_acc


# ==========================================
# GŁÓWNY BLOK WYKONAWCZY
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}\n")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Ścieżka do folderu
    dataset_path = 'Apple_Disease_Dataset'

    # 0: Apple___Apple_scab, 1: Apple___Black_rot, 2: Apple___Cedar_apple_rust, 3: Apple___healthy
    image_datasets = {
        'train': ImageFolder(root=os.path.join(dataset_path, 'train'), transform=data_transforms['train']),
        'val': ImageFolder(root=os.path.join(dataset_path, 'test'), transform=data_transforms['val'])
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=2)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Klasy wykryte przez ImageFolder: {image_datasets['train'].class_to_idx}")
    print(f"Ilość zdjęć treningowych: {dataset_sizes['train']}, walidacyjnych (test): {dataset_sizes['val']}\n")

    # Inicjalizacja Modelu
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # FAZA 1: Feature Extraction
    print("=== ROZPOCZĘCIE FAZY 1: FEATURE EXTRACTION ===")
    set_parameter_requires_grad(model, unfreeze_layer4=False)
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=3)

    model, best_val_acc = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer1, scheduler1,
        device, num_epochs=10, best_acc_start=best_val_acc
    )

    # FAZA 2: Fine-Tuning
    print("=== ROZPOCZĘCIE FAZY 2: FINE-TUNING ===")
    set_parameter_requires_grad(model, unfreeze_layer4=True)
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=6)

    model, best_val_acc = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer2, scheduler2,
        device, num_epochs=50, best_acc_start=best_val_acc
    )

    torch.save(model.state_dict(), 'resnet18_apple_disease.pth')
    print("\nZapisano ulepszony model jako: resnet18_apple_disease.pth")