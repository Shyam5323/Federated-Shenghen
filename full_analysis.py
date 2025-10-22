import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


MODEL_PATHS = {
    "Centralized": "models/best_resnet18_mura.pth",
    "Federated IID": "models/best_federated_iid.pth",
    "Federated Label Skew": "models/best_federated_label_skew.pth",
    "Federated Pathological": "models/best_federated_pathological_non_iid.pth"
}

VALID_CSV_PATH = os.path.join('MURA-v1.1', 'valid_image_paths.csv')

NUM_GRAD_CAM_IMAGES = 5 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



class MuraValidationDataset(Dataset):
    """Custom Dataset that also returns the image path for visualization."""
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file, header=None, names=['path'])
        self.transform = transform
        self.labels = [1 if 'positive' in path else 0 for path in self.dataframe['path']]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        return image_tensor, label, img_path 

def load_model(path, device):
    """Loads a ResNet-18 model with the saved state dictionary."""
    model = resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_ftrs, 1))
    
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Successfully loaded model from {path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}. Please check the path.")
        return None
        
    model = model.to(device)
    model.eval()
    return model

def get_predictions(model, dataloader, device):
    """Runs the model on the dataloader and returns labels, predictions, and probabilities."""
    all_labels = []
    all_preds_binary = []
    all_preds_probs = []
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            binary_preds = (probs > 0.5).astype(int)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds_binary.extend(binary_preds)
            all_preds_probs.extend(probs)
            
    return np.array(all_labels), np.array(all_preds_binary), np.array(all_preds_probs)


if __name__ == '__main__':
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_dataset = MuraValidationDataset(VALID_CSV_PATH, transform=valid_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    roc_results = {}

    print("\n--- Generating Confusion Matrices and Classification Reports ---")
    
    all_models = {}
    for name, path in MODEL_PATHS.items():
        model = load_model(path, DEVICE)
        if model is None: continue
        
        all_models[name] = model 
        
        labels, preds_binary, preds_probs = get_predictions(model, valid_loader, DEVICE)
        roc_results[name] = (labels, preds_probs)

        print(f"\n--- Results for: {name} ---")
        
        print(classification_report(labels, preds_binary, target_names=['Negative (0)', 'Positive (1)']))
        
        cm = confusion_matrix(labels, preds_binary)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Confusion Matrix for {name} Model')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        cm_filename = f'confusion_matrix_{name.lower().replace(" ", "_")}.png'
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        plt.close() # Close figure to avoid displaying it now
        print(f"✅ Confusion matrix saved to {cm_filename}")

    print("\n--- Generating Combined ROC Curve Plot ---")
    plt.figure(figsize=(10, 8))
    
    for name, (labels, preds_probs) in roc_results.items():
        fpr, tpr, _ = roc_curve(labels, preds_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='No Skill (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    roc_filename = 'roc_curves.png'
    plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Combined ROC curve plot saved to {roc_filename}")


    print("\n--- Generating Grad-CAM Visualizations ---")
    
    grad_cam_samples = [valid_dataset[i] for i in range(NUM_GRAD_CAM_IMAGES)]

    for name, model in all_models.items():
        cam_extractor = GradCAM(model, target_layer=model.layer4)
        
        for i, (img_tensor, label, img_path) in enumerate(grad_cam_samples):
            output = model(img_tensor.unsqueeze(0).to(DEVICE))
            prob = torch.sigmoid(output).item()
            pred_class = 1 if prob > 0.5 else 0

            activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
            
            pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
            result = overlay_mask(pil_img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.3)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(result)
            plt.title(f'{name}\nImg: {i} | Actual: {label} | Pred: {pred_class} ({prob:.2f})')
            plt.axis('off')
            grad_cam_filename = f'grad_cam_{name.lower().replace(" ", "_")}_image_{i}.png'
            plt.savefig(grad_cam_filename, dpi=300, bbox_inches='tight')
            plt.close()

    print(f"✅ Grad-CAM visualizations saved for {NUM_GRAD_CAM_IMAGES} images.")
    print("\nAdvanced evaluation complete!")