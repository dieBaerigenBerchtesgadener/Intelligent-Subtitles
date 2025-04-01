import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score, roc_auc_score, recall_score, fbeta_score
import matplotlib.pyplot as plt
import wandb
from types import SimpleNamespace
import yaml
import joblib

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

class BinaryClassifier(nn.Module):
    def __init__(self, input_features, config):
        super(BinaryClassifier, self).__init__()
        self.config = config

        layers = []
        current_size = input_features

        for hidden_size in self.config.hidden_layers:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate)
            ])
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Custom Loss Functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return torch.mean(focal_loss)

class FocalLossWithSigmoid(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return self.focal(inputs, targets)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1)
        inputs = torch.clamp(inputs, self.clip, 1 - self.clip)

        pt_pos = torch.where(targets == 1, inputs, torch.ones_like(inputs))
        loss_pos = -torch.log(pt_pos) * torch.pow(1 - pt_pos, self.gamma_pos) * targets

        pt_neg = torch.where(targets == 0, 1 - inputs, torch.ones_like(inputs))
        loss_neg = -torch.log(pt_neg) * torch.pow(1 - pt_neg, self.gamma_neg) * (1 - targets)

        return torch.mean(loss_pos + loss_neg)

class BCEWithSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return self.bce(inputs, targets)

class AsymmetricLossWithSigmoid(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.asl = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return self.asl(inputs, targets)

class LLoss(nn.Module):
    def __init__(self, beta=1):
        super(LLoss, self).__init__()
        self.beta = beta

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1)
        L = 1 + torch.pow(inputs - targets, 2)
        L = torch.log(L)
        return torch.mean(L)

class MLoss(nn.Module):
    def __init__(self, beta=1):
        super(MLoss, self).__init__()
        self.beta = beta

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1)
        M = torch.abs(inputs - targets)
        M = 1 - torch.exp(-self.beta * M)
        return torch.mean(M)

# Loss Function Factory
def get_loss_function(loss_name, device, loss_params, y_train=None):
    if loss_name == "BCE":
        return BCEWithSigmoid().to(device)
    elif loss_name == "BCEWithLogits":
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    elif loss_name == "Focal":
        return FocalLossWithSigmoid(
            alpha=loss_params["focal_alpha"],
            gamma=loss_params["focal_gamma"]
        ).to(device)
    elif loss_name == "Asymmetric":
        return AsymmetricLossWithSigmoid(
            gamma_neg=loss_params["asymmetric_gamma_neg"],
            gamma_pos=loss_params["asymmetric_gamma_pos"],
            clip=loss_params["asymmetric_clip"]
        ).to(device)
    elif loss_name == "L":
        return LLoss(beta=loss_params["l_beta"]).to(device)
    elif loss_name == "M":
        return MLoss(beta=loss_params["m_beta"]).to(device)
    else:
        raise ValueError(f"Unbekannte Loss-Funktion: {loss_name}")

def get_optimizer(model, optimizer_name, learning_rate, weight_decay):
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} nicht unterstützt")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions.extend((torch.sigmoid(outputs) > 0.5).squeeze().cpu().detach().numpy())
        true_labels.extend(y_batch.cpu().numpy())

    f1 = f1_score(true_labels, predictions)
    f2 = fbeta_score(true_labels, predictions, beta=2.0)

    return total_loss / len(train_loader), {"f1": f1, "f2": f2}

def calculate_recall_pos(all_targets, all_preds):
    return recall_score(all_targets, all_preds, pos_label=1)

def evaluate(model, val_loader, device, criterion=None, is_final=False, is_test=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            if not is_final:
                loss = criterion(outputs.squeeze(), y_batch)
                total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.view(-1).cpu().numpy())

    f1 = f1_score(all_targets, all_preds)
    f2 = fbeta_score(all_targets, all_preds, beta=2.0)

    if is_final:
        bacc = balanced_accuracy_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)

        print('\nFinale Evaluierung:')
        print(f'F1-Score: {f1:.4f}')
        print(f'F2-Score: {f2:.4f}')
        print(f'Balanced Accuracy: {bacc:.4f}')
        print(f'ROC-AUC: {auc:.4f}')

        if not is_test:
            wandb.log({
                "final_f1": f1,
                "final_f2": f2,
                "final_balanced_accuracy": bacc,
                "final_roc_auc": auc
            })

        return f1, f2, bacc, auc
    else:
        avg_loss = total_loss / len(val_loader)
        return avg_loss, {"f1": f1, "f2": f2}

def train_model(model, train_loader, val_loader, device, config, y_train):
    epochs = config.epochs
    patience = config.early_stopping_patience

    criterion = get_loss_function(config.loss_function, device, config.loss_params, y_train)
    optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f2 = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_f1s = []
    train_f2s = []
    val_f1s = []
    val_f2s = []

    for epoch in range(epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, device, criterion, is_final=False)

        val_f1 = val_metrics["f1"]
        val_f2 = val_metrics["f2"]

        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val F2: {val_f2:.4f}')

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_f2': val_f2
        })

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_metrics["f1"])
        train_f2s.append(train_metrics["f2"])
        val_f1s.append(val_f1)
        val_f2s.append(val_f2)

        if val_f2 > best_val_f2:
            best_val_f2 = val_f2
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save('best_model.pth')
            print(f'Epoch {epoch:03d}: New best model saved with F2 = {val_f2:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(torch.load('best_model.pth'))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_f2s, label='Train FBeta')
    plt.plot(val_f2s, label='Val FBeta')
    plt.title('Recall Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Recall Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions.extend((torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy())
            true_labels.extend(y_batch.cpu().numpy())

    print("\nFinal Classification Report:")
    print(classification_report(true_labels, predictions))

    return model

def prepare_data(df, device, config):
    X = df[['audio_complexity', 'word_complexity', 'sentence_complexity', 'word_importance', 'word_occurrence']].values
    y = df['display'].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = torch.FloatTensor(X_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    y_val = torch.FloatTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Hier wird geprüft, ob config None ist (passiert wenn load_model True ist)
    if config is not None and config.weighted_sampler:
        class_counts = np.bincount(y_train.int().cpu().numpy())
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        samples_weights = weights[y_train.int().cpu()]

        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            drop_last=True
        )
    else:
        # Wenn config None ist oder weighted_sampler False ist, wird ein normaler DataLoader ohne WeightedSampler verwendet
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size if config is not None else 16,  # Default batch size, wenn config None ist
            shuffle=True,
            drop_last=True
        )

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size if config is not None else 16, shuffle=False, drop_last=True) # Default batch size

    return train_loader, val_loader, y_train, y_val

def start_training(df, device):
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_f2',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 0.0001,
                'max': 0.01
            },
            'batch_size': {
                'values': [8, 16, 32, 64]
            },
            'optimizer': {
                'values': ['AdamW', 'Adam', 'RMSprop']
            },
            'weight_decay': {
                'min': 0.001,
                'max': 0.1
            },
            'loss_function': {
                'values': ['BCE', 'BCEWithLogits']
            },
            'weighted_sampler': {
                'values': [True, False]
            },
            'hidden_layers': {
                'values': [[16, 8], [8, 16], [32, 16], [64, 32], [8, 4, 2], [16, 8, 4], [32, 16, 8], [64, 32, 16], [128, 64, 32], [256, 128, 64], [64, 128, 64], [32, 64, 32], [16, 32, 16], [8, 16, 8], [4, 8, 8, 4], [4, 8, 16, 8], [8, 16, 16, 8], [8, 16, 32, 16, 8]]
            },
            'dropout_rate': {
                'min': 0.1,
                'max': 0.8
            },
            'epochs': {
                'value': 150
            },
            'early_stopping_patience': {
                'value': 20
            }
        }
    }

    wandb.init(
        project="Intelligent Subtitles Simple NN 7",
        config=sweep_config
    )

    train_loader, val_loader, y_train, y_val = prepare_data(df, device, wandb.config)

    num_features = train_loader.dataset.tensors[0].shape[1]
    model = BinaryClassifier(input_features=num_features, config=wandb.config).to(device)
    train_model(model, train_loader, val_loader, device, wandb.config, y_train)

    model.load_state_dict(torch.load('best_model.pth'))
    criterion = get_loss_function(wandb.config.loss_function, device, wandb.config.loss_params, y_train)
    evaluate(model, val_loader, device, criterion, is_final=True, is_test=False)
    wandb.finish()
    return model

def evaluate_model(model, val_loader, device, wandb_run_path = "/humorless5218-gymnasium-berchtesgaden/Intelligent Subtitles Simple NN 5/swdvym3w"):
    # If wandb_run_path is not None, load the config from wandb
    if wandb_run_path:
        api = wandb.Api()
        run = api.run(wandb_run_path)
        config = SimpleNamespace(**run.config)
        run.file("best_model.pth").download(replace=True)
    else: # Use default config if no path is specified
      config = SimpleNamespace(
          learning_rate=0.004252027091067649,
          architecture="MLP",
          dataset="3boys",
          epochs=150,
          batch_size=16,
          optimizer="AdamW",
          weight_decay=0.0344653149484944,
          loss_function="BCEWithLogits",
          loss_params={
              "focal_alpha": 1,
              "focal_gamma": 2,
              "asymmetric_gamma_neg": 4,
              "asymmetric_gamma_pos": 1,
              "asymmetric_clip": 0.05,
              "l_beta": 1,
              "m_beta": 1
          },
          weighted_sampler=True,
          hidden_layers=[64, 128, 64],
          dropout_rate=0.14279874595232844,
          early_stopping_patience=20,
          beta_score=2.0
      )

    # Load the model
    model = BinaryClassifier(input_features=5, config=config).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # Perform evaluation
    evaluate(model, val_loader, device, is_final=True, is_test=True)

    predictions = []
    true_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions.extend((torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy())
            true_labels.extend(y_batch.cpu().numpy())

    print("\nFinal Classification Report:")
    print(classification_report(true_labels, predictions))

    return model, config

# Modell laden (nur einmal beim Start der App)
#api = wandb.Api()
#run = api.run("/humorless5218-gymnasium-berchtesgaden/Intelligent Subtitles Simple NN 5/swdvym3w")
#wandb.config = SimpleNamespace(**run.config)
#run.file("best_model.pth").download(replace=True)
#model = BinaryClassifier(input_features=5, config=wandb.config).to(device)
#model.load_state_dict(torch.load('best_model.pth', map_location=device))
#model.eval()  # In den Evaluationsmodus wechseln

def predict_with_bias(df, device, type="GB", bias=0.0, batch_size=64):
    """
    Funktion zur Vorhersage mit Bias für Gradient Boosting.
    
    :param df: DataFrame mit den Features.
    :param model: Geladenes Gradient Boosting Modell.
    :param bias: Bias-Wert, der zu den Wahrscheinlichkeiten addiert wird.
    :param batch_size: Batch-Größe für die Verarbeitung.
    :return: Liste der Vorhersagen."
    """
    if type=="GB":
        name = "gradient_boosting_model"
        features = df[['audio_complexity', 'word_occurrence', 'word_complexity', 
                'sentence_complexity', 'word_importance', 'speed',
                'ambient_volume', 'speech_volume', 'frequency', 'audio_level', 'language_level']].values

        model = joblib.load(f'{name}.pkl')

        predictions = model.predict(features)
        
        # Wahrscheinlichkeiten ermitteln (falls benötigt)
        probabilities = model.predict_proba(features)[:,1]
        
        # Bias auf die Wahrscheinlichkeiten anwenden (falls gewünscht)
        biased_probabilities = probabilities + bias
        
        # Klassen basierend auf den biased Wahrscheinlichkeiten vorhersagen
        predictions = (biased_probabilities > 0.5).astype(int)
        
        return predictions
    
    if type=="NN":
        name = "best_model"
        # Load configuration
        with open(f"{name}.yml", "r") as file:
            config = yaml.safe_load(file)

        config = SimpleNamespace(**config)

        # Modell laden
        model = BinaryClassifier(input_features=11, config=config).to(device)
        model.load_state_dict(torch.load(f'{name}.pth', map_location=device))
        model.eval()

        features = df[['audio_complexity', 'word_occurrence', 'word_complexity', 
                'sentence_complexity', 'word_importance', 'speed',
                'ambient_volume', 'speech_volume', 'frequency', 'audio_level', 'language_level']].astype('float32').values
        features = torch.tensor(features, dtype=torch.float32).to(device)
        predictions = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size]
                outputs = model(batch)
                outputs += bias
                predictions.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten())

        return predictions