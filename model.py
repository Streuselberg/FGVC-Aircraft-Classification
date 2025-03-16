from os import path
from random import seed
import torch
import torch.nn as nn
from torch import device
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim

from dataset import FGVAircraftDataset, BATCH_SIZE, images_dir, create_training_datasets, create_validation_datasets
from transforms import train_transform, val_transform


class MoEModel(nn.Module):
    """A mixture of experts model designed for training on the FGVA Aircraft dataset"""

    def __init__(self, num_classes_dict):
        super(MoEModel, self).__init__()
        self.backbone = models.efficientnet_b3(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        # define an 'expert' for each identification stratum
        self.experts = nn.ModuleDict({
            'family': nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Linear(512, num_classes_dict['family'])),
            'manufacturer': nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(),
                                          nn.Linear(512, num_classes_dict['manufacturer'])),
            'variant': nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(),
                                     nn.Linear(512, num_classes_dict['variant']))
        })
        self.expert_weights_paths = {
            'family': path.join("weights", "family_expert.pth"),
            'manufacturer': path.join("weights", "manufacturer_expert.pth"),
            'variant': path.join("weights", "variant_expert.pth")
        }
        self.expert_names = ['family', 'manufacturer', 'variant']
        self.gate = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=-1)
        )
        self.filename = "moe_efficientnet_b3"
        self.weights_path = f"{self.filename}.pth"
        self.export_path = f"{self.filename}.onnx"
        self.train_datasets = create_training_datasets()
        self.val_datasets = create_validation_datasets()

    def export(self, dummy_input):
        """Export the model in an ONNX format"""
        torch.onnx.export(
            self,
            dummy_input,
            self.export_path,
            input_names=["input"],
            output_names=["family_output", "manufacturer_output", "variant_output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "family_output": {0: "batch_size"},
                "manufacturer_output": {0: "batch_size"},
                "variant_output": {0: "batch_size"},
            },
            opset_version=11
        )
        print(f"Model exported to {self.export_path}")

    @staticmethod
    def set_seed(s: int):
        torch.manual_seed(s)
        seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    def load_weights(self):
        """Load the weights of the model"""
        if path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}")
            self.load_state_dict(torch.load(self.weights_path))
        else:
            raise FileNotFoundError(
                f"Model weights not found at {self.weights_path}. Please ensure the model has been trained and saved.")

    def train_for(self, epochs: int, device):
        """train the model for a specified number of epochs"""
        # Setup learning parameters
        train_loaders = {k: DataLoader(v, batch_size=BATCH_SIZE, shuffle=True) for k, v in self.train_datasets.items()}
        val_loaders = {k: DataLoader(v, batch_size=BATCH_SIZE, shuffle=False) for k, v in self.val_datasets.items()}
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        best_acc = {label: 0.0 for label in self.expert_names}
        """keep track of best validation accuracy for each expert across epochs"""
        expert_loss_weights = {'family': 1.5, 'manufacturer': 1.5, 'variant': 2.0}

        # Start training for NUM_EPOCHS
        for epoch in range(epochs):
            self.train()  # put the model into training mode
            train_iterators = {k: iter(v) for k, v in train_loaders.items()}
            train_accs = {label: 0 for label in self.expert_names}
            train_totals = {label: 0 for label in self.expert_names}

            while True:
                try:
                    batches = {k: next(v) for k, v in train_iterators.items()}
                    optimizer.zero_grad()
                    total_loss = 0.0
                    gate_scores_sum = torch.zeros(3).to(device)
                    gate_count = 0

                    for expert_task in self.expert_names:
                        images, labels = batches[expert_task]
                        images, labels = images.to(device), labels.to(device)
                        outputs, gate_scores = self(images)
                        task_loss = criterion(outputs[expert_task], labels)
                        weight = expert_loss_weights[expert_task]
                        total_loss += weight * task_loss

                        _, predicted = torch.max(outputs[expert_task], 1)
                        train_totals[expert_task] += labels.size(0)
                        train_accs[expert_task] += (predicted == labels).sum().item()

                        gate_scores_sum += gate_scores.sum(dim=0)
                        gate_count += gate_scores.size(0)

                    total_loss /= 5.0
                    total_loss.backward()
                    optimizer.step()
                except StopIteration:
                    break

            train_accs = {k: 100 * v / train_totals[k] for k, v in train_accs.items()}
            avg_gate_scores = gate_scores_sum / gate_count

            # Evaluate the model(s)
            self.eval()  # set model into evaluation mode
            val_accs = {}
            """store the accuracies for each expert during validation"""

            # Go through evaluation
            for expert_task in self.expert_names:
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loaders[expert_task]:
                        images, labels = images.to(device), labels.to(device)
                        outputs, _ = self(images)
                        _, predicted = torch.max(outputs[expert_task], 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                val_accs[expert_task] = 100 * val_correct / val_total

                # Save the weights for the expert if its accuracy bests current max
                if val_accs[expert_task] > best_acc[expert_task]:
                    best_acc[expert_task] = val_accs[expert_task]
                    torch.save(self.experts[expert_task].state_dict(),
                               self.expert_weights_paths[expert_task])
                    print(f"Saved {expert_task} expert weights to {self.expert_weights_paths[expert_task]}" +
                          f" with Val Acc: {val_accs[expert_task]:.2f}%")

            self.log_epoch(avg_gate_scores, epoch, epochs, train_accs, val_accs)

            # Save the whole model if all experts have improved accuracy
            if all(val_accs[task] > best_acc[task] - 0.01 for task in best_acc):
                torch.save(self.state_dict(), self.weights_path)
                print(f"Saved full model weights to {self.weights_path}")

            scheduler.step()
        return best_acc

    @staticmethod
    def log_epoch(avg_gate_scores, epoch, epochs, train_accs, val_accs):
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Family Train: {train_accs['family']:.2f}%, Val: {val_accs['family']:.2f}%, "
              f"Manuf Train: {train_accs['manufacturer']:.2f}%, Val: {val_accs['manufacturer']:.2f}%, "
              f"Var Train: {train_accs['variant']:.2f}%, Val: {val_accs['variant']:.2f}%, "
              f"Gate: {avg_gate_scores.tolist()}")

    def forward(self, datum):
        features = self.backbone(datum)
        gate_scores = self.gate(features)
        outputs = {task: expert(features) for task, expert in self.experts.items()}
        return outputs, gate_scores
