"""
NAME: model.py
"""
from dataclasses import dataclass
from os import path, linesep
from random import seed

import torch
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, MofNCompleteColumn
from rich.table import Table, Column
from rich.text import Text
from torch.utils.data import DataLoader
from torchvision import models

from dataset import create_training_datasets, create_validation_datasets
from fgvc_config import ExpertName, Config


@dataclass
class ModelParameters:
    """Abstraction to consolidate parameters for model loaded from config"""

    def __init__(self):
        self.config = Config()
        self.datasets = {
            'train': create_training_datasets(self.config),
            'validation': create_validation_datasets(self.config)
        }

        self.loaders: dict = {
            'train': {
                k: DataLoader(v, batch_size=self.config.BATCH_SIZE, shuffle=True)
                for k, v in self.datasets['train'].items()
            },
            'val': {
                k: DataLoader(v, batch_size=self.config.BATCH_SIZE, shuffle=False)
                for k, v in self.datasets['validation'].items()
            }
        }


class MoEModel(torch.nn.Module):
    """A mixture of experts model designed for training on the FGVC Aircraft dataset"""

    def __init__(self, num_classes_dict, config: Config):
        super().__init__()
        self.config = config
        self.backbone = models.efficientnet_b3()
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = torch.nn.Identity()
        # define an 'expert' for each identification stratum
        self.experts = torch.nn.ModuleDict({
            expert.name: torch.nn.Sequential(torch.nn.Linear(in_features, 512),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(512, num_classes_dict[expert.name]))
            for expert in ExpertName
        })
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
            torch.nn.Softmax(dim=-1)
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.fgvc_parameters = ModelParameters()

    def export(self, dummy_input):
        """Export the model in an ONNX format"""
        torch.onnx.export(
            self,
            dummy_input,
            self.config.EXPORT_PATH,
            input_names=["input"],
            output_names=[expert.name + "_output" for expert in ExpertName],
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
    def set_seed(s: int) -> None:
        """
        sets the seed for reproducibility in training
        :param s: seed value
        """
        torch.manual_seed(s)
        seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    def load_weights(self):
        """Load the weights of the model"""
        if path.exists(self.config.MODEL_WEIGHTS_PATH):
            print(f"Loading weights from {self.config.MODEL_WEIGHTS_PATH}")
            self.load_state_dict(torch.load(self.config.MODEL_WEIGHTS_PATH))
        else:
            raise FileNotFoundError(
                f"Model weights not found at {self.config.MODEL_WEIGHTS_PATH}. Please ensure the model has been trained and saved.")

    def train_for(self, epochs: int, device, console: Console) -> dict[ExpertName, float]:
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.config.LEARNING_RATE, momentum=self.config.MOMENTUM)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.STEP_SIZE, gamma=self.config.GAMMA)
        best_acc = {label: 0.0 for label in ExpertName}
        expert_loss_weights = {ExpertName.family: 1.5, ExpertName.manufacturer: 1.5, ExpertName.variant: 2.0}

        # Setup Progress
        epoch_progress, epoch_task, phase_progress, phase_task, save_progress, save_task = self.create_rich_elements(
            console, epochs)
        prog_group = Panel(
            Group(epoch_progress, phase_progress, save_progress),
            title="FGVC Model Training Progress",
            title_align="center",
        )
        # State to track across epochs
        current_train_accs = {label: 0.0 for label in ExpertName}  # Last training accuracies
        save_messages = []  # List of save messages

        # Pure function to generate the layout
        def generate_layout():
            layout = Layout()
            layout.split_column(
                Layout(name="progress", ratio=1),
                Layout(name="accuracies", ratio=1),
                Layout(name="log", ratio=2)
            )
            # Progress bar
            layout["progress"].update(prog_group)

            # Training accuracies table
            acc_table = Table(title="Training Accuracies", expand=True)
            acc_table.add_column("Expert", ratio=2, justify="left")
            acc_table.add_column("Training Accuracy", ratio=1, justify="center")
            acc_table.add_column("Validation Accuracy", ratio=1, justify="right")
            for expert in ExpertName:
                acc_table.add_row(expert.name.capitalize(), f"[orange]{current_train_accs[expert]:.2f}%",
                                  f"[green]{best_acc[expert]:.2f}%")
            layout["accuracies"].update(acc_table)

            # Save log
            save_log = Panel(Text(linesep.join(save_messages)), title="Log",
                             title_align="center")  # Show last 5 messages
            layout["log"].update(save_log)
            return layout

        with Live(generate_layout(), console=console, auto_refresh=True, transient=False) as live:
            for epoch in range(epochs):
                epoch_progress.update(epoch_task, advance=1, epoch=epoch + 1, epochs=epochs)
                phase_progress.reset(phase_task)
                phase_progress.update(phase_task, advance=0, epoch=0, phase="Training")
                save_progress.reset(save_task)

                avg_gate_scores, train_accs = self.training_epoch(device, expert_loss_weights, optimizer)

                phase_progress.update(phase_task, phase="Validation", advance=1)

                val_accs = self.validation_epoch(best_acc, device, save_messages, save_progress, save_task)
                phase_progress.update(phase_task, advance=1, phase="Saving")
                save_messages.append(self.log_epoch(avg_gate_scores, epoch, epochs, train_accs, val_accs))

                if all(val_accs[task] > best_acc[task] - 0.01 for task in best_acc):
                    torch.save(self.state_dict(), self.config.MODEL_WEIGHTS_PATH)
                    for k, v in train_accs.items():
                        current_train_accs[k] = v
                    save_messages.append(f"Saved full model weights to {self.config.MODEL_WEIGHTS_PATH}")

                scheduler.step()
                live.update(generate_layout())
        return best_acc

    def validation_epoch(self, best_acc, device, save_messages, save_progress, save_task):
        self.eval()
        val_accs = {}
        for expert_task in ExpertName:
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in self.fgvc_parameters.loaders['val'][expert_task.name]:
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = self(images)
                    _, predicted = torch.max(outputs[expert_task.name], 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_accs[expert_task] = 100 * val_correct / val_total
            if val_accs[expert_task] > best_acc[expert_task]:
                best_acc[expert_task] = val_accs[expert_task]
                weight_path = self.config.expert_weights_paths[expert_task.name]
                torch.save(self.experts[expert_task.name].state_dict(), weight_path)
                save_progress.update(save_task, expert=expert_task.name, acc=val_accs[expert_task],
                                     visible=True)
                save_messages.append(
                    f"Saved {expert_task.name} weights to {weight_path} with Val Acc: {val_accs[expert_task]:.2f}%")
        return val_accs

    def training_epoch(self, device, expert_loss_weights, optimizer):
        """inner loop for a given epoch's training session"""
        self.train()
        train_accs = {label: 0 for label in ExpertName}
        train_iterators = {k: iter(v) for k, v in self.fgvc_parameters.loaders['train'].items()}
        train_totals = {label: 0 for label in ExpertName}
        while True:
            batch_data = {'total_loss': 0.0, 'gate_scores_sum': torch.zeros(3).to(device), 'gate_count': 0}
            try:
                batches = {k: next(v) for k, v in train_iterators.items()}
                optimizer.zero_grad()
                for expert_task in ExpertName:
                    images, labels = batches[expert_task.name]
                    images, labels = images.to(device), labels.to(device)
                    outputs, gate_scores = self(images)
                    task_loss = self.criterion(outputs[expert_task.name], labels)
                    weight = expert_loss_weights[expert_task]
                    batch_data['total_loss'] += weight * task_loss
                    _, predicted = torch.max(outputs[expert_task.name], 1)
                    train_totals[expert_task] += labels.size(0)
                    train_accs[expert_task] += (predicted == labels).sum().item()
                    batch_data['gate_scores_sum'] += gate_scores.sum(dim=0)
                    batch_data['gate_count'] += gate_scores.size(0)

                batch_data['total_loss'] /= 5.0
                batch_data['total_loss'].backward()
                optimizer.step()
            except StopIteration:
                break
        train_accs = {k: 100 * v / train_totals[k] for k, v in train_accs.items()}
        avg_gate_scores = batch_data['gate_scores_sum'] / batch_data['gate_count']
        return avg_gate_scores, train_accs

    def create_rich_elements(self, console, epochs):
        epoch_progress = Progress(
            TextColumn("[cyan]Epoch [white]{task.fields[epoch]}", table_column=Column(ratio=2)),
            SpinnerColumn(table_column=Column(ratio=1)),
            BarColumn(table_column=Column(ratio=3)),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MofNCompleteColumn(),
            console=console,
        )
        phase_progress = Progress(
            TextColumn("[cyan]Phase: [green]{task.fields[phase]}", table_column=Column(ratio=2)),
            SpinnerColumn(table_column=Column(ratio=1)),
            BarColumn(table_column=Column(ratio=3)),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%", table_column=Column(ratio=3)),
            console=console,
        )
        save_progress = Progress(
            TextColumn(
                "[green]Saving Model Weights for expert: [red]{task.fields[expert][/red] w/ Acc: {task.fields[acc]}%}",
                table_column=Column(ratio=3)),
            BarColumn(),
            console=console,
            expand=True,
            transient=True
        )
        epoch_task = epoch_progress.add_task("Epoch", total=epochs, epoch=1, total_epochs=epochs)
        phase_task = phase_progress.add_task("Phase", phase="Training", epoch=1, total=3)
        save_task = save_progress.add_task("Saving Model", total=1, expert="", acc=0.0, visible=False)
        return epoch_progress, epoch_task, phase_progress, phase_task, save_progress, save_task

    @staticmethod
    def log_epoch(avg_gate_scores, epoch, epochs, train_accs: dict[ExpertName, float],
                  val_accs: dict[ExpertName, float]) -> str:
        """format data for epoch"""
        scores = ", ".join(
            f"[blue]{expert.name}[/blue] Accuracies:[orange]Train: {train_accs[expert]}{linesep}[/orange] Val: [red]{val_accs[expert]}"
            for expert in ExpertName)
        return f"Epoch {epoch + 1}/{epochs} {scores} Gate: {avg_gate_scores.tolist()}"

    def forward(self, datum):
        """
        overload for torch.nn.Module::forward
        :param datum:
        :return:
        """
        features = self.backbone(datum)
        gate_scores = self.gate(features)
        outputs = {task: expert(features) for task, expert in self.experts.items()}
        return outputs, gate_scores
