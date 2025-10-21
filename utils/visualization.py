import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix and save as an image"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"visuals/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_weights(model, title="Weight Visualization"):
    """Visualize weight matrix of a trained linear classifier"""
    if hasattr(model, 'linear'):
        weights = model.linear.weight.data.cpu().numpy()

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(10):
            ax = axes[i // 5, i % 5]
            w = weights[i].reshape(28, 28)
            ax.imshow(w, cmap='RdBu', vmin=-w.max(), vmax=w.max())
            ax.set_title(f'Digit {i}')
            ax.axis('off')
        plt.suptitle(title)
        plt.savefig('visuals/weight_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_loss_curves(history, title="Training Loss Curves"):
    """Plot loss curves for different models"""
    plt.figure(figsize=(12, 8))
    
    for model_name, losses in history.items():
        plt.plot(losses, label=f'{model_name} Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visuals/loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_comparison(results, title="Model Accuracy Comparison"):
    """Plot bar chart comparing accuracy of different models"""
    plt.figure(figsize=(12, 8))
    
    models = list(results.keys())
    accuracies = list(results.values())
    
    # Create bar plot
    bars = plt.bar(range(len(models)), accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 90% accuracy for reference
    plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visuals/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_curves(history, title="Training Accuracy Curves"):
    """Plot accuracy curves for different models during training"""
    plt.figure(figsize=(12, 8))
    
    for model_name, accuracies in history.items():
        plt.plot(accuracies, label=f'{model_name} Accuracy', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.savefig('visuals/accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_metrics(loss_history, accuracy_history, title="Training Progress"):
    """Plot both loss and accuracy curves in subplots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss curves
    for model_name, losses in loss_history.items():
        ax1.plot(losses, label=f'{model_name}', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    for model_name, accuracies in accuracy_history.items():
        ax2.plot(accuracies, label=f'{model_name}', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('visuals/combined_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_linear_weights_detailed(model, title="Linear Classifier Weight Visualization"):
    """Visualize weight matrix of linear classifier as digit-like images"""
    if hasattr(model, 'linear'):
        weights = model.linear.weight.data.cpu().numpy()
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i in range(10):
            ax = axes[i // 5, i % 5]
            w = weights[i].reshape(28, 28)
            
            # Use a diverging colormap to show positive/negative weights
            im = ax.imshow(w, cmap='RdBu_r', vmin=-np.abs(w).max(), vmax=np.abs(w).max())
            ax.set_title(f'Digit {i} Weights', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar for each subplot
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('visuals/linear_weights_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()


def visualize_naive_bayes_probabilities(nb_model, title="Naive Bayes Probability Maps"):
    """Visualize Naive Bayes learned probabilities as digit-like images"""
    if hasattr(nb_model, 'class_priors_') and hasattr(nb_model, 'feature_log_prob_'):
        # Get the feature probabilities (log probabilities)
        log_probs = nb_model.feature_log_prob_  # Shape: (n_classes, n_features)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i in range(10):
            ax = axes[i // 5, i % 5]
            # Convert log probabilities back to probabilities for visualization
            probs = np.exp(log_probs[i]).reshape(28, 28)
            
            im = ax.imshow(probs, cmap='viridis', aspect='equal')
            ax.set_title(f'Digit {i} Probability Map', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('visuals/naive_bayes_probabilities.png', dpi=300, bbox_inches='tight')
        plt.close()


def visualize_cnn_filters(cnn_model, title="CNN First Layer Filters"):
    """Visualize CNN filters from the first convolutional layer"""
    if hasattr(cnn_model, 'conv_layers'):
        # Get the first conv layer weights
        first_conv = None
        for layer in cnn_model.conv_layers:
            if isinstance(layer, torch.nn.Conv2d):
                first_conv = layer
                break
        
        if first_conv is not None:
            filters = first_conv.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, H, W)
            
            # Show first 16 filters
            n_filters = min(16, filters.shape[0])
            rows = 4
            cols = 4
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            for i in range(n_filters):
                ax = axes[i // cols, i % cols]
                # Take the first channel (grayscale input)
                filter_img = filters[i, 0, :, :]
                
                im = ax.imshow(filter_img, cmap='RdBu_r', aspect='equal')
                ax.set_title(f'Filter {i+1}', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(n_filters, rows * cols):
                axes[i // cols, i % cols].axis('off')
            
            plt.tight_layout()
            plt.savefig('visuals/cnn_filters.png', dpi=300, bbox_inches='tight')
            plt.close()


def visualize_mlp_weights(mlp_model, title="MLP First Layer Weights"):
    """Visualize MLP first layer weights as digit-like images"""
    if hasattr(mlp_model, 'network'):
        # Find the first linear layer
        first_linear = None
        for layer in mlp_model.network:
            if isinstance(layer, torch.nn.Linear):
                first_linear = layer
                break
        
        if first_linear is not None:
            weights = first_linear.weight.data.cpu().numpy()  # Shape: (out_features, in_features)
            
            # Show first 20 neurons
            n_neurons = min(20, weights.shape[0])
            rows = 4
            cols = 5
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            for i in range(n_neurons):
                ax = axes[i // cols, i % cols]
                w = weights[i].reshape(28, 28)
                
                im = ax.imshow(w, cmap='RdBu_r', vmin=-np.abs(w).max(), vmax=np.abs(w).max())
                ax.set_title(f'Neuron {i+1}', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(n_neurons, rows * cols):
                axes[i // cols, i % cols].axis('off')
            
            plt.tight_layout()
            plt.savefig('visuals/mlp_weights.png', dpi=300, bbox_inches='tight')
            plt.close()


def visualize_model_comparison_grid(models_dict, title="Model Architecture Comparison"):
    """Create a grid showing different model visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # This would be called after all models are trained
    # For now, just create placeholder
    axes[0, 0].text(0.5, 0.5, 'Linear\nClassifier\nWeights', ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    axes[0, 1].text(0.5, 0.5, 'MLP\nFirst Layer\nWeights', ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    axes[0, 2].text(0.5, 0.5, 'CNN\nFilters', ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    axes[1, 0].text(0.5, 0.5, 'Naive Bayes\nProbability\nMaps', ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    axes[1, 1].text(0.5, 0.5, 'KNN\nDecision\nBoundaries', ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    axes[1, 2].text(0.5, 0.5, 'Model\nPerformance\nSummary', ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('visuals/model_comparison_grid.png', dpi=300, bbox_inches='tight')
    plt.close()


