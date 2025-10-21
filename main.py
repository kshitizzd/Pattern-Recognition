import numpy as np

from data_loader import MNISTLoader
from models.knn import KNNClassifier
from models.naive_bayes import NaiveBayesClassifier
from models.linear import train_linear_classifier
from models.mlp import train_mlp
from models.cnn import train_cnn
from utils.visualization import (plot_confusion_matrix, visualize_weights, 
                                plot_loss_curves, plot_accuracy_comparison, 
                                plot_accuracy_curves, plot_combined_metrics,
                                visualize_linear_weights_detailed, visualize_naive_bayes_probabilities,
                                visualize_cnn_filters, visualize_mlp_weights, visualize_model_comparison_grid)


def main():
    # Load data
    print("Loading MNIST dataset")
    data_loader = MNISTLoader(root_dir='MNIST')
    data_loader.load_data()
    # 80/20 split for training and test
    X_train, y_train, X_test, y_test = data_loader.split_data()
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    results = {}
    
    # KNN and Naive Bayes don't have iterative training
    loss_history = {}
    accuracy_history = {}

    #  KNN
    print("KNN")
    for k in [1, 3, 5]:
        print(f"\nTesting k={k}")
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        acc, pred = knn.evaluate(X_test[:1000], y_test[:1000])
        results[f'KNN (k={k})'] = acc
        print(f"KNN (k={k}) Accuracy: {acc:.4f}")

    # NAIVE BAYES
    print("NAIVE BAYES")
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    acc, pred = nb.evaluate(X_test, y_test)
    results['Naive Bayes'] = acc
    print(f"Naive Bayes Accuracy: {acc:.4f}")

    # LINEAR CLASSIFIER
    print("LINEAR CLASSIFIER")
    linear_model, acc, linear_losses, linear_accuracies = train_linear_classifier(X_train, y_train, X_test, y_test)
    results['Linear Classifier'] = acc
    loss_history['Linear Classifier'] = linear_losses
    accuracy_history['Linear Classifier'] = linear_accuracies

    print("MULTILAYER PERCEPTRON")
    mlp_model, acc, mlp_losses, mlp_accuracies = train_mlp(X_train, y_train, X_test, y_test)
    results['MLP'] = acc
    loss_history['MLP'] = mlp_losses
    accuracy_history['MLP'] = mlp_accuracies

    # CNN
    print("CNN")
    cnn_model, acc, cnn_losses, cnn_accuracies = train_cnn(X_train, y_train, X_test, y_test)
    results['CNN'] = acc
    loss_history['CNN'] = cnn_losses
    accuracy_history['CNN'] = cnn_accuracies

    # Print final results
    print("\n" + "="*10)
    print("FINAL RESULTS")
    print("="*10)
    for method, accuracy in results.items():
        print(f"{method}: {accuracy:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot accuracy comparison (all models)
    plot_accuracy_comparison(results, "MNIST Classification - Model Accuracy Comparison")
    # Plot loss curves (neural networks only - KNN and Naive Bayes don't have iterative training)
    plot_loss_curves(loss_history, "MNIST Classification - Training Loss Curves")
    # Plot accuracy curves (neural networks only)
    plot_accuracy_curves(accuracy_history, "MNIST Classification - Training Accuracy Curves")
    # Plot combined metrics (neural networks only)
    plot_combined_metrics(loss_history, accuracy_history, "MNIST Classification - Training Progress")
    # Visualize linear classifier weights (basic)
    visualize_weights(linear_model, "Linear Classifier Weight Visualization")    
    # Detailed linear classifier weights
    visualize_linear_weights_detailed(linear_model, "Linear Classifier - Detailed Weight Analysis")
    # Naive Bayes probability maps
    visualize_naive_bayes_probabilities(nb, "Naive Bayes - Learned Probability Maps")
    # CNN filters
    visualize_cnn_filters(cnn_model, "CNN - First Layer Filter Visualization")    
    # MLP weights
    visualize_mlp_weights(mlp_model, "MLP - First Layer Weight Visualization")    
    # Model comparison grid
    models_dict = {
        'Linear': linear_model,
        'MLP': mlp_model,
        'CNN': cnn_model,
        'Naive Bayes': nb
    }
    visualize_model_comparison_grid(models_dict, "Model Architecture Comparison")
    print("✓ Model comparison grid saved as 'visuals/model_comparison_grid.png'")
    
    print("\nAll visualizations have been generated and saved!")


if __name__ == "__main__":
    main()


