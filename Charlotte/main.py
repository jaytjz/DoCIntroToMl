from validation import KFoldValidator

if __name__ == "__main__":
    validator = KFoldValidator('../wifi_db/clean_dataset.txt', k=10)
    results = validator.validate()

    print("Clean Data Results:\n")
    print(f"Average Accuracy: {results['accuracy']:.4f}\n")
    print(f"Precision per Class:\n - Class 1:{results['precision_per_class'][0]:.4f}\n - Class 2:{results['precision_per_class'][1]:.4f}\n - Class 3:{results['precision_per_class'][2]:.4f}\n - Class 4:{results['precision_per_class'][3]:.4f}\n")
    print(f"Recall per Class:\n - Class 1:{results['recall_per_class'][0]:.4f}\n - Class 2:{results['recall_per_class'][1]:.4f}\n - Class 3:{results['recall_per_class'][2]:.4f}\n - Class 4:{results['recall_per_class'][3]:.4f}\n")
    print(f"F1 Score per Class:\n - Class 1:{results['f1_score_per_class'][0]:.4f}\n - Class 2:{results['f1_score_per_class'][1]:.4f}\n - Class 3:{results['f1_score_per_class'][2]:.4f}\n - Class 4:{results['f1_score_per_class'][3]:.4f}\n")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    print(f"\nAverage Confusion Matrix over folds:\n{results['average_confusion_matrix']}")