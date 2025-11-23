import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc # We use scikit-learn because it has reliable classification metrics. 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate(model, dataloader, show_plots = True, threshold=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval() # Set the model to evaluation mode. This disables dropout and batch norm learning so predictions are stable.
    
    preds, probs, labels = [], [], []

    with torch.no_grad(): #Disable gradient calculation: Evaluation should not change model weights and should use less memory. 
        for x, y in dataloader: # Loop through test dataloader: We get input signals and true labels batch by batch.
            x, y = x.to(device), y.to(device)

            logits = model(x)
            prob_1 = torch.softmax(logits, dim=1)[:, 1]

            predicted = (prob_1 >= threshold).long()
            

            # #Make predictions
            # logits = model(x)
            # # Class predictions (0 / 1)
            # predicted = torch.argmax(logits, dim=1)

            # # Probabilities for positive class (FOG = class 1)
            # prob_1 = torch.softmax(logits, dim=1)[:, 1]

            # store results: We collect results into two Python lists so we can compute metrics.
            preds.extend(predicted.cpu().numpy())
            probs.extend(prob_1.cpu().numpy())
            labels.extend(y.cpu().numpy())

    # Convert to numpy arrays
    preds = np.array(preds)
    probs = np.array(probs)
    labels = np.array(labels)

    # Metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    # ROC-AUC
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["No FOG", "FOG"], digits=4))

    # For FoG detection, F1 is more important than accuracy, 
    # because the model could get high accuracy by always predicting “no FoG”.

     # Visualization
    if show_plots:
        # Confusion Matrix
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["No FOG", "FOG"],
                    yticklabels=["No FOG", "FOG"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.close()

        # ROC Curve
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        plt.plot([0,1], [0,1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ROC_curve.png", dpi=150)
        plt.close()

    return acc, f1, auc_score, cm