import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate(Y_test, Y_pred, model, feature_names, plot_cm= False, verbose=True):
    # Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average="macro")

    # Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)

    # Top positive and negative features
    coefficients = model.coef_[0] 

    df_features = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    top_negative_features = df_features.sort_values(by='Coefficient', ascending=True).head(5)
    top_positive_features = df_features.sort_values(by='Coefficient', ascending=False).head(5)

    # Display evaluation results
    if verbose:
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1: ", f1)

        print("\nConfusion Matrix:\n", cm)

        if plot_cm:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
            disp.plot(cmap=plt.cm.Blues)
            plt.show()

        print("\nTop 5 Negative Features:")
        print(top_negative_features)

        print("\nTop 5 Positive Features:")
        print(top_positive_features)

    return accuracy, precision, recall, f1, cm, top_negative_features, top_positive_features