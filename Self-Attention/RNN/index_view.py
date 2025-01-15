import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


submission_df = pd.read_csv(r'data/sampleSubmission.csv')
prediction_df = pd.read_csv(r'data/SA_predict.csv')
true_labels = submission_df['Sentiment'].values
predicted_labels = prediction_df['Sentiment'].values
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1_score: {f1:.4f}")
