# ============================================================================
# CheXpert Classification Performance Evaluation
# ============================================================================
# Evaluate classification performance on CheXpert 14 pathology labels
# Metrics: Accuracy, Precision, Recall, F1-Score (per-label and aggregate)
# ============================================================================
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, multilabel_confusion_matrix
)
import re

# CheXpert label columns
label_columns = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other', 'Fracture', 'Support Devices'
]

def extract_predictions_from_text(text, label_columns):
    """
    Extract 3-class predictions for each CheXpert label from generated text.
    CheXpert labels: 1.0=Positive, 0.0=Negative, -1.0=Uncertain, NaN=Not mentioned
    Returns: List of values (1.0, 0.0, -1.0, or np.nan) for each label
    """
    predictions = []
    text_lower = text.lower()
    
    for label in label_columns:
        label_lower = label.lower()
        
        # Look for the structured output format: "Label: Positive/Negative/Uncertain"
        # Try multiple patterns
        patterns = [
            f"{label_lower}[:\\s]+(positive|negative|uncertain)",
            f"\\d+\\.\\s*{label_lower}[:\\s]+(positive|negative|uncertain)"
        ]
        
        found = False
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                classification = match.group(1)
                if 'positive' in classification:
                    predictions.append(1.0)
                elif 'negative' in classification:
                    predictions.append(0.0)
                elif 'uncertain' in classification:
                    predictions.append(-1.0)
                found = True
                break
        
        if not found:
            # Fallback: keyword-based detection
            if any(kw in text_lower for kw in ['positive', 'present', 'detected']):
                predictions.append(1.0)
            elif any(kw in text_lower for kw in ['negative', 'absent', 'no']):
                predictions.append(0.0)
            elif any(kw in text_lower for kw in ['uncertain', 'unclear', 'equivocal']):
                predictions.append(-1.0)
            else:
                predictions.append(np.nan)  # Not mentioned
    
    return predictions

# ============================================================================
# Extract ground truth and predictions
# ============================================================================

print("="*80)
print("CheXpert Classification Performance Evaluation")
print("="*80)

y_true = []  # Ground truth labels
y_pred = []  # Predicted labels

print("\nExtracting labels and predictions...")
for idx, result in enumerate(results):
    # Get ground truth from CheXpert
    image_path = result['image_path']
    row = df[df['Path'] == image_path].iloc[0]
    
    # Extract ground truth (preserve CheXpert's 3-class labels)
    # 1.0 = Positive, 0.0 = Negative, -1.0 = Uncertain, NaN = Not mentioned
    true_labels = []
    for col in label_columns:
        value = row[col]
        if pd.isna(value):
            true_labels.append(np.nan)
        else:
            true_labels.append(float(value))
    
    # Extract predictions from generated text
    prediction_text = result['prediction'][0]['generated_text'][-1]['content']
    pred_labels = extract_predictions_from_text(prediction_text, label_columns)
    
    # Debug first few samples
    if idx < 3:
        print(f"\nSample {idx + 1}:")
        print(f"Ground Truth: {true_labels}")
        print(f"Predictions:  {pred_labels}")
        print(f"Text excerpt: {prediction_text[:200]}...")
    
    y_true.append(true_labels)
    y_pred.append(pred_labels)

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"\nProcessed {len(y_true)} samples")
print(f"Shape: {y_true.shape}")

# ============================================================================
# Compute metrics per label
# ============================================================================

print("\n" + "="*80)
print("Per-Label Classification Metrics")
print("="*80)

per_label_results = []

for i, label in enumerate(label_columns):
    y_true_label = y_true[:, i]
    y_pred_label = y_pred[:, i]
    
    # Remove NaN values for metric computation
    valid_mask = ~(np.isnan(y_true_label) | np.isnan(y_pred_label))
    y_true_valid = y_true_label[valid_mask]
    y_pred_valid = y_pred_label[valid_mask]
    
    if len(y_true_valid) == 0:
        print(f"\n{label}: No valid samples (all NaN)")
        continue
    
    # For 3-class evaluation, compute accuracy
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    
    # Binary metrics for Positive (1.0) vs Others
    y_true_binary = (y_true_valid == 1.0).astype(int)
    y_pred_binary = (y_pred_valid == 1.0).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Count support for each class
    support_pos = np.sum(y_true_valid == 1.0)
    support_neg = np.sum(y_true_valid == 0.0)
    support_unc = np.sum(y_true_valid == -1.0)
    
    # Confusion matrix for binary (Positive vs Others)
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    
    # 3-class exact match
    exact_match = np.mean(y_true_valid == y_pred_valid)
    
    per_label_results.append({
        'label': label,
        'accuracy': accuracy,
        'exact_match': exact_match,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support_pos': int(support_pos),
        'support_neg': int(support_neg),
        'support_unc': int(support_unc),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    })
    
    print(f"\n{label}:")
    print(f"  3-Class Accuracy: {accuracy:.4f}")
    print(f"  Exact Match:      {exact_match:.4f}")
    print(f"  Precision (Pos):  {precision:.4f}")
    print(f"  Recall (Pos):     {recall:.4f}")
    print(f"  F1-Score (Pos):   {f1:.4f}")
    print(f"  Support: Pos={support_pos}, Neg={support_neg}, Unc={support_unc}")
    print(f"  Binary (Pos): TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# ============================================================================
# Aggregate metrics
# ============================================================================

print("\n" + "="*80)
print("Aggregate Metrics")
print("="*80)

# Overall accuracy (exact match - all labels must be correct, ignoring NaN)
sample_matches = []
for i in range(len(y_true)):
    valid_mask = ~(np.isnan(y_true[i]) | np.isnan(y_pred[i]))
    if np.sum(valid_mask) > 0:
        sample_matches.append(np.all(y_true[i][valid_mask] == y_pred[i][valid_mask]))
exact_match_accuracy = np.mean(sample_matches) if sample_matches else 0.0
print(f"\nExact Match Accuracy: {exact_match_accuracy:.4f}")
print("(Percentage of samples where ALL labels are predicted correctly)")

# Sample-wise accuracy (average per-sample accuracy, ignoring NaN)
sample_accuracies = []
for i in range(len(y_true)):
    valid_mask = ~(np.isnan(y_true[i]) | np.isnan(y_pred[i]))
    if np.sum(valid_mask) > 0:
        sample_acc = np.mean(y_true[i][valid_mask] == y_pred[i][valid_mask])
        sample_accuracies.append(sample_acc)
sample_wise_accuracy = np.mean(sample_accuracies) if sample_accuracies else 0.0
print(f"\nSample-wise Accuracy: {sample_wise_accuracy:.4f}")
print("(Average 3-class accuracy across all labels per sample)")

# Micro-average (aggregate all predictions, excluding NaN)
valid_mask_flat = ~(np.isnan(y_true.flatten()) | np.isnan(y_pred.flatten()))
y_true_flat_valid = y_true.flatten()[valid_mask_flat]
y_pred_flat_valid = y_pred.flatten()[valid_mask_flat]

# Binary classification: Positive vs Others
y_true_binary_flat = (y_true_flat_valid == 1.0).astype(int)
y_pred_binary_flat = (y_pred_flat_valid == 1.0).astype(int)

precision_micro = precision_score(y_true_binary_flat, y_pred_binary_flat, zero_division=0)
recall_micro = recall_score(y_true_binary_flat, y_pred_binary_flat, zero_division=0)
f1_micro = f1_score(y_true_binary_flat, y_pred_binary_flat, zero_division=0)

print(f"\n--- Micro-Average (Global) - Positive Detection ---")
print(f"Precision: {precision_micro:.4f}")
print(f"Recall:    {recall_micro:.4f}")
print(f"F1-Score:  {f1_micro:.4f}")
print("(Binary: Positive vs Others, all predictions aggregated)")

# Macro-average (average per-label metrics)
precision_macro = np.mean([r['precision'] for r in per_label_results])
recall_macro = np.mean([r['recall'] for r in per_label_results])
f1_macro = np.mean([r['f1'] for r in per_label_results])

print(f"\n--- Macro-Average (Per-Label) ---")
print(f"Precision: {precision_macro:.4f}")
print(f"Recall:    {recall_macro:.4f}")
print(f"F1-Score:  {f1_macro:.4f}")
print("(Equal weight to each label)")

# Weighted average (by positive support)
total_support = sum(r['support_pos'] for r in per_label_results)
precision_weighted = sum(r['precision'] * r['support_pos'] for r in per_label_results) / total_support if total_support > 0 else 0
recall_weighted = sum(r['recall'] * r['support_pos'] for r in per_label_results) / total_support if total_support > 0 else 0
f1_weighted = sum(r['f1'] * r['support_pos'] for r in per_label_results) / total_support if total_support > 0 else 0

print(f"\n--- Weighted Average (By Support) ---")
print(f"Precision: {precision_weighted:.4f}")
print(f"Recall:    {recall_weighted:.4f}")
print(f"F1-Score:  {f1_weighted:.4f}")
print("(Weighted by number of positive samples per label)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Total Samples: {len(y_true)}")
print(f"Total Labels: {len(label_columns)}")
print(f"Total Predictions: {y_true.size}")
print(f"\nBest Performing Labels (by F1):")
sorted_labels = sorted(per_label_results, key=lambda x: x['f1'], reverse=True)
for i, result in enumerate(sorted_labels[:5], 1):
    print(f"  {i}. {result['label']}: F1={result['f1']:.4f}")

print(f"\nWorst Performing Labels (by F1):")
for i, result in enumerate(sorted_labels[-5:][::-1], 1):
    print(f"  {i}. {result['label']}: F1={result['f1']:.4f}")

print("="*80)
