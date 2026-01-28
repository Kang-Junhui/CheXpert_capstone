import re
from sacrebleu import corpus_bleu

# Define label mapping for CheXpert
CHEXPERT_LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

def label_to_text(value):
    """Convert CheXpert label to text"""
    if pd.isna(value) or value == 0.0:
        return "Negative"
    elif value == 1.0:
        return "Positive"
    elif value == -1.0:
        return "Uncertain"
    else:
        return "Negative"

def create_reference_text(row, labels):
    """Create multiple reference texts from ground truth labels for better BLEU score"""
    # Create 3 different styles of reference texts
    references_list = []
    
    # Style 1: Simple declarative sentences
    findings1 = []
    for label in labels:
        label_value = row.get(label, None)
        classification = label_to_text(label_value)
        
        if label == 'No Finding':
            if classification == 'Positive':
                findings1.append("The chest X-ray shows no acute cardiopulmonary abnormality.")
            else:
                findings1.append("Abnormalities are present on the chest X-ray.")
        elif classification == 'Positive':
            findings1.append(f"{label} is present.")
        elif classification == 'Negative':
            findings1.append(f"No {label.lower()} is identified.")
        else:
            findings1.append(f"{label} is uncertain or questionable.")
    references_list.append(" ".join(findings1))
    
    # Style 2: More natural medical report style
    findings2 = []
    for label in labels:
        label_value = row.get(label, None)
        classification = label_to_text(label_value)
        
        if label == 'No Finding':
            if classification == 'Positive':
                findings2.append("No acute abnormality is seen.")
            else:
                findings2.append("Acute abnormalities are noted.")
        elif classification == 'Positive':
            findings2.append(f"There is evidence of {label.lower()}.")
        elif classification == 'Negative':
            findings2.append(f"{label} is not seen.")
        else:
            findings2.append(f"{label} cannot be definitively excluded.")
    references_list.append(" ".join(findings2))
    
    # Style 3: Varied expressions
    findings3 = []
    for label in labels:
        label_value = row.get(label, None)
        classification = label_to_text(label_value)
        
        if label == 'No Finding':
            if classification == 'Positive':
                findings3.append("Normal chest radiograph.")
            else:
                findings3.append("Abnormal findings present.")
        elif classification == 'Positive':
            findings3.append(f"{label} detected.")
        elif classification == 'Negative':
            findings3.append(f"No evidence of {label.lower()}.")
        else:
            findings3.append(f"Possible {label.lower()}.")
    references_list.append(" ".join(findings3))
    
    return references_list

def extract_prediction_text(prediction_output):
    """Extract structured text from model prediction"""
    try:
        # Get the actual text content from the pipeline output
        if isinstance(prediction_output, list) and len(prediction_output) > 0:
            if isinstance(prediction_output[0], dict):
                content = prediction_output[0].get('generated_text', [])
                if isinstance(content, list) and len(content) > 0:
                    # Get the last message content (assistant's response)
                    return content[-1].get('content', '')
        return str(prediction_output)
    except Exception as e:
        print(f"Error extracting prediction: {e}")
        return str(prediction_output)

# Prepare references and predictions
# BLEU supports multiple references - create varied reference texts for better matching
all_references = []  # List of lists: each prediction has multiple reference variants
predictions = []

for i, result in enumerate(results):
    # Get the corresponding row from the dataframe
    image_path = result['image_path']
    matching_row = df_subset[df_subset['Path'] == image_path]
    
    if not matching_row.empty:
        row = matching_row.iloc[0]
        
        # Create multiple reference texts (returns a list of 3 variants)
        ref_texts = create_reference_text(row, CHEXPERT_LABELS)
        all_references.append(ref_texts)
        
        # Extract prediction text
        pred_text = extract_prediction_text(result['prediction'])
        predictions.append(pred_text)
        
        if i < 3:  # Print first 3 examples for verification
            print(f"\n{'='*80}")
            print(f"Example {i+1}:")
            print(f"\nReference (Style 1):\n{ref_texts[0][:200]}...")
            print(f"\nReference (Style 2):\n{ref_texts[1][:200]}...")
            print(f"\nReference (Style 3):\n{ref_texts[2][:200]}...")
            print(f"\nPrediction:\n{pred_text[:200]}...")
            print(f"{'='*80}")

print(f"\nTotal pairs for evaluation: {len(predictions)}")
print(f"Number of reference variants per prediction: {len(all_references[0]) if all_references else 0}")

# Calculate BLEU Score with multiple references
# Transpose references: from [[ref1_v1, ref1_v2, ref1_v3], [ref2_v1, ...]]
#                       to [[ref1_v1, ref2_v1, ...], [ref1_v2, ref2_v2, ...], ...]
transposed_refs = list(zip(*all_references))
bleu = corpus_bleu(predictions, transposed_refs)
print(f"\n{'='*80}")
print(f"BLEU Score (with {len(transposed_refs)} reference variants): {bleu.score:.2f}")
print(f"{'='*80}")