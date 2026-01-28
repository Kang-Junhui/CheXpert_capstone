import re
from bert_score import score as bert_score

# Calculate BERT Score
print("Calculating BERT Score (this may take a few minutes)...")
# For BERT Score, use the first reference style
references_for_bert = [refs[0] for refs in all_references]
P, R, F1 = bert_score(predictions, references_for_bert, lang='en', verbose=True)

print(f"\n{'='*80}")
print("BERT Score Results:")
print(f"Precision: {P.mean():.4f} (±{P.std():.4f})")
print(f"Recall: {R.mean():.4f} (±{R.std():.4f})")
print(f"F1: {F1.mean():.4f} (±{F1.std():.4f})")
print(f"{'='*80}")