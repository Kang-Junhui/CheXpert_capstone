import pandas as pd
from transformers import pipeline
from PIL import Image
import os

# Load CheXpert validation dataset
df = pd.read_csv('./CheXpert-v1.0/valid.csv')

# Filter to only use frontal images
df_frontal = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

print(f"Total images: {len(df)}")
print(f"Frontal images: {len(df_frontal)}")

# Prepare data for batch processing
num_samples = 202  # You can adjust this
df_subset = df_frontal.head(num_samples)

# Define structured prompt with template and terminology
# CheXpert uses 3-class labels: Positive, Negative, Uncertain
prompt = """You are an expert radiologist analyzing chest X-ray images.
You must classify each of the 14 CheXpert findings as exactly one of: Positive, Negative, or Uncertain.

Classification Guidelines:
- Positive: The finding is clearly present
- Negative: The finding is clearly absent
- Uncertain: The finding is unclear, equivocal, or cannot be determined

Evaluate these 14 findings (respond in EXACT format below):

1. No Finding: [Positive/Negative/Uncertain]
2. Enlarged Cardiomediastinum: [Positive/Negative/Uncertain]
3. Cardiomegaly: [Positive/Negative/Uncertain]
4. Lung Opacity: [Positive/Negative/Uncertain]
5. Lung Lesion: [Positive/Negative/Uncertain]
6. Edema: [Positive/Negative/Uncertain]
7. Consolidation: [Positive/Negative/Uncertain]
8. Pneumonia: [Positive/Negative/Uncertain]
9. Atelectasis: [Positive/Negative/Uncertain]
10. Pneumothorax: [Positive/Negative/Uncertain]
11. Pleural Effusion: [Positive/Negative/Uncertain]
12. Pleural Other: [Positive/Negative/Uncertain]
13. Fracture: [Positive/Negative/Uncertain]
14. Support Devices: [Positive/Negative/Uncertain]

IMPORTANT: Respond ONLY with the format above. For each finding, use EXACTLY one of these three words: Positive, Negative, or Uncertain.

Example Output:
1. No Finding: Negative
2. Enlarged Cardiomediastinum: Negative
3. Cardiomegaly: Positive
4. Lung Opacity: Uncertain
...(continue for all 14)

Now analyze this chest X-ray:"""

# Filter only existing images and prepare data
data_list = []

for idx, row in df_subset.iterrows():
    image_path = row['Path']
    if os.path.exists(image_path):
        data_list.append({
            'image': Image.open(image_path),
            'image_path': image_path,
            'sex': row['Sex'],
            'age': row['Age'],
            'view': row['Frontal/Lateral'],
            'ap_pa': row['AP/PA'],
            'prompt': prompt
        })
    else:
        print(f"Image not found: {image_path}")

print(f"\nProcessing {len(data_list)} images with batch processing...")

# Initialize the pipeline with batch processing
pipe = pipeline(
    "image-text-to-text", 
    model="google/medgemma-4b-it",
    batch_size=8  # Adjust based on your GPU memory
)

# Prepare messages for batch processing
def prepare_messages(example):
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": example['image']},
            {"type": "text", "text": example['prompt']}
        ]
    }

# Process all images in batches
results = []
batch_inputs = [[prepare_messages(data)] for data in data_list]

print("Running inference with batch processing...")
outputs = pipe(batch_inputs, batch_size=8)

# Combine results with metadata
for i, (data, output) in enumerate(zip(data_list, outputs)):
    results.append({
        'image_path': data['image_path'],
        'patient_info': f"{data['sex']}, {data['age']} years old",
        'view': data['view'],
        'ap_pa': data['ap_pa'],
        'prediction': output
    })
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(data_list)} images")

print(f"\nCompleted processing {len(results)} images")