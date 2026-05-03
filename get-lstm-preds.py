import torch
import importlib
import os
from lstm import MyModel

data_processor = importlib.import_module('lstm-data-processor')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load trained MyModel
print("Loading trained MyModel...")
model = MyModel(bidirectional=True).to(device)

# Load the DDP-saved module weights cleanly
model_path = './data/models/best_model.pt'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# 2. Process Test Files
data_files = [
    './data/test/test01_text_only.csv', 
    './data/test/test02_text_only.csv', 
    './data/test/test03_text_only.csv'
]

for i, path in enumerate(data_files):
    print(f"\nProcessing {path}...")
    
    dataset, tokenizer = data_processor.load_and_preprocess_data(
        data_paths=[path],
        data_type='test',
        model_type='transformer'
    )

    # Use standard dataloader for sequential testing to preserve row_id ordering
    dataloader = data_processor.prepare_standard_dataloader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []

    # Extract features and predict
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            
            # Predict
            logits = model(input_ids, attention_mask=attention_mask)
            preds_prob = torch.sigmoid(logits)
            
            # Convert to binary int predictions
            preds = (preds_prob > 0.5).int()
            all_preds.extend(preds.cpu().tolist())

    # Write results to CSV
    output_dir = './data/test/preds/'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'test0{i+1}-pred.csv')
    
    with open(output_filename, 'w') as f:
        f.write('row_id,prediction\n')
        for row_id, pred in enumerate(all_preds):
            f.write(f'{row_id},{pred}\n')
            
    print(f"Saved {len(all_preds)} predictions to {output_filename}")