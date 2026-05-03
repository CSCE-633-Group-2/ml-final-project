import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import csv
from tqdm import tqdm

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# Replace with your actual Hugging Face token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Llama 3 on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=None)
# Load in 16-bit precision to save GPU memory
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    token=None, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
model.eval()

# ==========================================
# 2. BUILD THE FEW-SHOT SYSTEM PROMPT
# ==========================================
# The system prompt sets the strict rules of engagement
system_instructions = """You are an expert clinical coder. Your task is to analyze a short chunk of medical text and determine if it contains ICD-codable information (diagnoses, procedures, or billable medical events).
You must output ONLY the number '1' if it contains codable information, or the number '0' if it does not. Do not provide explanations, notes, or any other text.
"""

def build_few_shot_context(train_csv_path):
    """
    Reads your 20 training examples and formats them as a historical 
    conversation to teach Llama exactly how to classify the text.
    """
    # Initialize the conversation with the system prompt
    messages = [
        {"role": "system", "content": system_instructions}
    ]
    
    print(f"Loading 20 few-shot examples from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    
    # Append the 20 examples as User/Assistant turns
    for _, row in train_df.iterrows():
        text = str(row['text'])
        label = str(int(row['label'])) # Ensure it is strictly "1" or "0"
        
        messages.append({"role": "user", "content": f"Classify this text:\n\n{text}"})
        messages.append({"role": "assistant", "content": label})
        
    return messages

# ==========================================
# 3. INFERENCE LOOP
# ==========================================
def generate_prediction(messages_context, test_text):
    """Appends the new test text to the few-shot context and asks for a prediction."""
    
    # Create a fresh copy of the context so we don't permanently alter the history
    current_messages = messages_context.copy()
    current_messages.append({"role": "user", "content": f"Classify this text:\n\n{test_text}"})
    
    # FIX 1: First, format the messages into a single Llama-3 formatted string
    prompt_str = tokenizer.apply_chat_template(
        current_messages, 
        add_generation_prompt=True, 
        tokenize=False # Don't tokenize yet, just give us the raw string
    )
    
    # FIX 2: Standardize the tokenization so it outputs exactly what model.generate expects
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate( # type: ignore
            **inputs, # Unpack the dictionary safely
            max_new_tokens=2, 
            do_sample=False,  # Strict deterministic classification
            # FIX 3: Removed temperature=0.0 to clear the warning spam
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the newly generated text (ignoring the input prompt)
    response_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    # Fallback parsing
    if "1" in response: return 1
    if "0" in response: return 0
    return 0

# --- Execution ---
if __name__ == "__main__":
    # 1. Build the base conversation history using your 20 gold-standard notes
    # Adjust path as necessary
    base_messages = build_few_shot_context('./data/train/train_data-text_and_labels.csv')
    
    # 2. Iterate through the test files
    test_files = [
        './data/test/test01_text_only.csv', 
        './data/test/test02_text_only.csv', 
        './data/test/test03_text_only.csv'
    ]
    
    output_dir = './data/test/preds/'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, path in enumerate(test_files):
        print(f"\nProcessing Llama predictions for {path}...")
        test_df = pd.read_csv(path)
        
        results = []
        
        # tqdm progress bar for the inference loop
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Test 0{i+1}"):
            test_text = str(row['text'])
            prediction = generate_prediction(base_messages, test_text)
            results.append([index, prediction])
            
        # Write results to CSV
        output_filename = os.path.join(output_dir, f'test0{i+1}-llama-pred.csv')
        with open(output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row_id', 'prediction'])
            writer.writerows(results)
            
        print(f"Saved {len(results)} Llama predictions to {output_filename}")