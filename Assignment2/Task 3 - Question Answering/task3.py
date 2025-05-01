import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

seed = 10
train_size = 18000
val_size = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)
dataset = load_dataset("squad_v2")
train_data = dataset["train"].shuffle(seed = seed).select(range(train_size))
val_data = dataset["validation"].shuffle(seed = seed).select(range(val_size))

tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")

def preprocess(data):
    inputs = tokenizer(
        data["question"],
        data["context"],
        truncation = True,
        padding = "max_length",
        max_length = 512,
        stride = 128,
        return_overflowing_tokens = True,
        return_offsets_mapping = True
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    start_positions, end_positions = [], []
    
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = data["answers"][sample_idx]
        
        if len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            
            token_start_index = next((idx for idx, (start, end) in enumerate(offsets) if start_char >= start and start_char < end), 0)
            token_end_index = next((idx for idx, (start, end) in enumerate(offsets) if end_char > start and end_char <= end), 0)
            
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["overflow_to_sample_mapping"] = sample_map
    
    return inputs


tokenized_train_data = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names)
tokenized_val_data = val_data.map(preprocess, batched=True, remove_columns=val_data.column_names)

tokenized_train_data.set_format(type="torch", device=device)
tokenized_val_data.set_format(type="torch", device=device)
model = AutoModelForQuestionAnswering.from_pretrained("SpanBERT/spanbert-large-cased").to(device)

training_args = TrainingArguments(
    output_dir = "/kaggle/working/spanbert_model",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    learning_rate = 3e-5,
    num_train_epochs = 8,
    weight_decay = 0.01,
    logging_dir = "/kaggle/working/logs",
    logging_steps = 500,
    save_total_limit = 2,
    report_to = "none",
    fp16 = True
)


def extract_answer(start_logits, end_logits, input_ids):
    """Extracts the best answer span based on softmax probability."""
    start_probs = torch.softmax(torch.tensor(start_logits), dim=-1).numpy()
    end_probs = torch.softmax(torch.tensor(end_logits), dim=-1).numpy()

    best_start, best_end = 0, 0
    max_prob = 0

    for i in range(len(start_probs)):
        for j in range(i, min(i + 30, len(end_probs))):
            prob = start_probs[i] * end_probs[j]
            if prob > max_prob:
                best_start, best_end = i, j
                max_prob = prob

    predicted_answer = tokenizer.decode(input_ids[best_start:best_end + 1], skip_special_tokens=True)
    return predicted_answer

def exact_match_score(predictions, references):
    """Computes the exact match (EM) score between predictions and references."""
    assert len(predictions) == len(references), "Lists must have the same length"
    matches = sum(p == r for p, r in zip(predictions, references))
    return matches / len(references) * 100  # Convert to percentage

def compute_metrics(eval_pred):
    """Computes exact match (EM) for evaluation."""
    start_logits, end_logits = eval_pred.predictions

    sample_map = tokenized_val_data["overflow_to_sample_mapping"]
    
    predictions = []
    references = []

    for i, (start, end, input_ids) in enumerate(zip(start_logits, end_logits, tokenized_val_data["input_ids"])):
        sample_idx = sample_map[i]
        
        if sample_idx >= len(val_data["answers"]):
            continue
        
        pred_text = extract_answer(start, end, input_ids)
        ref_text = val_data["answers"][sample_idx]["text"][0] if val_data["answers"][sample_idx]["text"] else ""

        predictions.append(pred_text)
        references.append(ref_text)

    em_score = exact_match_score(predictions, references)
    return {"exact_match": em_score}

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train_data,
    eval_dataset = tokenized_val_data,
    processing_class = tokenizer,
    compute_metrics = compute_metrics
)
trainer.train()
with torch.no_grad():
    eval_results = trainer.evaluate()


print(f"Evaluation Results: {eval_results}")
logs = trainer.state.log_history

train_loss = [entry["loss"] for entry in logs if "loss" in entry]
eval_loss = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]

plt.figure(figsize=(8, 6))
plt.plot(train_loss, label="Training Loss", marker="o")
plt.plot(eval_loss, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
model.save_pretrained("/kaggle/working")
tokenizer.save_pretrained("/kaggle/working")




# For SpanBERT CRF


!pip install pytorch-crf
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, AdamW
from datasets import load_dataset
from torchcrf import CRF
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed = 10
train_size = 18000
val_size = 5000

model_identifier = "SpanBERT/spanbert-base-cased"
dataset = load_dataset("squad_v2")
train_data = dataset["train"].shuffle(seed = seed).select(range(train_size))
val_data = dataset["validation"].shuffle(seed = seed).select(range(val_size))
tokenizer = AutoTokenizer.from_pretrained(model_identifier)
def preprocess(data):
    encoding = tokenizer(
        data['context'],
        data['question'],
        truncation = True,
        padding = True,
        max_length = 512,
        return_tensors = 'pt'
    )

    start_idx = data['answers']['answer_start'][0] if data['answers']['text'] else -1
    end_idx = start_idx + len(data['answers']['text'][0]) if data['answers']['text'] else -1

    targets = torch.zeros(encoding['input_ids'].shape, dtype = torch.long)
    if start_idx != -1 and end_idx != -1:
        targets[0, start_idx:end_idx] = 1

    return {
        'input_ids' : encoding['input_ids'].squeeze(0),
        'attention_mask' : encoding['attention_mask'].squeeze(0),
        'targets' : targets.squeeze(0)
    }

def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    targets = [torch.tensor(item['targets']) for item in batch]
    
    max_len = max(x.shape[0] for x in input_ids)
    
    input_ids = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], dtype=torch.long)]) for x in input_ids])
    attention_mask = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], dtype=torch.long)]) for x in attention_mask])
    targets = torch.stack([torch.cat([x, torch.full((max_len - x.shape[0],), -1, dtype=torch.long)]) for x in targets])
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'targets': targets}


train_data = train_data.map(preprocess, remove_columns = train_data.column_names)
val_data = val_data.map(preprocess, remove_columns = val_data.column_names)

train_dataloader = DataLoader(train_data, batch_size = 8, shuffle = True, collate_fn = collate_fn)
val_dataloader = DataLoader(val_data, batch_size = 8, shuffle = False, collate_fn = collate_fn)
class SpanBERT_CRF(nn.Module):
    def __init__(self, model_identifier, num_classes):
        super(SpanBERT_CRF, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_identifier)
        self.feature_dim = self.encoder.config.hidden_size
        self.output_layer = nn.Linear(self.feature_dim, num_classes)
        self.crf_layer = CRF(num_classes, batch_first=True)

    def forward(self, input_ids, attention_mask, targets=None):
        encoded_output = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = encoded_output.last_hidden_state
        emissions = self.output_layer(hidden_states)

        if targets is not None:
            loss_value = -self.crf_layer(emissions, targets, mask=attention_mask.bool(), reduction='mean')
            return {"loss": loss_value, "logits": emissions}
        
        return {"logits": emissions}

def train_model(model, train_dataloader, val_dataloader, device, optimizer, epochs=6):
    model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, targets)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=total_train_loss / len(train_dataloader))
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        avg_val_loss = evaluate_model(model, val_dataloader, device, compute_metrics=False)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, val_dataloader, device, compute_metrics=True, tokenizer = None):
    model.eval()
    total_loss = 0
    all_predictions, all_labels = [] ,[]
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask, targets)
            loss = outputs['loss']
            total_loss += loss.item()
            predictions = outputs['logits'].cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(val_dataloader)
    if compute_metrics:
        eval_results = evaluate_metrics(all_predictions, tokenizer, val_dataloader.dataset)
        print(f"Final Validation Metrics: {eval_results}")
    
    return avg_loss

def exact_match_score(predictions, references):
    assert len(predictions) == len(references), "Lists must have the same length"
    matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return matches / len(references) * 100

def evaluate_metrics(predictions, tokenizer, test_data):
    predicted_labels = [
        (np.argmax(point[:, 0]), np.argmax(point[:, 1]))
        for point in predictions
    ]
    
    true_labels = []
    for sequence in test_data:
        start_idx, end_idx = 0, 0
        
        for idx, label in enumerate(sequence['targets']):
            if label == 1:
                start_idx = idx
                break
        
        for idx in range(len(sequence['targets'])):
            if sequence['targets'][idx] == 1:
                end_idx = idx
        
        true_labels.append((start_idx, end_idx))
    
    decoded_preds = [
        tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                test_data[i]['input_ids'][pred_start:pred_end]
            )
        ).strip()
        for i, (pred_start, pred_end) in enumerate(predicted_labels)
    ]
    
    decoded_labels = [
        tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                test_data[i]['input_ids'][label_start:label_end]
            )
        ).strip()
        for i, (label_start, label_end) in enumerate(true_labels)
    ]
    
    exact_match = exact_match_score(decoded_preds, decoded_labels)
    return {'exact_match': exact_match}

model = SpanBERT_CRF(model_identifier, 2)
optimizer = AdamW(model.parameters(), lr = 5e-5)

train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, device, optimizer)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss")
plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Over Epochs")
plt.show()
evaluate_model(model, val_dataloader, device, compute_metrics=True, tokenizer = tokenizer)

torch.save(model.state_dict(), "spanbert_crf_model.pth")
print("Model Saved Successfully !")