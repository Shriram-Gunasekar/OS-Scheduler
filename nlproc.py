from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Step 1: Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Binary classification (schedule or not)

# Step 2: Define scheduling instructions and labels
scheduling_instructions = [
    "Schedule task A for tomorrow at 10 AM.",
    "Do not schedule task B on weekends.",
    "Schedule task C as soon as possible.",
    "Ensure task D is completed before Friday.",
    "Schedule task E every Monday and Wednesday at 3 PM."
]
labels = [1, 0, 1, 1, 1]  # 1 for schedule, 0 for not schedule

# Step 3: Tokenize and prepare the input data
encoded_inputs = tokenizer(scheduling_instructions, padding=True, truncation=True, return_tensors='pt')

# Step 4: Fine-tune the BERT model for scheduling classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Convert labels to tensor
labels_tensor = torch.tensor(labels)

# Fine-tune the model
num_epochs = 3
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(**encoded_inputs, labels=labels_tensor)
    loss = loss_fn(outputs.logits, labels_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Step 5: Use the fine-tuned model for scheduling predictions
def predict_schedule(model, tokenizer, scheduling_instruction):
    encoded_input = tokenizer(scheduling_instruction, padding=True, truncation=True, return_tensors='pt')
    output = model(**encoded_input)
    predicted_label = torch.argmax(output.logits, dim=1).item()
    return predicted_label

# Example usage
new_instruction = "Schedule task F for next Thursday at 2 PM."
predicted_label = predict_schedule(model, tokenizer, new_instruction)
if predicted_label == 1:
    print("Schedule this task.")
else:
    print("Do not schedule this task.")
