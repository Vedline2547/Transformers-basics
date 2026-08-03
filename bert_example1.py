from transformers import BertTokenizer,BertModel
# Creating an environment in PyTorch
# Load a pretrained BERT Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
# Tokenize a simple input
text = "Transformers are powerful models for NLP Tasks"
inputs = tokenizer(text,return_tensors="pt")
# Pass input through model
outputs=model(**inputs)
print("Hidden States Shape:",outputs.last_hidden_state.shape)
