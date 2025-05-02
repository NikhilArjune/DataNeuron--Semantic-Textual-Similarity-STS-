from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Global variable to hold the model
model = None

# Define request body structure
class Texts(BaseModel):
    text1: str
    text2: str

# Load model on startup
@app.on_event("startup")
async def startup_event():
    global model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to split text into chunks using model's tokenizer
def split_text(text, max_length=512):
    tokenizer = model[0].tokenizer
    tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    chunks = []
    for i in range(0, len(tokens), max_length - 2):
        chunk = tokens[i:i + max_length - 2]
        chunk = torch.cat([torch.tensor([tokenizer.cls_token_id]), chunk, torch.tensor([tokenizer.sep_token_id])])
        chunks.append(chunk)
    return chunks

# Function to encode text with chunking
def encode_text(text):
    chunks = split_text(text)
    embeddings = []
    for chunk in chunks:
        chunk_text = model[0].tokenizer.decode(chunk, skip_special_tokens=True)
        emb = model.encode(chunk_text, convert_to_tensor=True)
        embeddings.append(emb)
    if len(embeddings) > 1:
        return torch.mean(torch.stack(embeddings), dim=0)
    else:
        return embeddings[0]

# Define similarity endpoint
@app.post("/similarity")
async def compute_similarity_endpoint(texts: Texts):
    # Encode texts with chunking
    emb1 = encode_text(texts.text1).unsqueeze(0)  # Shape: (1, embedding_dim)
    emb2 = encode_text(texts.text2).unsqueeze(0)  # Shape: (1, embedding_dim)
    
    # Compute cosine similarity along dim=1
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
    
    # Map similarity from [-1, 1] to [0, 1]
    score = (similarity.item() + 1) / 2
    
    return {"similarity score": score}