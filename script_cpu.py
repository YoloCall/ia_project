# pip install transformers torch numpy datasets wandb tqdm

# Decoder only transformer : generating text

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np

# hyperparamters
batch_size = 12 # how many independent sequences will process in parallel ?
block_size = 64 # max input lenght for prediction
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
device = 'cpu' # device='cuda' if you want to run with GPU
eval_iters = 20

# Every head = n_embd/n_head dimensions
n_embd = 128 # number of embedding dimensions
n_head = 4 # number of heads that we would like
n_layer = 4 # how many layers of the blocks we are going to have

dropout = 0.0 # network too small 

# ---------------

torch.manual_seed(2023)

# Get Data
with open('input.txt', 'r', encoding='UTF-8') as f:
    text = f.read()
    
# Get all characters and vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Transform characters into integers
stoi = { ch:i for i,ch in enumerate(chars) }
# Transform integers into charaters
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Transform all characters in input to integers
data = torch.tensor(encode(text), dtype=torch.long)

# Split data to train and test
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Split data in 2 tensors input and target
def get_batch(split):
    
    data = train_data if split == 'train' else val_data
    
    # If batch_size = 4, generate 4 random numbers
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # ix : tensor([634674, 832604, 689697, 870724])
    
    # Get values corresponding to ix
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # if block_size = 8 get 2 tensor (8 columns x 4 rows)
    x, y = x.to(device), y.to(device) # Optimization device = 'cuda'
    return x, y

def estimate_loss():
    out = {}
    # Setting to evaluation phase
    model.eval()
    for split in ['train', 'val']:
        # Returns a tensor filled with the <eval_iters> values 0
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        # Get average of the loss for train and test
        out[split] = losses.mean()
    # Reset back to training phase
    model.train()
    return out

# Self-Attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Every tokens will emit 3 vectors : 
            # Query (what i'm looking for) 
            # Key (what do i contain)
            # Value (If you find me interesting here's what i will communicate to you)
        # Single head, with linear modules, bias=False to apply a matrix multiply with some fixed weights
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Get a tensor with lower triangular part
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        
        # https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        # Counter over-fitting with random desactivation of neurons every forward
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        
        # Learn more about the token as opposed to any other token in the sequence
        k = self.key(x) #(B,T,16)
        q = self.query(x) #(B,T,16)
        # Transpose the last 2 dimensions
        # Every single batch elements will have different sort of way, not data dependent
        # Scaled dot-product attention : dividing by 1/square_root(head_size) -> normalized
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,T) T_square matrix given affinities between tokens
        
        # Replace 0 by -inf, token from the past cannot communicate
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        
        # weight for each row
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # X is a private information of the token to communicate with
        v = self.value(x)
        
        # Weight inputs, average of past token and current token
        out = wei @ v
        
        return out

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create multiple-heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # Create a projection
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        # Run all of them in parallel into a list and concat all the ouputs
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating over channel dimension
        
        # apply the projection before the dropout
        out = self.dropout(self.proj(out))
        return out
    
# Position-wise Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # All the token do this independently
        self.net = nn.Sequential(
            # Linear followed by a relative nonlinearity
            # cf https://arxiv.org/pdf/1706.03762v5.pdf : part 3.3 --> dff/dmodel = 2048/512 = 4
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            
            # Projection layer going back into the residual pathway
            nn.Linear(4 * n_embd, n_embd),
            
            # Add before the residual connection back into the original pathway
            nn.Dropout(dropout),
        )
    
    def forward(self,x):
        return self.net(x)

    

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        # Communication between tokens with multi-head self-attention
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,n_embd)
        
        # Computation with feed-forward network on all the tokens idependently
        self.ffwd = FeedForward(n_embd)
        
        # Layer normalization : 
        # Made sure that across the batch dimension any individual neuron had unit gaussian distribution
        # Zero mean one standart deviation : normalizing every single rows
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self,x):
        # Residual connections = x + ...
        # Pre-norm formulation = apply layer norm before multi-head and feed-forward
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Encoding the identity of the token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Encoding the position of the token
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Create blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # Layer norm after the transformer and before the final linear layer
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Create single layer feed forward network : Ax = b where x is input, b is output, A is weight
        # From n_embd to vocab_size
        # Decode into vocabulary
        self.lm_head = nn.Linear(n_embd, vocab_size) #(B,T,vocab_size)
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        # Generate a Tensor with index. 
        # The inputs data will refer to the embedding table and link the row with the idx
        # If the input is a Tensor(8 x 4), (Batch, Time, Channel) = (4, 8, vocab_size)
        # tok_emd = the score of the next characters in the sequence
        tok_emd = self.token_embedding_table(idx) # (B,T,C)
        
        # Embedding the integer from 0 to T-1
        pos_emd = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        
        # Token embeddings with the positional embeddings
        x = tok_emd + pos_emd
        
        # Feed-forward many times with blocks
        x = self.blocks(x)
        # Decode
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        # targets is (B,T)
        if targets is None:
            loss = None
        else:
            
            # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # (B, T, C) --> multi-dimensional so we need reshape in 2 dimensions
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            # Loss = cross entropy on the predictions and the targets
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # Get the idx (B,T)
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            # Cancel out of range
            idx_cond = idx[:,-block_size:]
            
            # Get the predictions
            logits, loss = self(idx_cond)
            
            # Focus on the last step
            # Instead (B,T,C), 
            # Pluck out the last elements in time dimension, because those are the prediction for what comes next
            logits = logits[:, -1, :] # (B,C)
            
            # Convert the probabilities, exponentiate and normalize
            probs = F.softmax(logits, dim=-1) # (B,C)
            
            # Sample the probabilities
            # one of the batch dimensions = single prediction for what comes next
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            
            # Append en make (B,T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device) # calculate nn.Embedding_table on the GPU if 'cuda'

# Take the gradients and update the parameters
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Trainning loop
for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        # Estimate loss function
        losses = estimate_loss()
        # Report accurate for train and test
        print(f"step {iter}: train losse {losses['train']:.4f}, val loss : {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    
    # Look futher in the history
    # Zeroing all the gradients
    optimizer.zero_grad(set_to_none=True)
    # Gettings the gradients for all the parameters
    loss.backward()
    # Using those gradiants to update our parameters
    optimizer.step()
    
# Optimization with device = 'cuda'
context = torch.zeros((1,1), dtype=torch.long, device=device) #(B=1,T=1)

# Generate <max_new_tokens> tokens
# Works on level of batches
# we then have to index into the zero throw to unplug the single batch dimension
# Give a time steps = one dimensional array of all the indices convert in list and decode
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
