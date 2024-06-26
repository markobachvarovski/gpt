import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
chunk_size = 256
epoches = 5000
learning_rate = 3e-4
eval_iters = 200
eval_interval = 500
num_embed_dim = 84
# head_size = 16
n_layer = 6
n_head = 6
dropout = 0.2

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - chunk_size, (batch_size,))
    x = torch.stack([data[i:i + chunk_size] for i in ix])
    y = torch.stack([data[i + 1:i + chunk_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed_dim)
        self.position_embedding_table = nn.Embedding(chunk_size, num_embed_dim)
        self.lm_head = nn.Linear(num_embed_dim, vocab_size)
        self.self_attention_heads = MultiHeadAttention(4, num_embed_dim // 4)
        self.blocks = nn.Sequential(*[Block(num_embed_dim, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(num_embed_dim)

    def forward(self, x, y=None):
        B, T = x.shape

        token_embeddings = self.token_embedding_table(x)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = y.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, current_tokens, max_new_tokens):

        for i in range(max_new_tokens):
            cropped_tokens = current_tokens[:, -chunk_size:]
            logits, loss = self.forward(cropped_tokens)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            current_tokens = torch.cat((current_tokens, next_token), dim=1)

        return current_tokens


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embed_dim, head_size, bias=False)
        self.query = nn.Linear(num_embed_dim, head_size, bias=False)
        self.value = nn.Linear(num_embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(chunk_size, chunk_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, num_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


xb, yb = get_batch('train')

model = BigramLanguageModel().to(device)
logits_outer, loss_outer = model.forward(xb, yb)

current_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
max_new_tokens = 100
print("Before optimization:")
print(decode(model.generate(current_tokens, max_new_tokens)[0].tolist()) + "\n")

# Optimization step
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(epoches):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    x_train, y_train = get_batch('train')

    logits, loss = model(x_train, y_train)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("After optimization:")
print(decode(model.generate(current_tokens, max_new_tokens)[0].tolist()))
