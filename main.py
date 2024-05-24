import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

batch_size = 32
chunk_size = 8

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
    return x, y


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        logits = self.token_embedding_table(x)

        if y is None:
            loss = None
        else:
            logits = logits.view(batch_size * chunk_size, vocab_size)
            targets = y.view(batch_size * chunk_size)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, current_tokens, max_new_tokens):

        for i in range(max_new_tokens):
            logits, loss = self.forward(current_tokens)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            current_tokens = torch.cat((current_tokens, next_token), dim=1)

        return current_tokens


xb, yb = get_batch('train')

model = BigramLanguageModel(vocab_size)
logits_outer, loss_outer = model.forward(xb, yb)

current_tokens = torch.zeros((1, 1), dtype=torch.long)
max_new_tokens = 100
print(decode(model.generate(current_tokens, max_new_tokens)[0].tolist()))

# Optimization step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(10000):
    x_train, y_train = get_batch('train')

    logits, loss = model(x_train, y_train)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(current_tokens, max_new_tokens)[0].tolist()))
