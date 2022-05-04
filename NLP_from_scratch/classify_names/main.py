# Cell
from __future__ import unicode_literals, print_function, division
from io import open

import torch
import torch.nn as nn

from glob import glob
import os
import unicodedata
import string
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Cell
dataroot = "/media/king_rob/DataDrive/data/torch_tutorials/name_data"
# Cell
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        and c in all_letters
    )

category_lines = {}
all_categories = []

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return list(map(unicode_to_ascii, lines))

for filename in glob(os.path.join(dataroot, "names/*.txt")):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


# Cell
def letter_to_idx(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tens = torch.zeros(1, n_letters)
    tens[0][letter_to_idx(letter)] = 1
    return tens

def line_to_tensor(line):
    return torch.cat([
        letter_to_tensor(c).unsqueeze(0)
        for c in line
    ], dim=0)


# Cell
letter_to_tensor('j')
# Cell
line_to_tensor('Jones').size()

# Cell
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, height):
#         super().__init__()

#         self.hidden_size = hidden_size

#         self.i2h = nn.Sequential(
#             nn.Linear(input_size + hidden_size, height),
#             nn.ReLU(),
#             nn.Linear(height, height),
#             nn.ReLU(),
#             nn.Linear(height, hidden_size)
#         )

#         self.h2o = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.h2o(hidden)
#         output = self.softmax(output)
#         return output, hidden

#     def init_hidden(self, device=torch.device('cuda')):
#         return torch.zeros(1, self.hidden_size).to(device)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, prev):
        hidden, cell_state = prev

        lstm_out, (next_hidden, next_cell) = self.lstm(input, (hidden, cell_state))

        out = self.h2o(lstm_out)

        return out, (next_hidden, next_cell)

    def init_hidden(self, device=torch.device('cuda')):
        hidden = torch.zeros(1, self.hidden_size).to(device)
        cell_s = torch.zeros(1, self.hidden_size).to(device)

        return hidden, cell_s


# Cell
n_hidden = 128
height = 128
# rnn = nn.LSTM(n_letters, n_hidden, n_categories, height)
rnn = LSTM(n_letters, n_hidden, n_categories)

# Cell
inp = line_to_tensor("Robert")
hidden, cell_s = rnn.init_hidden(torch.device('cpu'))
output, (next_hidden, next_c) = rnn(inp[0], (hidden, cell_s))
print(output)
# Cell
def category_from_output(output):
    pred = output.argmax(1)[0].item()
    return all_categories[pred], pred

print(category_from_output(output))

# Cell
def random_train_sample():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = random_train_sample()
    print(f'{category = }, {line = }')

# Cell
lr = 0.002
rnn = rnn.cuda()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(rnn.parameters(), lr=lr)


def train(category_tensor, line_tensor, optim):
    hidden, cell_s = rnn.init_hidden()
    optim.zero_grad()

    # category_tensor, line_tensor = category_tensor.cuda(), line_tensor.cuda()

    for i in range(line_tensor.size(0)):
        out, (hidden, cell_s) = rnn(line_tensor[i], (hidden, cell_s))

    loss = criterion(out, category_tensor)

    loss.backward()

    optim.step()

    return out.detach().cpu(), loss.detach().cpu().item()

# Cell
n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    return m

# Cell
start_time = time.time()

for i in range(n_iters):
    cat, line, cat_tensor, line_tensor = random_train_sample()
    cat_tensor, line_tensor = cat_tensor.cuda(), line_tensor.cuda()
    out, loss = train(cat_tensor, line_tensor, optim)
    current_loss += loss

    if (i + 1) % print_every == 0:
        print(f"Iteration {i} / {n_iters - 1} [{time_since(start_time)}m]")
        print(f'\t{loss =}')
        pred = out.argmax(1)[0]
        pred_cat = all_categories[pred]
        print(f"\tSample: Line: {line}, Guess: {pred_cat}, Corect: {cat}")

    if (i + 1) % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Cell
plt.plot(all_losses)
plt.show()
# Cell
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size(0)):
        out, hidden = rnn(line_tensor[i], hidden)
    return out

# fill the confusion matrix
rnn.eval()
with torch.no_grad():
    for i in range(n_confusion):
        cat, line, cat_tensor, line_tensor = random_train_sample()
        cat_tensor, line_tensor = cat_tensor.cuda(), line_tensor.cuda()

        out = evaluate(line_tensor)

        pred, pred_i = category_from_output(out.detach().cpu())

        cat_idx = all_categories.index(cat)

        confusion[cat_idx][pred_i] += 1

confusion /= confusion.sum(1)

# Cell
confusion

# Cell
fig = plt.figure(dpi=100)
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()
# Cell
