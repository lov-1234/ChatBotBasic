'''
Implementation of a simple chat module using transformer model.
'''

############################
# Packages
############################
import pandas as pd
import pickle
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import regex as re
from ast import literal_eval
import os
from torch.utils.data import random_split
import matplotlib.ticker


############################
# Constants
############################

# Delimiter of the fields in the files that we are supposed to read
DELIMITER = "+++$+++"
# Escape sequence of the character of the file. It was throwing an error with standard read,
ESCAPE_TYPE = "unicode_escape"
# so I had to read it in binary format and then decode that string
EOS = "<EOS>"
SOS = "<SOS>"
PAD = "<PAD>"
BATCH_SIZE = 32
BATCH_SIZE_GA = 64
THRESHOLD = 20  # Sentence should not be longer than this
PICKLE_FILENAME = "list.pkl"
RANDOM_SAMPLE_THRESHOLD = 30000
MOVIE_CONV_FILE_PATH = 'data/movie_conversations.txt'
MOVIE_LINES_FILE_PATH = 'data/movie_lines.txt'
LOSS_VALID_THRESHOLD_NORMAL = 1.5
LOSS_VALID_THRESHOLD_GA = 1.2

############################
# Classes
############################
# Vocabulary class


class Vocabulary:
    '''
    Class for dealing with our corpus
    '''

    def __init__(self, name, pairs):
        """
        Args:
            name (str): name of the language
            pairs (list): list of pairs of sentences
        """
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.pairs = pairs
        for pair in pairs:
            for sentence in pair:
                # Here we initialise the two dictionaries
                self.add_sentence(sentence.split())

    def add_word(self, word):
        '''
        Add a word to the vocabulary
        :param word: a string
        '''
        if not self.word2index.get(word):
            biggest_val = max(self.word2index.values())
            self.word2index[word] = biggest_val + 1
            self.index2word[biggest_val + 1] = word

    def add_sentence(self, sentence):
        '''
        Add a sentence to the vocabulary
        :param sentence: list of strings (words)
        '''
        for word in sentence:
            self.add_word(word)


def clear_punctuation(s):
    '''
    This function removes all the punctuation from a sentence and insert a blank between any letter and !?.
    :param s: a string
    :return: the "cleaned" string
    '''
    re.sub(r"[^a-zA-Z,.!?]+", r" ",       # Added a comma as well so it separates a comma as well
           s)  # Remove all the character that are not letters, puntuation or numbers
    # Insert a blank between any letter and !?. using regex
    s = re.sub(r"([a-zA-Z])([,!?.])", r"\1 \2", s)
    return s

# Dataset class


class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocabulary, pairs: list):
        self.vocabulary = vocabulary
        self.pairs = pairs
        self.default_pair_one, self.default_pair_two = "<EOS>", "<EOS> <SOS>"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, ix):
        try:
            pair = self.pairs[ix]
        except IndexError:
            pair = self.default_pair_one, self.default_pair_two
        return torch.tensor([self.vocabulary.word2index[i] for i in pair[0].split()]), torch.tensor(
            [self.vocabulary.word2index[i] for i in pair[1].split()])   # tokenisation happens here


class PositionalEncoding(nn.Module):
    '''
    Adapted from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        try:
            assert x.size(0) < self.max_len
        except:
            print("The length of the sequence is bigger than the max_len of the positional encoding. Increase the max_len or provide a shorter sequence.")
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, pad_id=0, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, num_heads=8, dropout_p=0.1):
        super().__init__()
        # Stuff you may need
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.num_heads = num_heads
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=d_model,
            padding_idx=self.pad_id,
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_p)
        self.transformer = nn.Transformer(
            d_model, num_heads, encoder_layers, decoder_layers, dim_feedforward, dropout_p, batch_first=True)
        self.linear = nn.Linear(in_features=d_model,
                                out_features=self.vocab_size)

    def create_padding_mask(self, x, pad_id=0):
        return (x == pad_id)

    def forward(self, src, tgt):
        # S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        # src: (N, S)
        # tgt: (N, T)
        # src_pad_mask: (N, S)
        # tgt_pad_mask: (N, T)
        # mask the future : (N * num_heads, T, T)

        src_pad_mask = self.create_padding_mask(src, self.pad_id)  # (N, S)
        tgt_pad_mask = self.create_padding_mask(tgt, self.pad_id)  # (N, T)

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.pos_encoder(src)  # (N, S, E)
        tgt = self.pos_encoder(tgt)  # (N, T, E)

        # Mask the memory
        memory_key_padding_mask = src_pad_mask  # (N, S)

        # Mask the future
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(1), dtype=torch.bool).to(tgt.device)  # (T, T)
        # Expand to make it N * num_heads, T, T
        tgt_mask = tgt_mask.unsqueeze(0).repeat(
            tgt.size(0) * self.num_heads, 1, 1)  # (N, T, T)
        # Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=memory_key_padding_mask)  # (N, T, E)
        # Linear layer
        output = self.linear(output)  # (N, T, V)

        return output

############################
# Methods
############################


def get_sentence_length_for_all_lines(line_dict):
    sentence_length_dict = dict()
    for k, v in line_dict.items():
        sentence = v.split()
        if len(sentence) != 0:
            sentence_length_dict[k] = len(sentence)
    return sentence_length_dict


def get_word_count_dict(filtered_line_dict):
    res = list(map(lambda x: clear_punctuation(
        x).split(), list(filtered_line_dict.values())))  # Splits the lines by spaces so that we have a tokenised sequence
    word_count_dict = dict()
    for i in res:
        for j in i:
            try:
                word_count_dict[j] += 1  # Increase the word count if we saw it
            except KeyError:
                # Initialise the word count if we see it for the first time
                word_count_dict[j] = 1
    word_count_dict = dict(sorted(word_count_dict.items(),
                                  key=lambda item: item[1], reverse=True))    # gives the sorted word count dict.
    return word_count_dict


def read_conv_and_process_lines(filename=None):
    f = open(filename, 'r')
    lines = f.readlines()
    lines = [literal_eval(i[-1]) for i in [[k.strip() for k in j]
                                           for j in [i.split(DELIMITER) for i in lines]]]
    return lines


def form_conv_pairs(filename_conv=None, filename_lines=None):
    f = open(filename_lines, 'rb')
    lines = f.readlines()
    line_dict = {i[0].strip(): i[-1].strip()
                 for i in [i.decode(ESCAPE_TYPE).split(DELIMITER) for i in lines]}
    conv_list = read_conv_and_process_lines(filename_conv)
    convs = [[line_dict[j] for j in i] for i in conv_list]
    conv_pairs = list()
    for i in convs:
        for j in range(len(i) - 1):
            conv_pairs.append([i[j], i[j + 1]])
    return conv_pairs, line_dict


def add_spaces_for_split(cp):
    for i in range(len(cp)):
        for j in range(len(cp[i])):
            cp[i][j] = clear_punctuation(cp[i][j])


def add_eos_sos(cp):
    for i in range(len(cp)):
        for j in range(len(cp[i])):
            cp[i][j] = " ".join(cp[i][j].split()
                                + [EOS]) if j == 0 else " ".join([SOS] + cp[i][j].split() + [EOS])


def clear_corpus_by_length(length_allowed, corpus):
    new_corpus = list()
    for pairs in corpus:
        pass_thresh = False
        for sentence in pairs:
            if len(sentence.split()) > length_allowed:
                pass_thresh = True
        if not pass_thresh:
            new_corpus.append(pairs[:])
    return new_corpus


def filter_pairs_from_corpus(hit_words, corpus):
    corpus_cp = list()
    for pairs in corpus:
        found = False
        for sentence in pairs:
            for hit_word in hit_words:
                if hit_word in sentence.split():
                    found = True
                    break
            if found:
                break
        if not found:
            corpus_cp.append(pairs[:])
    return corpus_cp


def collate_fn(vocabulary, batch):
    data, targets = zip(*batch)
    pad_value = vocabulary.word2index[PAD]
    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True,
                                            padding_value=pad_value)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True,
                                               padding_value=pad_value)
    return padded_data, padded_targets


def train_one(model: TransformerModel, dataloader, optimizer, loss_fn, device=torch.device("cpu")):
    model.train()
    running_loss = 0
    # print("Starting training")
    for source, target in dataloader:
        # Removes <EOS> from input to decoder, and <SOS> from the target sequence.
        source, input_decoder, output_decoder = source.to(
            device), target[:, :-1].to(device), target[:, 1:].to(device)
        transformer_output = model(source, input_decoder)
        # print("Got output")
        optimizer.zero_grad()
        loss = loss_fn(transformer_output.transpose(1, 2), output_decoder)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    # Returns loss and perplexity value
    return running_loss / float(len(dataloader)), np.exp(running_loss / float(len(dataloader)))


def validate_one(model: TransformerModel, dataloader, loss_fn, device=torch.device("cpu")):
    model.eval()
    running_loss = 0
    # print("Starting Validation")
    with torch.no_grad():
        for source, target in dataloader:
            # Removes <EOS> from input to decoder, and <SOS> from the target sequence.
            source, input_decoder, output_decoder = source.to(
                device), target[:, :-1].to(device), target[:, 1:].to(device)
            transformer_output = model(source, input_decoder)

            loss = loss_fn(transformer_output.transpose(1, 2), output_decoder)
            running_loss += loss.item()
    # Returns loss and perplexity value
    return running_loss / float(len(dataloader)), np.exp(running_loss / float(len(dataloader)))


def train(model: TransformerModel, trainloader, validloader, optimizer, loss_fn, scheduler, print_every=1, max_epochs=20, device=torch.device("cpu")):
    train_loss_history = list()
    train_perplexity_history = list()
    valid_loss_history = list()
    valid_perplexity_history = list()
    for epoch in range(max_epochs):
        train_loss, train_perplexity = train_one(
            model, trainloader, optimizer, loss_fn, device=device)
        valid_loss, valid_perplexity = validate_one(
            model, validloader, loss_fn, device=device)
        train_loss_history.append(train_loss)
        train_perplexity_history.append(train_perplexity)
        valid_loss_history.append(valid_loss)
        valid_perplexity_history.append(valid_perplexity)
        scheduler.step()
        if print_every and (epoch % print_every) == 0:
            print(
                "Epoch: {}/{}, Training Loss: {:8.4f}, Validation Loss: {:8.4f}".format(
                    int(epoch + 1),
                    int(max_epochs),
                    train_loss, valid_loss
                )
            )
    return train_loss_history, train_perplexity_history, valid_loss_history, valid_perplexity_history


def train_two(model: TransformerModel, dataloader, optimizer, loss_fn, accum_iter, device=torch.device("cpu")):
    model.train()
    running_loss = 0
    # print("Starting training")
    for batch_idx, (source, target) in enumerate(dataloader):
        # Removes <EOS> from input to decoder, and <SOS> from the target sequence.
        source, input_decoder, output_decoder = source.to(
            device), target[:, :-1].to(device), target[:, 1:].to(device)
        transformer_output = model(source, input_decoder)
        # print("Got output")
        optimizer.zero_grad()
        loss = loss_fn(transformer_output.transpose(
            1, 2), output_decoder) / accum_iter
        running_loss += loss.item()
        loss.backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
            optimizer.step()
    # Returns loss and perplexity value
    return (running_loss * accum_iter) / float(len(dataloader)), np.exp((running_loss * accum_iter) / float(len(dataloader)))


def train_ga(model: TransformerModel, trainloader, validloader, optimizer, loss_fn, scheduler, accum_iter=4, print_every=1, max_epochs=10, device=torch.device("cpu")):
    train_loss_history = list()
    train_perplexity_history = list()
    valid_loss_history = list()
    valid_perplexity_history = list()
    for epoch in range(max_epochs):
        train_loss, train_perplexity = train_two(
            model, trainloader, optimizer, loss_fn, accum_iter, device=device)
        valid_loss, valid_perplexity = validate_one(
            model, validloader, loss_fn, device=device)
        train_loss_history.append(train_loss)
        train_perplexity_history.append(train_perplexity)
        valid_loss_history.append(valid_loss)
        valid_perplexity_history.append(valid_perplexity)
        scheduler.step()
        if print_every and (epoch % print_every) == 0:
            print(
                "Epoch: {}/{}, Training Loss: {:8.4f}, Validation Loss: {:8.4f}".format(
                    int(epoch + 1),
                    int(max_epochs),
                    train_loss, valid_loss
                )
            )
    return train_loss_history, train_perplexity_history, valid_loss_history, valid_perplexity_history


def sample(model, input_sequence, vocabulary, max_length=THRESHOLD, stop_on=EOS, device=torch.device("cpu"), take_greedy=True, k=5):
    model.eval()
    seed = clear_punctuation(input_sequence)    # Makes spaces in the sentence
    seed = " ".join(seed.split() + [EOS])       # Adds EOS to the end of the sentence
    seed_ind = [dataset.vocabulary.word2index[i] for i in seed.split()] # Makes it into index values of the vocab
    seed_ind = torch.tensor([seed_ind]).to(device)  
    y_input = torch.tensor([[vocabulary.word2index[SOS]]]).to(device)   # We start with the SOS token as the input tgt to the transformer, we will keep appending to it.

    for _ in range(max_length):
        pred = model(seed_ind, y_input)
        if take_greedy:
            next_item = torch.argmax(F.softmax(pred[0, -1, :].detach().cpu(), dim=-1)).item()
            next_item = torch.tensor([[next_item]]).to(device)
        else:
            sf_out = torch.topk(
                F.softmax(pred[0, -1, :].detach().cpu(), dim=-1), k)
            next_item = np.random.choice(sf_out[1].numpy(), 1)
            next_item = torch.tensor([next_item]).to(device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == vocabulary.word2index[stop_on]:
            break

    return y_input.view(-1).tolist()[1:]

if __name__ == "__main__":
    # !!! Don't change the seed !!!
    torch.manual_seed(42)
    np.random.seed(42)

    # !!!!!!

    # Create the pairs
    conv_pairs, line_dict = form_conv_pairs(
        'data/movie_conversations.txt', 'data/movie_lines.txt')
    add_spaces_for_split(conv_pairs)

    # Tokenize the data
    add_eos_sos(conv_pairs) # Not really tokenising it here as doing it later is much easier, I have added EOS and SOS though
    # Filter out the sentences that are too long
    sentence_length_dict = get_sentence_length_for_all_lines(line_dict)
    ld_df = pd.DataFrame.from_dict(sentence_length_dict, 'index')    # Making a dataframe Iof the dict above
    ld_df.columns = ["len"]

    ld_df.hist("len", bins=50)
    plt.axvline(THRESHOLD, color='r', linestyle='dashed', linewidth=1)
    plt.legend(["Threshold"])
    plt.title("Length vs Frequency Distribution")
    plt.xlabel("Sentence Length")
    plt.ylabel("Sentence Frequency")
    plt.savefig("length_vs_freq.png", format="png")
    plt.show()

    ld_df = ld_df[ld_df["len"] < THRESHOLD]     # Only taking the sentences below the threshold
    temp = ld_df.to_dict()["len"]
    new_line_dict = dict()
    for k, _ in temp.items():
        new_line_dict[k] = line_dict[k]     # Storing the sentences below the threshold

    ld_df.hist("len", bins=19)
    plt.title("Length vs Frequency Distribution Below Threshold")
    plt.xlabel("Sentence Length")
    plt.ylabel("Sentence Frequency")
    plt.savefig("length_vs_freq.png", format="png")
    plt.show()
    new_cp = clear_corpus_by_length(THRESHOLD, conv_pairs)

    # Filter out the words that are too rare
    word_count_dict = get_word_count_dict(new_line_dict)
    wdc = (dict(filter(lambda kv: kv[1] < 20, word_count_dict.items())))    # All the words having less than 20 occurences.

    # SAVE and put the code above into a function that you will call if you need to generate something slightly different
    if os.path.isfile(PICKLE_FILENAME):
        print("File Found. Loading ...")
        with open(PICKLE_FILENAME, "rb") as f:
            filtered_list = pickle.load(f)
    else:
        print("File not Found. Building ...")
        filtered_list = filter_pairs_from_corpus(wdc.keys(), new_cp)    # Since I didn't tokenise it before, this part takes a little longer
        with open(PICKLE_FILENAME, "wb") as f:
            pickle.dump(filtered_list, f)
    np_filtered_list = np.array(filtered_list)

    random_samples_idx = np.random.choice(np_filtered_list.shape[0], RANDOM_SAMPLE_THRESHOLD)
    sampled_list = np_filtered_list[random_samples_idx] # Random sampling
    dataset = Dataset(Vocabulary("new_vocab", sampled_list), sampled_list)
    trainset, validset = random_split(dataset, [0.8, 0.2])

    if BATCH_SIZE == 1:
        trainloader, validloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False), DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        trainloader, validloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_fn(dataset.vocabulary, b), shuffle=False), DataLoader(validset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_fn(dataset.vocabulary, b), shuffle=False)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device {device}")
    # Training loop (Consider writing a function for this/two separate functions for training and validation)
    model = TransformerModel(len(dataset.vocabulary.word2index), dropout_p=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    # Evaluation by feeding the model with one input sentence at a time
    train_loss_history, train_perplexity_history, valid_loss_history, valid_perplexity_history = train(
        model, trainloader, validloader, optimizer, loss_fn, scheduler, device=device)  # Normal Training Routine
        # Evaluation by feeding the model with one input sentence at a time

    plt.title("Normal Training Routine")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.axhline(y=LOSS_VALID_THRESHOLD_NORMAL, color="r", label="Loss OK Threshold", linestyle="--")
    plt.plot(train_loss_history, 'b', label="Loss Value Training")
    plt.plot(valid_loss_history, 'r', label="Loss Value Validation")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.savefig(f"./Train_normal_LossvEpochs.png")
    plt.show()

    if BATCH_SIZE_GA == 1:
        trainloader_ga, validloader_ga = DataLoader(trainset, batch_size=BATCH_SIZE_GA, shuffle=False), DataLoader(validset, batch_size=BATCH_SIZE_GA, shuffle=False)
    else:
        trainloader_ga, validloader_ga = DataLoader(trainset, batch_size=BATCH_SIZE_GA, collate_fn=lambda b: collate_fn(dataset.vocabulary, b), shuffle=False), DataLoader(validset, batch_size=BATCH_SIZE_GA, collate_fn=lambda b: collate_fn(dataset.vocabulary, b), shuffle=False)

    model_ga = TransformerModel(
        len(dataset.vocabulary.word2index), dropout_p=0.2).to(device)
    optimizer = optim.Adam(model_ga.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.9)
    train_loss_history, train_perplexity_history, valid_loss_history, valid_perplexity_history = train_ga(
        model_ga, trainloader_ga, validloader_ga, optimizer, loss_fn, scheduler, max_epochs=20, device=device)  # Gradient Accumulation Training Routine

    plt.title("Training Routine with Gradient Accumulation")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.axhline(y=LOSS_VALID_THRESHOLD_GA, color="r", label="Loss OK Threshold", linestyle="--")
    plt.plot(train_loss_history, 'b', label="Loss Value Training")
    plt.plot(valid_loss_history, 'r', label="Loss Value Validation")
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.savefig(f"./Train_grad_accum_LossvEpochs.png")
    plt.show()

    res_g = sample(model, "What are you doing?",
                   dataset.vocabulary, device=device, take_greedy=True)
    res_s = sample(model, "What are you doing?", dataset.vocabulary,
                device=device, take_greedy=False)
    print(
        f'Greedy -> {" ".join([dataset.vocabulary.index2word[i] for i in res_g])}')
    print(
        f'Top_k Random Sample -> {" ".join([dataset.vocabulary.index2word[i] for i in res_s])}')
    print()
    res_g = sample(model, "Hey!", dataset.vocabulary,
                device=device, take_greedy=True)
    res_s = sample(model, "Hey!", dataset.vocabulary,
                device=device, take_greedy=False)
    print(
        f'Greedy -> {" ".join([dataset.vocabulary.index2word[i] for i in res_g])}')
    print(
        f'Top_k Random Sample -> {" ".join([dataset.vocabulary.index2word[i] for i in res_s])}')
    print()
    res_g = sample(model, "What is this?", dataset.vocabulary,
                device=device, take_greedy=True)
    res_s = sample(model, "What is this?", dataset.vocabulary,
                device=device, take_greedy=False)
    print(
        f'Greedy -> {" ".join([dataset.vocabulary.index2word[i] for i in res_g])}')
    print(
        f'Top_k Random Sample -> {" ".join([dataset.vocabulary.index2word[i] for i in res_s])}')
