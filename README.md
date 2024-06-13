# ChatBotBasic
This project aims to build a simple chatbot that can reply to some messages depending on a prompt. We aim to better understand PyTorch and some basic Machine Learning Principles here.

## Introduction
In this project, we will train a conversational model. We will leverage the Cornell Movie Dialogues Corpus — a rich dataset containing movie character dialogues. We will follow the following steps in doing so:

## Steps:
  - Download and inspect the data from [Cornell Link Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
  - Create a list of pairs, each containing pairs of sentences (pairs of string at this level). Note that some conversations are not simple question-and-answer conversations, but we want pairs at the end of the day. If one conversation contains more than two sentences, for instance, 4 sentences, namely $(s_1, s_2, s_3, s_4)$, we take the successive combinations as pairs. Therefore, at the end of it all, we will be left with the following tuples: $(s_1, s_2), (s_2, s_3), (s_3, s_4)$
  - Tokenize the sentences at a word level, for instance. Create one token for each punctuation symbol, such as ?., ! and remove other types of punctuation.
  - Append at the end of each sentence the token <EOS> and at the beginning of each answer - just for the answers! - the token <SOS>.
  -  Remove all the pairs where a sentence is longer than a certain length max length. (Plotting a distribution might help)
  -  Count the words in your corpus and eliminate all the words below a certain threshold. We look at a frequency distribution to decide on a suitable threshold. Then, we eliminate all the pairs that have at least one sentence containing one unknown word, namely, a word you have eliminated.  <b>Note</b>: This is not what you want to do in practice. In practice, you want to replace each unknown word with a dedicated token <UNK>, but for the purposes of our assignment, this will do.
  -  Implement the Transformer
