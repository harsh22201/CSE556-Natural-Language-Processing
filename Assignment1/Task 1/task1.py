import json
import re

class WordPieceTokenizer:
    def __init__(self):
        self.vocab = []
        self.vocab_set = set()
        self.special_tokens = ['[UNK]', '[PAD]']
        self.splits = []
    
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return words
    
    def _split_word(self, word):
        if not word:
            return []
        tokens = [word[0]]
        for c in word[1:]:
            tokens.append('##' + c)
        return tokens
    
    def construct_vocabulary(self, corpus, vocab_size):
        preprocessed_corpus = []
        for text in corpus:
            preprocessed_corpus.extend(self.preprocess(text))
        
        self.splits = [self._split_word(word) for word in preprocessed_corpus]
        
        initial_vocab = set()
        for split in self.splits:
            for token in split:
                initial_vocab.add(token)
        self.vocab = list(initial_vocab) + self.special_tokens
        self.vocab = list(set(self.vocab))
        
        current_vocab_size = len(self.vocab)
        if current_vocab_size >= vocab_size:
            self._save_vocab()
            self.vocab_set = set(self.vocab)
            return
        
        while current_vocab_size < vocab_size:
            pair_counts = {}
            token_counts = {}
            for split in self.splits:
                for token in split:
                    token_counts[token] = token_counts.get(token, 0) + 1
                for i in range(len(split) - 1):
                    pair = (split[i], split[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
            if not pair_counts:
                break
            
            scores = {}
            for pair, count in pair_counts.items():
                a, b = pair
                fa = token_counts.get(a, 0)
                fb = token_counts.get(b, 0)
                if fa == 0 or fb == 0:
                    scores[pair] = 0.0
                else:
                    scores[pair] = count / (fa * fb)
            
            if not scores:
                break
            best_pair = max(scores, key=lambda k: scores[k])
            a, b = best_pair
            
            merged_token = a + (b[2:] if b.startswith('##') else b)
            
            new_splits = []
            for split in self.splits:
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split)-1 and split[i] == a and split[i+1] == b:
                        new_split.append(merged_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                new_splits.append(new_split)
            self.splits = new_splits
            
            self.vocab.append(merged_token)
            current_vocab_size += 1
        
        self._save_vocab()
        self.vocab_set = set(self.vocab)
    
    def _save_vocab(self):
        with open('vocabulary_43.txt', 'w') as f:
            for token in self.vocab:
                f.write(f"{token}\n")
    
    def tokenize_word(self, word):
        processed_words = self.preprocess(word)
        if not processed_words:
            return ['[UNK]']
        processed_word = processed_words[0]
        initial_split = self._split_word(processed_word)
        tokens = initial_split.copy()
        i = 0
        while i < len(tokens) - 1:
            current = tokens[i]
            next_token = tokens[i+1]
            merged = current + (next_token[2:] if next_token.startswith('##') else next_token)
            if merged in self.vocab_set:
                tokens = tokens[:i] + [merged] + tokens[i+2:]
                i = 0
            else:
                i += 1
        for token in tokens:
            if token not in self.vocab_set:
                return ['[UNK]']
        return tokens
    
    def tokenize(self, sentence):
        words = self.preprocess(sentence)
        tokenized = []
        for word in words:
            tokenized.extend(self.tokenize_word(word))
        return tokenized

f = open('corpus.txt', 'r')
corpus = f.readlines()
f.close()

tokenizer = WordPieceTokenizer()
tokenizer.construct_vocabulary(corpus, 69)

f = open('sample_test.json', 'r')
test_data = json.load(f)
f.close()

tokenized_data = {}
for entry in test_data:
    id_ = entry['id']
    sentence = entry['sentence']
    tokens = tokenizer.tokenize(sentence)
    tokenized_data[id_] = tokens

f = open('tokenized_43.json', 'w')
json.dump(tokenized_data, f)
f.close()


