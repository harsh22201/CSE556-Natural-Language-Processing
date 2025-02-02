import re
import collections
import json

GROUP_NO = 43  
VOCAB_SIZE = 6000

class WordPieceTokenizer:
    def __init__(self, vocab_size = None):
        self.vocab_size = vocab_size
        self.vocabulary = {"[PAD]", "[UNK]"} # Special tokens  

    def get_vocabulary(self):
        '''
        Returns the vocabulary as a sorted list.
        '''
        return sorted(list(self.vocabulary))

    def preprocess_data(self, text):
        """
        Preprocesses the input text data.
        Steps:
        1. Lowercase all text
        2. Remove unwanted characters (punctuation, special symbols)
        3. Split the text into words
        """
        text = text.lower()  # Convert all to lowercase
        text = re.sub(r'[^\w\s]', '' , text)  # Remove punctuation and special symbols
        words = text.split() # Split text into words
        return words


    def get_splits(self, words):
        """
        Computes the frequency of each word and splits each word into characters.
        """
        word_freqs = collections.defaultdict(int)
        for word in words:
            word_freqs[word] += 1

        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in word_freqs.keys()
        }

        return word_freqs, splits

    def compute_pair_scores(self, word_freqs, splits):
        """ 
        Computes frequency-based score for token pairs. 
        pair_scores = (freq_of_pair)/(freq_of_first_element x freq_of_second_element)
        """
        token_freqs = collections.defaultdict(int)
        pair_freqs = collections.defaultdict(int)

        # Compute frequencies of tokens and token pairs
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                token_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            token_freqs[split[-1]] += freq

        # Compute scores for each token pair
        pair_scores = {}
        for pair, freq in pair_freqs.items():
            pair_scores[pair] = freq / (token_freqs[pair[0]] * token_freqs[pair[1]])

        return pair_scores

    def merge_pair(self, a, b, new_token, splits):
        """
        Merges the highest scoring token pair and updates the splits dictionary.
        """
        for word in splits.keys():
            split = splits[word]
            if (len(split) < 2):
                continue
            i = 0
            while (i < len(split) - 1):
                if (split[i] == a and split[i + 1] == b):
                    split = split[:i] + [new_token] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def construct_vocabulary(self, text):
        """ 
        Builds the vocabulary by merging the highest scoring token pairs iteratively.
        """
        words = self.preprocess_data(text)
        words_freqs, splits = self.get_splits(words)
        
        for split in splits.values():
            for token in split:
                self.vocabulary.add(token) # Add intial tokens to vocabulary

        while len(self.vocabulary) < self.vocab_size:
            scores = self.compute_pair_scores(words_freqs, splits)

            # Find the best pair to merge
            best_pair, max_score = None, -1
            for pair, score in scores.items():
                if max_score < score:
                    best_pair = pair
                    max_score = score
            
            if(best_pair is None): # No more pairs to merge
                print("No more pairs to merge.")
                break

            # Merge the best pair
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            splits = self.merge_pair(*best_pair, new_token, splits)

            # Update vocabulary
            self.vocabulary.add(new_token)

    def encode_word(self, word):
        """
        Encodes a word into a list of tokens using the WordPiece algorithm.
        """
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocabulary:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(self, sentence, pad_size=0):
        """
        Tokenizes the input sentence using the WordPiece algorithm.
        """
        pre_processed_sentence = self.preprocess_data(sentence)
        encoded_words = [self.encode_word(word) for word in pre_processed_sentence]
        tokenized_sentence =  sum(encoded_words, [])

        tokenized_sentence = (["[PAD]"] * pad_size) +  tokenized_sentence + (["[PAD]"] * pad_size) 

        return tokenized_sentence


if __name__ == "__main__":

    tokenizer = WordPieceTokenizer(vocab_size=VOCAB_SIZE)

    # Read corpus.txt
    with open("corpus.txt", "r") as file:
        corpus = file.read()

    # Construct vocabulary from corpus
    tokenizer.construct_vocabulary(corpus)

    # Save vocabulary to file
    with open(f"vocabulary_{GROUP_NO}.txt", "w") as file:
        for token in tokenizer.get_vocabulary():
            file.write(f"{token}\n")
    print(f"Vocabulary saved as 'vocabulary_{GROUP_NO}.txt'.")

    # Read sample_test.json 
    with open("Task 1/sample_test.json", "r") as file:
        sample_test = json.load(file)

    # Tokenize each sentence and store in a dictionary
    tokenized_data = {
        entry["id"]: tokenizer.tokenize(entry["sentence"])
        for entry in sample_test
    }

    # Save tokenized output to a JSON file
    with open(f"tokenized_{GROUP_NO}.json", "w") as file:
        json.dump(tokenized_data, file, indent=4)

    print(f"Tokenized output saved as 'tokenized_{GROUP_NO}.json'.")
