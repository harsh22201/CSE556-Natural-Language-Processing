import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from task1 import WordPieceTokenizer, GROUP_NO, VOCAB_SIZE
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 2
BATCH_SIZE = 64
EMBEDDING_DIM = 100
EPOCHS = 100
LEARNING_RATE = 0.005
DROPOUT = 0 # Not used in word2vec training

class Word2VecDataset(Dataset):
    def __init__(self, corpus, tokenizer, window_size=2):
        self.tokenizer = tokenizer
        self.token2idx = {token: i for i, token in enumerate(self.tokenizer.get_vocabulary())} # Create token to index mapping
        self.idx2token = {i: token for token, i in self.token2idx.items()} # Create index to token mapping
        self.window_size = window_size # Total (2*window_size) context words
        self.data = self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        """Prepares CBOW training data using WordPieceTokenizer."""
        corpus = corpus.split('\n')  # Split corpus into sentences
        data = []  # List to store training data
        for sentence in corpus:
            tokenized_sentence = self.tokenizer.tokenize(sentence, pad_size = self.window_size)  # Tokenize using WordPiece
            tokenized_sentence = [self.token2idx.get(token, -1) for token in tokenized_sentence]  # Map tokens to indices (integer values)

            # Generate CBOW training pairs
            for i in range(len(tokenized_sentence) - (2 * self.window_size) ):
                context = tokenized_sentence[i:i + self.window_size] + tokenized_sentence[i + self.window_size + 1:i + (2*self.window_size+1)]
                target = tokenized_sentence[i + self.window_size]
                data.append((context, target))        

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)  


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size = VOCAB_SIZE, embedding_dim = EMBEDDING_DIM, dropout=DROPOUT):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # Embedding layer to convert indices to embeddings
        # self.dropout = nn.Dropout(dropout) 
        self.linear = nn.Linear(embedding_dim, vocab_size) # Fully connected layer to predict target word

    def forward(self, context):
        """Forward pass for CBOW."""
        embedded = self.embeddings(context).mean(dim=1)  # Averaging embeddings
        # embedded = self.dropout(embedded)
        out = self.linear(embedded) 
        return out

def train(model, train_loader, val_loader, epochs=100, lr=0.001):

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss is used because the model predicts a probability distribution over the vocabulary.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train() # Set moSets the model to training mode (dropout/batch normalization if used).
        train_loss = 0.0 # Training loss for the epoch
        for context, target in train_loader: # Iterate over the training dataset batch by batch
            context, target = context.to(model.embeddings.weight.device), target.to(model.embeddings.weight.device) # Move data to the device (GPU/CPU)
            optimizer.zero_grad() # Clears the gradients from the previous iteration.
            output = model(context) # Forward pass
            loss = criterion(output, target) # Compute loss 
            loss.backward() # Backpropagates the loss to compute gradients.
            optimizer.step() # Updates the model parameters using the computed gradients.
            train_loss += loss.item() # Accumulate loss for the epoch
        train_losses.append(train_loss / len(train_loader)) # Average training loss for the epoch
        
        model.eval() # Sets the model to evaluation mode (disables dropout/batch normalization if used).
        val_loss = 0.0
        with torch.no_grad(): # Disables gradient computation for validation
            for context, target in val_loader: # Iterate over the validation dataset batch by batch
                context, target = context.to(model.embeddings.weight.device), target.to(model.embeddings.weight.device) # Move data to the device (GPU/CPU)
                output = model(context) # Forward pass
                loss = criterion(output, target) # Compute loss
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader)) # Average validation loss for the epoch
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("cbow_loss_plot.png")
    plt.show()

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_triplets(dataset, model, no_of_triplets=2):
    """
    Returns a list of triplets: (anchor, positive, negative), 
    where anchor and positive have high cosine similarity, 
    and negative has low similarity with the anchor.
    """
    triplets = []
    embeddings = model.embeddings.weight.data.cpu().numpy()  # Get embedding matrix
    print(embeddings.shape)
    tokens_idx = list(dataset.idx2token.keys())  # List of token indices
    token_list = list(dataset.idx2token.values())  # List of actual tokens

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings, embeddings.T)  # Dot product of embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix /= np.dot(norms, norms.T)  # Normalize to get cosine similarity
    
    # Generate triplets
    np.random.shuffle(tokens_idx)  # Shuffle to avoid bias
    for anchor_idx in tokens_idx[:no_of_triplets]:  # Pick triplet count
        similarities = similarity_matrix[anchor_idx]  # Similarity scores for the anchor
        
        # Find the most similar (positive) and least similar (negative) words
        sorted_indices = np.argsort(-similarities)  # Sort in descending order (highest first)
        positive_idx = sorted_indices[1]  # First most similar (excluding itself)
        negative_idx = sorted_indices[-1]  # Least similar word

        anchor, positive, negative = token_list[anchor_idx], token_list[positive_idx], token_list[negative_idx]
        triplets.append((anchor, positive, negative))

    return triplets


if __name__ == "__main__":

    # Load corpus
    with open("corpus.txt", "r") as file:
        corpus = file.read()

    # Initialize tokenizer from Task 1
    tokenizer = WordPieceTokenizer()

    # Load vocabulary   
    # Method 1: Load vocabulary from file
    with open(f"vocabulary_{GROUP_NO}.txt", "r") as file:  
        for line in file:
            tokenizer.vocabulary.add(line.strip())
    # Method 2: Construct vocabulary from corpus
    # tokenizer.construct_vocabulary(corpus) 

    # Create dataset
    dataset = Word2VecDataset(corpus, tokenizer, window_size=WINDOW_SIZE)
    # Split dataset into training and validation sets (90:10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Word2Vec CBOW model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbow_model = Word2VecModel(vocab_size=len(tokenizer.vocabulary), embedding_dim=EMBEDDING_DIM).to(device)

    # Train model
    train_losses, val_losses = train(cbow_model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    # Save model
    torch.save(cbow_model.state_dict(), "word2vec_cbow.pth")
    print("CBOW Model trained and checkpoint saved as 'word2vec_cbow.pth'.")
    # Plot losses
    plot_losses(train_losses, val_losses)
    # get triplets
    print(find_triplets(dataset, cbow_model, no_of_triplets=2))
