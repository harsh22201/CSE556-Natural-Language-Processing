import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from task1 import WordPieceTokenizer, GROUP_NO
from task2 import Word2VecDataset, Word2VecModel
import numpy as np
import matplotlib.pyplot as plt

CONTEXT_SIZE_LM = 4
BATCH_SIZE_LM = 512
LEARNING_RATE_LM = 0.0005
EPOCHS_LM = 10

class NeuralLMDataset(Dataset):
    def __init__(self, corpus, tokenizer, word2vec_model, context_size):
        """
        Custom dataset for training a Neural Language Model.
        
        Args:
            corpus (str): The text corpus.
            tokenizer (WordPieceTokenizer): The tokenizer from Task 1.
            word2vec_model (Word2VecModel): Pre-trained Word2Vec model from Task 2.
            context_size (int): Number of previous words to use as input.
        """
        self.tokenizer = tokenizer
        self.token2idx = {token: i for i, token in enumerate(self.tokenizer.get_vocabulary())} # Create token to index mapping
        self.word2vec_model = word2vec_model
        self.context_size = context_size
        # Tokenize corpus and map tokens to embeddings
        self.data = self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        """Tokenizes the corpus and prepares training data."""
        corpus = corpus.split("\n")  # Split into sentences
        data = []

        for sentence in corpus:
            tokenized_sentence = self.tokenizer.tokenize(sentence, pad_size = self.context_size)  # Tokenize using WordPiece
            tokenized_sentence = [self.token2idx.get(token, -1) for token in tokenized_sentence]  # Map tokens to indices (integer values)

            # Generate context-target pairs
            for i in range(len(tokenized_sentence) - (2 * self.context_size)):
                context = tokenized_sentence[i:i + self.context_size]  # Previous N words
                target = tokenized_sentence[i + self.context_size]  # Next word
                data.append((context, target))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns concatenated word embeddings as input and target word index."""
        context, target = self.data[idx]

        # Convert context tokens to embeddings (get embeddings from Word2Vec model)
        with torch.no_grad():  # Disable gradient tracking
            context_embeddings = self.word2vec_model.embeddings(torch.tensor(context, dtype=torch.long))

        # Flatten the concatenated embeddings (size: embedding_dim * context_size)
        context_vector = context_embeddings.view(-1)

        return context_vector, torch.tensor(target, dtype=torch.long)



class NeuralLM1(nn.Module):
    """Single hidden layer [256] with ReLU activation."""
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NeuralLM1, self).__init__()
        input_dim = embedding_dim * context_size  # Compute input size
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, vocab_size)  # Predict next word index

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)  # No softmax (CrossEntropyLoss applies it internally)
        return x


class NeuralLM2(nn.Module):
    """Two hidden layers [256, 512] with ReLU activation."""
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NeuralLM2, self).__init__()
        input_dim = embedding_dim * context_size
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x


class NeuralLM3(nn.Module):
    """Single hidden layer [256] with Tanh activation."""
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NeuralLM3, self).__init__()
        input_dim = embedding_dim * context_size
        self.fc1 = nn.Linear(input_dim, 256)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.output(x)
        return x

def train(model, train_loader, val_loader, epochs, lr):

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss is used because the model predicts a probability distribution over the vocabulary.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train() # Set moSets the model to training mode (dropout/batch normalization if used).
        train_loss = 0.0 # Training loss for the epoch
        for context, target in train_loader: # Iterate over the training dataset batch by batch
            context, target = context.to(next(model.parameters()).device), target.to(next(model.parameters()).device) # Move data to the device (GPU/CPU)
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
                context, target = context.to(next(model.parameters()).device), target.to(next(model.parameters()).device) # Move data to the device (GPU/CPU)
                output = model(context) # Forward pass
                loss = criterion(output, target) # Compute loss
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader)) # Average validation loss for the epoch
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

def compute_accuracy(model, dataloader, device):
    """Computes accuracy of the model on a given dataset."""
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Forward pass
            
            # Get predicted token (highest probability)
            predicted = torch.argmax(outputs, dim=1)
            
            # Compare with actual target tokens
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total * 100  # Convert to percentage
    return accuracy


def compute_perplexity(model, dataloader, device):
    """Computes perplexity score for the model on a given dataset."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_words = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Forward pass
            
            # Compute loss using cross-entropy
            loss = torch.nn.functional.cross_entropy(outputs, targets, reduction="sum")
            total_loss += loss.item()
            total_words += targets.size(0)

    # Compute perplexity = exp(average loss)
    perplexity = torch.exp(torch.tensor(total_loss / total_words)).item()
    return perplexity


def predict_next_tokens(model, sentence, tokenizer, word2vec_model, context_size, num_predictions=3):
    """
    Predicts the next 'num_predictions' tokens one by one for a given sentence.
    
    Args:
        model: Trained neural language model.
        sentence (str): Input sentence.
        tokenizer (WordPieceTokenizer): Tokenizer.
        word2vec_model (Word2VecModel): Pre-trained Word2Vec model.
        context_size (int): Context window size.
        num_predictions (int): Number of tokens to predict.
    
    Returns:
        List of predicted tokens.
    """
    model.eval()  # Set model to evaluation mode
    predicted_tokens = []
    
    # Tokenize input sentence
    tokenized_sentence = tokenizer.tokenize(sentence, pad_size=context_size)
    
    # Convert tokens to indices
    token_to_idx = {token: i for i, token in enumerate(tokenizer.get_vocabulary())}
    idx_to_token = {i: token for token, i in token_to_idx.items()}
    context = [token_to_idx.get(token, -1) for token in tokenized_sentence]  # Map tokens to indices
    context = context[-2*context_size:-context_size]  # Limit context to context_size 
    
    for _ in range(num_predictions):
        # Convert context tokens to embeddings
        with torch.no_grad():
            context_tensor = torch.tensor(context, dtype=torch.long).to(next(model.parameters()).device)
            context_embeddings = word2vec_model.embeddings(context_tensor).view(-1)  # Flatten
        
        # Predict next token
        output = model(context_embeddings.unsqueeze(0))  # Add batch dimension
        predicted_idx = torch.argmax(output, dim=1).item()  # Get highest probability token
        
        # Get the corresponding token
        predicted_token = idx_to_token.get(predicted_idx, "[UNK]")  # Default to [UNK] if token not found
        predicted_tokens.append(predicted_token)
        
        # Update context by adding the predicted token and removing the first one
        context.append(predicted_idx)
        context.pop(0)  # Maintain context size

    return predicted_tokens


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained Word2Vec  model
    trained_cbow_checkpoint = torch.load("word2vec_cbow.pth", map_location=device, weights_only=True)
    saved_vocab_size, saved_embedding_dim = trained_cbow_checkpoint["embeddings.weight"].shape
    word2vec_model = Word2VecModel(vocab_size=saved_vocab_size, embedding_dim=saved_embedding_dim)
    word2vec_model.load_state_dict(trained_cbow_checkpoint)
    word2vec_model.eval() # Set model to evaluation mode

    # Create dataset
    dataset = NeuralLMDataset(corpus, tokenizer, word2vec_model,context_size=CONTEXT_SIZE_LM)
    
    # Split dataset into training and validation sets (90:10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_LM, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_LM, shuffle=False)

    for model_class, name in zip([NeuralLM1, NeuralLM2, NeuralLM3], ["LM1", "LM2", "LM3"]):
        print(f"\nTraining {name}...")
        model = model_class(saved_vocab_size, saved_embedding_dim, context_size=CONTEXT_SIZE_LM).to(device)
        train_losses, val_losses = train(model, train_loader, val_loader, epochs=EPOCHS_LM, lr=LEARNING_RATE_LM)

        # Compute accuracy and perplexity
        train_accuracy = compute_accuracy(model, train_loader, device)
        val_accuracy = compute_accuracy(model, val_loader, device)
        train_perplexity = compute_perplexity(model, train_loader, device)
        val_perplexity = compute_perplexity(model, val_loader, device)

        print(f"{name} - Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")
        print(f"{name} - Train Perplexity: {train_perplexity:.2f}, Validation Perplexity: {val_perplexity:.2f}")
        
        # Save model
        torch.save(model.state_dict(), f"neural_{name}.pth")

        # Plot losses
        plt.plot(train_losses, label=f"Train {name}")
        plt.plot(val_losses, label=f"Val {name}")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="lower left") 
    plt.title("Training & Validation Loss")
    plt.savefig("Neural_LM_losses")
    plt.show()


