import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=0):
    """Calculate softmax function for an array x along specified axis.

    Args:
        x (numpy.ndarray): The input array.
        axis (int): The axis along which softmax is calculated. 
                    0 calculates softmax across rows, making each column sum to 1.
                    1 calculates softmax across columns, making each row sum to 1.
    
    Returns:
        numpy.ndarray: The softmax values along the specified axis.
    """
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis)

def tokenize(sentence, token_mapping):
    """Tokenize a sentence using a provided token mapping.

    Args:
        sentence (str): The input sentence to be tokenized.
        token_mapping (dict): A dictionary mapping words to tokens.

    Returns:
        list: A list of tokens corresponding to the words in the input sentence.
             Unknown words are represented by -1.
    """
    tokenized = []
    
    for word in sentence.lower().split(" "):
        try:
            tokenized.append(token_mapping[word])
        except KeyError:
            # Using -1 to indicate an unknown word
            tokenized.append(-1)
        
    return tokenized

def embed(tokens, embeddings):
    """Embed tokens using pre-trained embeddings.

    Args:
        tokens (list): A list of tokens to be embedded.
        embeddings (numpy.ndarray): Pre-trained word embeddings.

    Returns:
        numpy.ndarray: An array of embeddings for the input tokens.
                      Unknown tokens are represented as zero vectors.
    """
    embed_size = embeddings.shape[1]
    
    output = np.zeros((len(tokens), embed_size))
    for i, token in enumerate(tokens):
        if token == -1:
            output[i] = np.zeros((1, embed_size))
        else:
            output[i] = embeddings[token]
            
    return output


def visualize_alignment(alignment, sentence_en, sentence_fr):
    """
    Visualize alignment weights between two sentences.

    Args:
        alignment (numpy.ndarray): The alignment weights between sentences.
        sentence_en (str): The English sentence (source).
        sentence_fr (str): The French sentence (target).

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(alignment, cmap='gray')
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(alignment.shape[1]))
    ax.set_xticklabels(sentence_en.split(" "), rotation=90, size=16)
    ax.set_yticks(np.arange(alignment.shape[0]))
    ax.set_yticklabels(sentence_fr.split(" "), size=16)
    plt.show()

