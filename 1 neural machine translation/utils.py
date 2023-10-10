import numpy as np
import matplotlib.pyplot as plt 


def softmax(x, axis=0):
    """ Calculate softmax function for an array x along specified axis
    
        axis=0 calculates softmax across rows which means each column sums to 1 
        axis=1 calculates softmax across columns which means each row sums to 1
    """
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis)


def tokenize(sentence, token_mapping):
    tokenized = []
    
    for word in sentence.lower().split(" "):
        try:
            tokenized.append(token_mapping[word])
        except KeyError:
            # Using -1 to indicate an unknown word
            tokenized.append(-1)
        
    return tokenized



def embed(tokens, embeddings):
    embed_size = embeddings.shape[1]
    
    output = np.zeros((len(tokens), embed_size))
    for i, token in enumerate(tokens):
        if token == -1:
            output[i] = np.zeros((1, embed_size))
        else:
            output[i] = embeddings[token]
            
    return output

def visualize_alignment(alignment, sentence_en, sentence_fr):
    # Visualize weights to check for alignment
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(alignment, cmap='gray')
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(alignment.shape[1]))
    ax.set_xticklabels(sentence_en.split(" "), rotation=90, size=16);
    ax.set_yticks(np.arange(alignment.shape[0]));
    ax.set_yticklabels(sentence_fr.split(" "), size=16);