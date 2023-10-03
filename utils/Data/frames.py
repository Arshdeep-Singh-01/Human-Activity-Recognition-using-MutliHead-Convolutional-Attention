import pandas as pd

# window size
batch_size = 10

def create_sequences(df, window_size=90):
    sequences = []
    labels = []

    for i in range(0, len(df) - window_size,1):
        window = df.iloc[i:i+window_size]
        sequence = window[['x', 'y', 'z']]
        label = window['label'].mode().values[0]
        sequences.append(sequence.values)
        labels.append(label)

    return sequences, labels
