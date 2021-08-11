# BabbleBot
Text synthesis bot that learns from a given text corpus using an LSTM-based deep neural network and produces new text given a "seed" prompt.

We treat the text as one long list of tokens, break it up into <batch_size> contiguous sequences, divide this stack of long sequences into batches, and process the batches in order using a deep neural network with a stateful LSTM. The learning task is to predict the next token in each sequence given the current timestamp and the current state of the LSTM layer.

The example used is the "Sample European Parliament Proceedings Parallel Corpus" downloaded directly using the NLTK package:

```python
    bot = babble.BabbleBot()
    
    # Load the text corpus as a single list of words
    text = load_text_corpus()
    
    # Parse the data and generate the encodings
    # required for training
    bot.parse_text(text)
    
    # Instantiate the model and train
    bot.learn(epochs=100)
    
    # Give the bot a seed string to get it babbling!
    seed = "Today"
    for i in range(20):
        print(bot.babble(seed, reset=False))
```
