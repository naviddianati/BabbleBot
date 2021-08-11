'''
Created on Aug 11, 2021

@author: Navid Dianati
'''

import nltk
from nltk.corpus import europarl_raw as eur
import babble

out = nltk.download('europarl_raw')


def load_text_corpus():
    """Load the European parliament proceedings corpus,
    as a single list of tokens"""
    
    nltk.download('europarl_raw')
    corp = eur.english.chapters()
    text = [word for chapter in corp for sentence in chapter for word in sentence]
    return text

    
def main():
    
    bot = babble.BabbleBot()
    
    # Load the text corpus as a single
    # list of words
    text = load_text_corpus()
    bot.parse_text(text)
    bot.learn(epochs=100)
    
    # Give the bot a seed string to get it babbling!
    seed = "Today"
    for i in range(20):
        print(bot.babble(seed, reset=False))


if __name__ == "__main__":
    main()
