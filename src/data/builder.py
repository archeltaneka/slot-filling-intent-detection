import logging
import string

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SLUDataBuilder:
    def __init__(self, train_split: pd.DataFrame, val_split: pd.DataFrame):
        self.train_split = train_split
        self.val_split = val_split

    def _get_word_shape(self, word):
        # Simple shape: X for uppercase, x for lowercase, d for digit, p for punctuation
        shape = []
        for ch in word:
            if ch.isupper():
                shape.append('X')
            elif ch.islower():
                shape.append('x')
            elif ch.isdigit():
                shape.append('d')
            elif ch in string.punctuation:
                shape.append('p')
            else:
                shape.append('o')
        # Collapse repeats (e.g., xxxx -> x*)
        collapsed = []
        for ch in shape:
            if not collapsed or collapsed[-1] != ch:
                collapsed.append(ch)
        return ''.join(collapsed)


    def _token2features(self, sent_words, i):
        """Extract features for token at position i in sentence.
        sent_words: list of tokens
        i: index
        """
        word = sent_words[i]
        lower = word.lower()
        features = {
            'bias': 1.0,
            'word.lower': lower,
            'word[-3:]': lower[-3:],
            'word[-2:]': lower[-2:],
            'word.isupper': word.isupper(),
            'word.istitle': word.istitle(),
            'word.isdigit': word.isdigit(),
            'word.shape': self._get_word_shape(word),
            'has.hyphen': '-' in word,
            'has.digit': any(ch.isdigit() for ch in word),
            'has.alpha': any(ch.isalpha() for ch in word),
            'is.punct': all(ch in string.punctuation for ch in word),
        }
        
        # Context features: previous 2 and next 2 tokens
        for offset, prefix in [(-2, 'prev2'), (-1, 'prev1'), (1, 'next1'), (2, 'next2')]:
            j = i + offset
            if 0 <= j < len(sent_words):
                w = sent_words[j]
                wl = w.lower()
                features.update({
                    f'{prefix}.lower': wl,
                    f'{prefix}.istitle': w.istitle(),
                    f'{prefix}.isupper': w.isupper(),
                    f'{prefix}.shape': self._get_word_shape(w),
                })
            else:
                features[f'BOS/EOS_{prefix}'] = True
        
        return features


    def _sent2features(self, words):
        return [self._token2features(words, i) for i in range(len(words))]

    def _sent2labels(self, slots):
        return slots

    def _sent2tokens(self, words):
        return words

    def build_crf_dataset(self, df):
        X = [self._sent2features(words) for words in df['words']]
        y = [self._sent2labels(slots) for slots in df['slots']]
        tokens = [self._sent2tokens(words) for words in df['words']]
        return X, y, tokens