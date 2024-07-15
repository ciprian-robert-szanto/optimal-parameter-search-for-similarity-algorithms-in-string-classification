from nltk.tokenize import word_tokenize


def _preprocess(text):
    """
    Preprocess the data in order to make the words 
    lowercase and split each word in the sentence.
    """
    return word_tokenize(text.lower(), language="italian")
