import string
import re


# return the tokens list converted to ids
def tokens_to_ids(tokens: list, vocab: dict):
    return [vocab.get(token, vocab.get("<unk>")) for token in tokens]

# tokenization function
def greedy_tokenize(text:str, vocab: dict):
    ## Prep text with </w> tokens
    # split text by spaces to remove spaces
    tokens = text.split()
    separated_text = re.findall(r"\w+|[^\w\s]", text)
    # reconstruct text with no spaces and with </w> tokens at the end of words
    # KNOWN ISSUE: stuff like Ca2+ gets a </w> character between 2 and + to be Ca2</w>+
    text = ""
    for word in separated_text:
        # append only word if it is punctuation
        if word in string.punctuation:
            text += word
        else:
            text += f"{word}</w>"

    # Tokenize text
    i = 0
    tokens = []
    while i < len(text):
        matched = False
        for j in range(len(text), i, -1):
            sub = text[i:j]
            if sub in vocab:
                tokens.append(sub)
                i = j
                matched = True
                break
        
        # if cant find matching token
        if not matched:
            # append unknown token and increment
            tokens.append("<unk>")
            i += 1
    return tokens