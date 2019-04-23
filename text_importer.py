import hashlib


class TextImporter(object):
    _PUNCTUATION = {'.', ',', '?', '!', ':', ';', '-', '—', '&'}

    def __init__(self, filename):
        self._filename = filename
        self._read_words()
        self._hash = self._calculate_hash()

    def words(self):
        return self._words

    def hash(self):
        return self._hash

    def _read_words(self):
        self._words = []
        with open(self._filename, 'r') as f:
            # Read the file line by line
            for line in f:
                current_word = ''
                # Convert to lowercase
                line = line.lower()
                # Iterate over each symbol in the line
                for i in range(len(line)):
                    symbol = line[i]
                    # If symbol is a letter, just add it to the current word
                    if symbol.isalpha():
                        current_word += symbol
                    # Special processing for the apostrophe symbol
                    elif symbol == '\'' and i + 1 < len(line) and (
                        line[i:i + 2] == '\'s' or
                        line[i:i + 2] == '\'t'  # for words like "don't", "didn't"
                    ):
                        self._append_word(current_word + line[i:i + 2])
                        current_word = ''
                    elif symbol == '\'' and i + 2 < len(line) and (
                        line[i:i + 3] == '\'em' or  # for word "'em" (them)
                        line[i:i + 3] == '\'ll'
                    ):
                        self._append_word(current_word + line[i:i + 3])
                        current_word = ''
                    # Punctuation symbol terminates the current word
                    # In addition, punctuation symbols are considered as 'words' for the purpose
                    # of the neural network training
                    elif symbol in self._PUNCTUATION:
                        if current_word:
                            self._append_word(current_word)
                        current_word = ''
                        self._append_word(symbol)
                    # Whitespace or unusual symbol
                    # Terminate the current word
                    elif current_word:
                        self._append_word(current_word)
                        current_word = ''
                if current_word:
                    self._append_word(current_word)
        return self._words

    def _calculate_hash(self):
        file_hash = hashlib.md5()
        with open(self._filename, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def _append_word(self, word):
        # Word I should be as "I", not "i", undo lowercase transformation
        if word == 'i':
            word = 'I'
        self._words.append(word)
