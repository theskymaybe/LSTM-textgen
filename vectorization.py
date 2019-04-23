from numpy import bool, zeros


class Vectorization:
    _WORDS_PER_SAMPLE = 10

    class TextSample:
        def __init__(self, phrase, continuation):
            self.phrase = phrase
            self.continuation = continuation

    def __init__(self, words):
        # Construct a set of unique words of the text
        # by removing duplicate words in the 'words' list
        self._unique_words = set(words)
        self._index_by_word = {}
        self._word_by_index = {}
        for index, word in enumerate(self._unique_words):
            self._index_by_word[word] = index
            self._word_by_index[index] = word
        self._generate_text_samples(words)
        self._generate_vectors()

    def unique_word_count(self):
        return len(self._unique_words)

    def text_samples_count(self):
        return len(self._text_samples)

    def words_per_sample(self):
        return self._WORDS_PER_SAMPLE

    def training_data(self):
        return self._training_data

    def target_data(self):
        return self._target_data

    def encode_word(self, word):
        if word in self._index_by_word:
            return self._index_by_word[word]
        return None

    def decode_word(self, index):
        if index in self._word_by_index:
            return self._word_by_index[index]
        return None

    def decode_words(self, sequence):
        return ' '.join(self.decode_word(word) for word in sequence)

    def _generate_text_samples(self, words):
        STEP_BETWEEN_SAMPLES = 3

        self._text_samples = []

        i = 0
        while i + self._WORDS_PER_SAMPLE < len(words):
            self._text_samples.append(
                Vectorization.TextSample(
                    phrase=words[i:i + self._WORDS_PER_SAMPLE],
                    continuation=words[i + self._WORDS_PER_SAMPLE]
                ))
            i += STEP_BETWEEN_SAMPLES

    def _generate_vectors(self):
        self._training_data = zeros(
            shape=(self.text_samples_count(), self._WORDS_PER_SAMPLE, self.unique_word_count()),
            dtype=bool)
        self._target_data = zeros(
            shape=(self.text_samples_count(), self.unique_word_count()),
            dtype=bool)
        for sample_index, sample in enumerate(self._text_samples):
            for word_in_phrase_index, word in enumerate(sample.phrase):
                self._training_data[sample_index, word_in_phrase_index, self.encode_word(word)] = 1
            self._target_data[sample_index, self.encode_word(sample.continuation)] = 1
