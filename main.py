import os
from model import compile_model, generate_text, train_model, setup_model
from text_importer import TextImporter
from vectorization import Vectorization


TEXTS_FOLDER = 'texts'
WEIGHTS_FOLDER = 'weights'
MAX_EPOCH = 50
GENERATE_TEXT_WORDS_COUNT = 350


def ensure_weights_folders_exists():
    if not os.path.exists(WEIGHTS_FOLDER):
        os.mkdir(WEIGHTS_FOLDER)
    elif os.path.isfile(WEIGHTS_FOLDER):
        raise Exception('{} is a file, must be a folder'.format(WEIGHTS_FOLDER))


def try_find_precalculated_weights(hash):
    filename_prefix = 'weights_{}-{}'.format(hash, MAX_EPOCH)
    files = os.listdir(WEIGHTS_FOLDER)
    for file in files:
        if file.startswith(filename_prefix):
            return os.path.join(WEIGHTS_FOLDER, file)
    # Cannot find weights file
    return None


def get_seed_from_user(seed_max):
    while True:
        try:
            print('You might want to set seed above, say 2000, to skip text on title pages')
            seed = int(input('Please enter seed (integer from 1 to {})\n'.format(seed_max)))
            if 1 <= seed <= seed_max:
                return seed
        finally:
            pass


def get_diversity_from_user():
    while True:
        try:
            diversity = float(input('Please enter diversity (float from 0.2 to 1.5)\n'))
            if 0.2 - 1e-8 < diversity < 1.5 + 1e-8:
                return diversity
        finally:
            pass


def get_textfile_from_user():
    while True:
        try:
            print('Please place text file into {} folder'.format(TEXTS_FOLDER))
            filenames = os.listdir(TEXTS_FOLDER)
            print('List of files in {} folder'.format(TEXTS_FOLDER))
            for i, filename in enumerate(filenames):
                print('[{}]\t{}'.format(i + 1, filename))
            choice = int(input('Please enter your choice (integer from 1 to {})\n'.format(
                len(filenames)
            )))
            if 1 <= choice <= len(filenames):
                return os.path.join(TEXTS_FOLDER, filenames[choice - 1])
        finally:
            pass


def main():
    filename = get_textfile_from_user()
    print('Reading file {}'.format(filename))
    text_importer = TextImporter(filename)
    print('Words read:\t{}'.format(len(text_importer.words())))
    print('File hash:\t{}'.format(text_importer.hash()))

    print('Starting vectorization')
    vectorization = Vectorization(words=text_importer.words())
    print('Finished vectorization')
    print('Unique words count:\t{}'.format(vectorization.unique_word_count()))
    print('Text samples count:\t{}'.format(vectorization.text_samples_count()))

    print('Setting up the model')
    lstm_model = setup_model(vectorization)
    weights_filename_template = os.path.join(
        WEIGHTS_FOLDER,
        'weights_' + text_importer.hash() + '-{epoch}-{loss:.5f}.hdf5')
    ensure_weights_folders_exists()

    # Load weights is they have been calculated already
    weights_filename = try_find_precalculated_weights(text_importer.hash())
    if weights_filename:
        print('Using existing weights from file {}'.format(weights_filename))
        lstm_model.load_weights(weights_filename)
        compile_model(lstm_model)
    else:
        compile_model(lstm_model)
        train_model(lstm_model, vectorization, weights_filename_template,
                    MAX_EPOCH)

    while True:
        print('\n\nReady to generate a text (press Ctrl+C to terminate)')
        seed_max = len(text_importer.words()) - GENERATE_TEXT_WORDS_COUNT - 1
        seed = get_seed_from_user(seed_max)
        diversity = get_diversity_from_user()
        print('Generating text')
        generated_text = generate_text(text_importer.words(), lstm_model, vectorization, seed, diversity)
        print(generated_text)
        print('\n')


if __name__ == '__main__':
    main()
