import json
import logging
import pip
import os
import shutil


def create_config():
    shutil.copyfile('robotreviewer/config.json.example', 'robotreviewer/config.json')
    with open('robotreviewer/config.json', 'r') as f:
        obj = json.load(f)
    print('Please enter path to grobid:')
    print('EXAMPLE: {}'.format('/Users/Robot/Documents/grobid'))
    path = input()
    obj['robotreviewer']['grobid_path'] = path
    with open('robotreviewer/config.json', 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    print('Installing nltk.')
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    print('Ensuring Keras backend is set to tensorflow.')
    # Ensure Keras backend is Theano
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    print('')

    # Create config file
    print('Setting up configuration file...')
    if not os.path.isfile('robotreviewer/config.json'):
        create_config()
    print('')
