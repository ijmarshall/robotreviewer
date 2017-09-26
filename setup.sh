echo "Creating conda environment.."
conda env create -f robotreviewer_env_local.yml
echo "Done."

echo "Setting up requirements.."
source activate robotreviewer3
python setup.py
echo "Done."

echo "Starting up rabbitmq.."
brew update
brew install rabbitmq
brew services start rabbitmq
echo "Started."

echo "Installing spacy requirements.."
python -m spacy.en.download
echo "Done."

source deactivate
echo "Setup finished."
