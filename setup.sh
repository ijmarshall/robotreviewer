echo "Creating conda environment.."
conda env create -f robotreviewer_env_local.yml
echo "Done.\n\n"

echo "Setting up requirements.."
source activate robotreviewer3
python setup.py
echo "Done.\n\n"

echo "Starting up rabbitmq.."
brew update
brew install rabbitmq
brew services start rabbitmq
echo "Started."

source deactivate
echo "Setup finished."
