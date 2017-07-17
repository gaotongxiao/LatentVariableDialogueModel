# LatentVariableDialogueModel
An implementation of "Latent Variable Dialogue Models and their Diversity"

# Usage
1. extract train.txt from opensubtitle(https://s3.amazonaws.com/opennmt-trainingdata/opensub_qa_en.tgz) to data/
2. run 1.GetPartialData.py to get first 10000 question-answer pairs without punctuation
3. run 2.Process.py with customized hyperparameters defined in the first few lines to use this model
