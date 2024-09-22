# Training LMs with synthetic data

This repo contains the code used to train Language Models (LMs) to perform Context Leveraging OCR Correction (CLOCR-C). 
The code is related to the [scrambledtext](https://github.com/JonnoB/scrambledtext) repo as both were used in the paper "Scrambled text: training Language Models to correct OCR errors using synthetic data".
This repo is designed to be run on the lightning.ai platform and can be opened in as a lightning studio using the button above. To reduce cost the [unsloth.ai](https://github.com/unslothai/unsloth) implementation of Llama3 is used.
Other models can be used however using unsloth reduces RAM and increases training time, see their documentation for details.

# Key files in the repo
The most important file in the repo is is `training_script.py`. This python script can be executed from the command line using its arguments to train an LM to perform CLOCR-C using training data and base LM of your choice.

# Using this repo

There are several different ways of using this repo, from basic use just to see how it works, reproducing the results in the paper, to training your own custom model.

## Before you start

- Make sure you have initialised your weights and biases account in the studio
- Have a hugging face token ready if you plant to use other models than those provided by Unsloth.

## Basic use

- Open as a studio in lightning
- run `python training_script.py xxx xxx xxx` to run the example script
- The output will be saved to the folder xxx

## Reproducing the work in the paper

- Open as a studio in lightning
- Upload the training and test data from the data repository, the necessary files are as follows
    - `synth_gt.zip`: The synthetic training data
    - `ncse_hf_dataset: A hugginface dataset of the NCSE test data
- Run the following scripts
  - `cer_Experiment_grid.py`:Train 9 models on different CER values with a uniform distribution
  - `cer_wer_grid.py`: Train 35 models on different CER WER pairs.
  - `blend_experiment.py` Train 2 models on a mixtures of CER WER pairs.
  - 'data_length.py': Train 36 models on different quanities of token per observation and different quantities of total tokens in the dataset
  - `compare_dataset.py`: Train 4 models using different pre-existing OCR datasets for comparison to the base synthetic data.
- Each of the scripts will open a lightning 'job' meaning that each sub-experiment will be run in parallel greatly speeding up the process. However after running data will need to be brought back to the lightning stdio using `job_transfer.sh` or 'job_transfer_csv.sh`. Make sure you have modified the script such that the project name and regex are correct. See the scripts for details.

## Training on custom data and models

The studio can be easily used to train on alternative base data and models.

Prepare your custom dataset:
- Ensure your dataset is in a parquet file format.
  The dataset should contain columns for ground truth text ('gt_text') and OCR text ('ocr_text').
- Ensure your test dataset is uploaded as a huggingface dataset dictionary.
- Choose a pre-trained model:
- Select any Unsloth or other Hugging Face model (e.g., Llama, Phi, Mistral).
- Run the training script with custom parameters: `python training_script.py <corruption_type> <corruption_args> <dataset_path> <output_folder> <project_name> --model <model_name> [--data_obs <num_observations>]` See script Docstring/help for detailed instructions.
- The script will train the model and output the test results in the named output folder

## Data

All custom data such as the original synthetic 19th century newspaper articles, training and test sets are available from the data repoository at xxx In addition this work made use of tw high quality OCR dataset the [BLN600](https://github.com/Shef-AIRE/llms_post-ocr_correction), and the [overproof](https://dlp2.pdst.ie/) dataset

**N.B** The character corruption model is trained on archival newspaper text in english using the file `learned_corruption_distribs.json`. 
If you wish to use another language/script or domain, please see the `scrambledtext` library for how to create a new corruption transmission probability dataset.

# Citing this repo

If this repo is useful in your own work please cite
xxx Paper still in progress no citation information yet xxx
