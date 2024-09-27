<a target="_blank" href="https://lightning.ai/ucabbou/studios/scrambledtext">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a> ![made with unsloth](https://github.com/user-attachments/assets/3d1247b3-9090-455f-b85e-1377c6d1c04b)

# Training LMs with synthetic data

This repo contains code for training Language Models (LMs) to perform Context Leveraging OCR Correction (CLOCR-C). 
The code is related to the [scrambledtext](https://github.com/JonnoB/scrambledtext) repo, as both repos were used in the paper "Scrambled text: training Language Models to correct OCR errors using synthetic data".
This repo is designed to be run on the lightning.ai platform and can be opened as a lightning studio using the button above. The [unsloth.ai](https://github.com/unslothai/unsloth) implementation of Llama3 is used to reduce cost.
Other models can be used; however, using Unsloth reduces RAM and increases training time; see the Unsloth documentation for details.

# Related repos
This repo is part of the larger ScrambledText project:

- [scrambledtext](https://github.com/JonnoB/scrambledtext): Library used to create the synthetic data.
- [scrambledtext_analysis](https://github.com/JonnoB/scrambledtext_analysis): Contains code for training the Language Models.


# Key files in the repo

- `training_script.py`. This Python script can be executed from the command line using its arguments to train an LM to perform CLOCR-C using training data and the base LM of your choice
- `learned_corruption_distribs.json`: The conditional probability distributions used to corrupt data
- `example_dataset_df.parquet`: parquet file containing a small amount of training data to allow running of the scripts
- `/example_hf_dataset`: a folder which can be loaded as a Huggingface dataset. Contains 5 examples and acts as an example test set

# Using this repo

There are several ways of using this repo, from basic use to see how the system works, reproducing the results in the paper, to training your own custom model.

## Before you start

- Make sure you have initialised your weights and biases account in the studio
- Have a hugging face token ready if you plan to use other models than those provided by Unsloth
- The models can be trained on a 24Gb NVIDIA L4 GPU. However, if using the job scripts discussed in the 'Reproducing the work in the paper section,' GPU is handled automatically, and the studio can be started with the free CPU.

## Basic use

- Open as a studio in lightning
- Switch from the free CPU to an L4 GPU (Can use more expensive if you want)
- run `python training_script.py cer_wer "{'cer': 0.10, 'wer': 0.20}" example_data_df.parquet example example --example True` to run the example script
- The output will be saved to the folder `example`

This example run will take about 2 minutes to run you can change the parameters as you wish. However, changing --example to False will run the full test set which takes about 40 minutes so should probably be avoided unless you are sure.

## Reproducing the work in the paper

- Open as a studio in lightning
- Upload the training and test data from the data repository; the necessary files are as follows
    - `synth_gt.zip`: The synthetic training data
    - `ncse_hf_dataset: A Huggingface dataset of the NCSE test data
- Run the following scripts
  - `cer_Experiment_grid.py`:Train 9 models on different CER values with a uniform distribution.
  - `cer_wer_grid.py`: Train 35 models on different CER WER pairs.
  - `blend_experiment.py` Train 2 models on mixtures of CER WER pairs.
  - 'data_length.py': Train 36 models on different quantities of tokens per observation and different amounts of total tokens in the dataset.
  - `compare_dataset.py`: Train 4 models using different pre-existing OCR datasets for comparison to the base synthetic data
- Each script will open a lightning 'job', meaning that each sub-experiment will be run in parallel, significantly speeding up the process. However, after running, data must be returned to the lightning studio using `job_transfer.sh` or 'job_transfer_csv.sh`. Ensure you have modified the script so the project name and regex are correct. See the scripts for details.

## Training on custom data and models

The studio can easily train on alternative base data and models.

Prepare your custom dataset:
- Ensure your dataset is in a parquet file format.
 The dataset should contain columns for ground truth text ('gt_text') and OCR text ('ocr_text')
- Ensure your test dataset is uploaded as a huggingface dataset dictionary.
- Select any Unsloth or other Hugging Face model (e.g., Llama, Phi, Mistral).
- Run the training script with custom parameters: `python training_script.py <corruption_type> <corruption_args> <dataset_path> <output_folder> <project_name> --model <model_name> [--data_obs <num_observations>]` See script Docstring/help for detailed instructions.
- The script will train the model and output the test results in the named output folder.

# Data

All custom data, such as the original synthetic 19th-century newspaper articles, training and test sets, are available from the data repository at [Scrambled text Datasets from the paper](https://doi.org/10.5522/04/27108334.v1). In addition, this work made use of two high-quality OCR datasets, the [BLN600](https://github.com/Shef-AIRE/llms_post-ocr_correction) and the [overproof](https://dlp2.pdst.ie/) dataset (Overproof website has issues).

**NB** The character corruption model is trained on archival newspaper text in English using the file `learned_corruption_distribs.json`. 
If you wish to use another language/script or domain, please see the `scrambledtext` library for information on creating a new corruption transmission probability dataset.

# Citing this repo

If this repo is helpful in your own work, please cite
xxx paper still in progress. No citation information yet xxx
