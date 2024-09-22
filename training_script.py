
"""
OCR Text Correction Training and Evaluation Script

This script trains a language model to correct OCR-corrupted text and evaluates its performance.

Usage:
    python training_script.py <corruption_type> <corruption_args> <dataset> <output> <project_name> [--model MODEL] [--data_obs DATA_OBS]

Arguments:
    corruption_type (str): Type of corruption to apply (e.g., 'cer', 'cer_wer', 'blend', 'blend_zero', 'compare')
    corruption_args (str): Arguments for the corruption as a dictionary-like string (e.g., '{"cer":0.2}')
    dataset (str): Path to the dataset stored as a parquet file
    output (str): Path to the output folder
    project_name (str): Name of the project or experiment
    --model (str): Model to be used for training (default: 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit')
    --data_obs (int): Number of observations to use for training (default: all)

The script performs the following steps:
1. Loads and corrupts the dataset based on the specified corruption type and arguments
2. Prepares the dataset for training
3. Loads and configures the language model
4. Trains the model using the SFTTrainer
5. Evaluates the model on a test set
6. Computes and saves performance metrics (CER, WER, ERP)

Results are saved in the specified output folder and logged to Weights & Biases (wandb).

Requirements:
- Python 3.x
- Required libraries: pandas, numpy, torch, transformers, datasets, trl, wandb, evaluate, unsloth

Note: Make sure to have wandb credentials set up before running the script.
"""

from scrambledtext import (ProbabilityDistributions, CorruptionEngine, modify_and_renormalize_probs)

import pandas as pd
import numpy as np
import argparse
import ast
import os 
from datasets import Dataset, DatasetDict, load_from_disk
from lm_support_functions import training_prompt, compute_metric, infer_on_test_set, cleaning_prompt_formatter, infer_on_test_set_split
from unsloth import FastLanguageModel   
import torch
from trl import SFTTrainer#, SFTConfig #the confilarge_gold_datag does not exist in the trl required by unsloth
from transformers import TrainingArguments
import wandb
wandb.login() # you need to have your wandb credentials stored
import time 
import evaluate


# Initialize the parser
parser = argparse.ArgumentParser(description='Process corruption type and arguments.')

# Add arguments
parser.add_argument('corruption_type', type=str, help='Type of corruption to apply')
parser.add_argument('corruption_args', type=str, help='Arguments for the corruption as a dictionary-like string')
parser.add_argument('dataset', type=str, help='path to the dataset stored as a parquet file')
parser.add_argument('output', type=str, help='path to the output folder')
parser.add_argument('project_name', type=str, help='name of the project or experiment')
parser.add_argument(
    '--model', 
    type=str, 
    help='Model which will be used for training, defaults to Unlsoths Llama 3.1 instruct', 
    default= 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'
)
parser.add_argument(
    '--data_obs', 
    type=int, 
    help='number of observations to use, max is total in dataset, default is all', 
    default= None
)
parser.add_argument(
    '--example', 
    type=bool, 
    help='Use the example data as the test set, contains 30 obs only', 
    default= False
)

# Parse the arguments
args = parser.parse_args()

project_name = args.project_name
# Access the arguments
corruption_type = args.corruption_type

model_name = args.model

# Convert corruption_args to a dictionary
corruption_args = ast.literal_eval(args.corruption_args)

dataset_path = args.dataset

#all experiments in the project will be saved here
output_path = args.output

example_mode = args.example

results_folder = os.path.join(output_path, 'results')
#create the output folder if it doesn't already exist
if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(results_folder)

print(f"Results path at {os.path.join(output_path, 'results')}")

#get the dataset file name which will be used as part of the output filename for the results
file_name = os.path.splitext(os.path.basename(dataset_path))[0]

##
## Testing only three being used
##
synth_data = pd.read_parquet(dataset_path)#.sample(3)

#number of training observations to use in the training
training_obs = args.data_obs

#For now use a fixed corruption
print('Create corruption tables')

corruption_probs = ProbabilityDistributions()
#load the premade corruption distribution
corruption_probs = corruption_probs.load_from_json('learned_corruption_distribs.json')

#corruption function for the blended corruption experiments
def get_random_corruption_engine(corruption_samples, corruption_probs):
    # Randomly select a row from corruption_samples
    random_sample = corruption_samples.sample(n=1).iloc[0]
    
    # Create and return a CorruptionEngine instance with the randomly selected WER and CER
    return CorruptionEngine(
        corruption_probs.conditional, 
        corruption_probs.substitutions, 
        corruption_probs.insertions,
        target_cer=random_sample['cer'], 
        target_wer=random_sample['wer']
    )

#
# Corrupting the text
# The below if statements allow the text to be corrupted dependent on the arguments provided to the script
# The idea is to make it easy to run different experiments with the same script
#
print('creating corrupted text')
if corruption_type == 'cer':

    corruption_function = CorruptionEngine(corruption_probs.conditional, corruption_probs.substitutions, corruption_probs.insertions, 
                                            target_cer= corruption_args['cer'], target_wer=1)

    synth_data['ocr_text'], synth_data['wer'], synth_data['cer'], synth_data['effect_cer'] = zip( 
        *synth_data['gt_text'].apply(lambda text:corruption_function.corrupt_text(text))
        )

    experiment_name = f"""{file_name}_cer_{int(corruption_args['cer']*100)}"""

elif corruption_type =='cer_wer':

    corruption_function = CorruptionEngine(corruption_probs.conditional, corruption_probs.substitutions, corruption_probs.insertions, 
                                            target_cer= corruption_args['cer'], target_wer=corruption_args['wer'])

    synth_data['ocr_text'], synth_data['wer'], synth_data['cer'], synth_data['effect_cer'] = zip( 
        *synth_data['gt_text'].apply(lambda text:corruption_function.corrupt_text(text))
        )   

    experiment_name = f"""{file_name}_cer_{int(corruption_args['cer']*100)}_wer_{int(corruption_args['wer']*100)}"""

elif corruption_type == 'blend':
    print("creating a  mixed corruption dataset this may be slow")
    
    corruption_samples = pd.read_csv('corruption_samples.csv')
    # Apply the function to each row in synth_data
    synth_data['ocr_text'], synth_data['wer'], synth_data['cer'], synth_data['effect_cer'] = zip(
        *synth_data.apply(
            lambda row: get_random_corruption_engine(corruption_samples, corruption_probs).corrupt_text(row['gt_text']),
            axis=1
        )
    )

    print("corruption complete!")

    experiment_name = f"""{file_name}_blend_cer_999_wer_999"""

elif corruption_type == 'blend_zero':
    print("creating a  mixed corruption dataset this may be slow")
    
    corruption_samples = pd.read_csv('corruption_samples_zero.csv')
    # Apply the function to each row in synth_data
    synth_data['ocr_text'], synth_data['wer'], synth_data['cer'], synth_data['effect_cer'] = zip(
        *synth_data.apply(
            lambda row: get_random_corruption_engine(corruption_samples, corruption_probs).corrupt_text(row['gt_text']),
            axis=1
        )
    )

    print("corruption complete!")

    experiment_name = f"""{file_name}_blendzero_cer_999_wer_999"""

elif corruption_type =='compare':
    print('Comparing existing datasets no corruption required')

    experiment_name = f"""{file_name}_compare_cer_999_wer_999"""

else:
    print('No correct argument entered hot crash incoming')

#So that the file names have enough information to actually be understandable
if project_name == "data_length":

    experiment_name = experiment_name+f"_obs_{training_obs}"


results_path = os.path.join(output_path, 'results', experiment_name+'.csv')


hf_dataset = Dataset.from_pandas(synth_data)

if corruption_type =='compare':
    dataset_dict = DatasetDict({   
        'train': hf_dataset
    })
else:
# Split the dataset based on the 'data_type' column into training, validation, and test sets
    dataset_dict = DatasetDict({
        'train': hf_dataset.filter(lambda example: example['data_type'] == 'training'),
        'validation': hf_dataset.filter(lambda example: example['data_type'] == 'validation'),
        'test': hf_dataset.filter(lambda example: example['data_type'] == 'test')
    })

#nopte doing the subset here is very inefficient as all data is corrupted first!
if training_obs is not None:
    print(f'Reducing training data to {training_obs} observations')
    dataset_dict['train'] = dataset_dict['train'].shuffle(seed=1880).select(range(training_obs))

else:
    print('All training obersvations used')

#clean up uneccessary dataframes
del hf_dataset
del synth_data



##
## Load model and set parameters
##

max_seq_length = 1024#512

#We take the instruct model 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  model_name,#'unsloth/llama-3-8b-bnb-4bit',#'unsloth/Phi-3-mini-4k-instruct-bnb-4bit',#"unsloth/mistral-7b-v0.3-bnb-4bit", 
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16, #None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True
)

#Sometimes you may get an "offload to cpu" type error, this can happen if you stop/crash part way through training, #
# check the VRAM on the GPU is not full of the old model


model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      ],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True
)


dataset_dict = dataset_dict.map(lambda x:training_prompt(x, 'ocr_text', 'gt_text', tokenizer), batched=False)


#### should be 64 and 1 for phi!
args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        warmup_steps = 50,
        max_steps = -1, #should be -1
        num_train_epochs=1,
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 16,
        eval_steps=128,
        evaluation_strategy = 'no', #For this version of trl need to use this setting. This changes for more recent versions I think
        save_strategy = 'epoch',
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        remove_unused_columns=True, 
        seed = 3407,
        output_dir = os.path.join(output_path, experiment_name),
        report_to="wandb"
    )



trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset_dict["train"], #NO EVAL FOR THIS SET OF EXPERIMENTS
    dataset_text_field="full_prompt",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False,#False if training_obs is None else True,  # Set packing based on training_obs
    args = args
)


##
## Begin Training
##


run = wandb.init(
    # set the wandb project where this run will be logged
    project=project_name
)

# Get the W&B run name
run_name = run.name

# Training
start = time.time()
trainer.train()
wandb.finish()
print(f"Training complete: {round((time.time() - start)/60)} minutes")
##
## Post training 
##


##
## Load ncse dataset
##
if example_mode:
    print("Using example dataset corrupted with CER = 0.17 WER = 0.55 and 5 observations")
    data = load_from_disk('example_hf_dataset')    
else:
    data = load_from_disk('ncse_hf_dataset')

#data = data.map(lambda x: cleaning_prompt_formatter(x, 'ocr_text', tokenizer), batched=False)

##
## infer over test set
##

#switch to inference mode
FastLanguageModel.for_inference(model)

temp  = infer_on_test_set(data, model, tokenizer)

## These multiple saves are to make sure there isn't an error losing everything if for example the LLM makes an error
temp.to_csv(results_path)

#temp['inference'] = 'single'
#temp['clocrc_text'] =  temp['clocrc_text'].apply(lambda x: x.split('###Recovery###')[1].split('###')[0])
#check if the 
#temp2 = infer_on_test_set_split(data, model, tokenizer, device="cuda", n=500, m=100)
#temp['inference'] = 'split'

#temp = pd.concat([temp, temp2], ignore_index = 1)
#temp.to_csv(results_path)



##
## add wer and cer
##

metric_cer = evaluate.load("cer")
metric_wer = evaluate.load("wer")



# Apply the function to each row for 'output' and 'raw_text' columns
temp['type'] = model_name
temp['tokens'] = temp['clocrc_text'].apply(lambda x: len(tokenizer.encode(x)))
temp['cer'] = temp.apply(compute_metric, axis=1, metric =metric_cer, prediction_col='clocrc_text', reference_col='gt_text')
temp['wer'] = temp.apply(compute_metric, axis=1, metric =metric_wer, prediction_col='clocrc_text', reference_col='gt_text')


# Compute the ERP (Error Reduction Percentage)
temp['erp_cer'] = (temp['cer_orig'] - temp['cer']) / temp['cer_orig']
temp['erp_wer'] = (temp['wer_orig'] - temp['wer']) / temp['wer_orig']

temp.to_csv(results_path)


print(temp[['cer', 'wer']].median())
print(f"Train and test completed. results saved to {results_path} terminating script successfully")