#To run a single example use the below as a test
#python training_script.py cer_wer "{\"cer\": 10, \"wer\": 20}" synth_gt/synth10.parquet data_length data_length --data_obs 128

from lightning_sdk import Studio, Machine
import os
import json

# reference to the current studio
# if you run outside of Lightning, you can pass the Studio name
studio = Studio('finetuneexperiments')
studio.start()
# use the jobs plugin
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

#this is not going to happen 
project_name = 'data_length_exp'
output_dir = os.path.join(project_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

results_dir = os.path.join(output_dir, 'results')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


# Token length to corresponding data observations based on the table
data_obs_mapping = {
    200: [8192, 4096, 2048, 1024, 512, 256, 128],
    100: [16384, 8192, 4096, 2048, 1024, 512, 256],
    50:  [32768, 16384, 8192, 4096, 2048, 1024, 512],
    25:  [65536, 32768, 16384, 8192, 4096, 2048, 1024],
    10:  [163840, 81920, 40960, 20480, 10240, 5120, 2560]
}

# Token lengths and total tokens
token_length_list = [200, 100, 50, 25, 10]
total_tokens_list = [1638400, 819200, 409600, 204800, 102400, 51200, 25600]

# Iterate over each combination from the table
for token_length in token_length_list:
    for i, total_tokens in enumerate(total_tokens_list):
        data_obs = data_obs_mapping[token_length][i]  # Get the data_obs value based on the table
        json_args = json.dumps({"cer": 0.1, "wer": 0.2})
        cmd = f'python training_script.py cer_wer \'{json_args}\' synth_gt/synth{token_length}.parquet {output_dir} {project_name} --data_obs {data_obs}'
        job_name = f'data2_obs_{int(data_obs)}_token_length_{int(token_length)}_exp'
        job_plugin.run(cmd, machine=Machine.L4, name=job_name)