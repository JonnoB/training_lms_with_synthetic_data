#To run a single example use the below as a test
#python training_script.py cer_wer '{"cer": 0.2, "wer": 0.4}' synth_gt/synth200.parquet cer_wer_test cer_wer_test


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
project_name = 'compare_datasets_exp'
output_dir = os.path.join(project_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

results_dir = os.path.join(output_dir, 'results')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


json_args = json.dumps({"cer": 0, "wer": 0})
datasets_list = ['BLN600', 'CA', 'SMH', 'overproof']

# Iterate over each combination
for data_set in datasets_list:
    cmd = f'python training_script.py compare  \'{json_args}\'  compare_datasets/{data_set}.parquet {output_dir} {project_name}'
    job_name = f'dataset_{data_set}_exp'
    job_plugin.run(cmd, machine=Machine.L4, name=job_name)
