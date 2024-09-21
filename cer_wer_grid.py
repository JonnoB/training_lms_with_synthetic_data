#To run a single example use the below as a test
#python training_script.py cer_wer '{"cer": 0.2, "wer": 0.4}' synth_gt/synth200.parquet cer_wer_test cer_wer_test


from lightning_sdk import Studio, Machine
import os

# reference to the current studio
# if you run outside of Lightning, you can pass the Studio name
studio = Studio('finetuneexperiments')
studio.start()
# use the jobs plugin
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

#this is not going to happen 
project_name = 'cer_wer_exp'
output_dir = os.path.join(project_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

results_dir = os.path.join(output_dir, 'results')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


# do a sweep over the CER and WER values
cer_vals = [0.05, 0.1, 0.2, 0.3, 0.4]
wer_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# Generate all combinations of CER and WER values
cer_wer_combinations = [(cer, wer) for cer in cer_vals for wer in wer_vals]

# Iterate over each combination
for cer_for_exp, wer_for_exp in cer_wer_combinations:
    cmd = f'python training_script.py cer_wer "{{\'cer\': {cer_for_exp}, \'wer\': {wer_for_exp}}}" synth_gt/synth200.parquet {output_dir} {project_name}'
    job_name = f'cer_{int(cer_for_exp*100)}_wer_{int(wer_for_exp*100)}_exp'
    job_plugin.run(cmd, machine=Machine.L4, name=job_name)