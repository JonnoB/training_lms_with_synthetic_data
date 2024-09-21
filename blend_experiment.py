#python training_script.py cer '{"cer": 0.2}'synth_gt/synth200.parquet cer_exp cer_exp

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
project_name = 'blend_exp'
output_dir = os.path.join(project_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

results_dir = os.path.join(output_dir, 'results')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


# do a sweep over the CER values
blend_type = ['blend', 'blend_zero']

for btype in blend_type:
    cmd = f'python training_script.py {btype} "{{\'cer\': 0}}" synth_gt/synth200.parquet {output_dir} {project_name}'
    job_name = f'blend_exp_{btype}'
    job_plugin.run(cmd, machine=Machine.L4, name=job_name)
