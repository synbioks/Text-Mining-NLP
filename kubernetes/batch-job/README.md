# Running pods as batch jobs
We want to run jupyter notebooks (which preferably are bug-free) as Kubernetes Jobs, in order to streamline experiment running. 

## How it works
The job yaml, once submitted, spins up a job pod as specified. Then it tells the job pod (via the 'command' section) to execute an entrypoint script. The entrypoint script converts a specified jupyter notebook to an ipython script, then executes this script (with the path to a config file as an argument).

## files

test_job.yaml: template for the job creation yaml
- specifies pod configs, and path to the entrypoint script
- possible to pass arguments (e.g. config name for the current job) to the entrypoint script

job_test.sh: template for the job pod entrypoint script
- place this shell script on persistent storage accessible from the job pod, so that the job yaml can run it on pod startup. 
- specify the path to the notebook to run in this script
- once run, this script will:
    - assemble the path to the config file according to the argument passed in by the job yaml
    - convert a specified jupyter notebook to an ipython script, then execute it (passing the path of the config file)

wrapped-job.yaml: wrap a pod spec inside a Job to circumvent cluster restrictions
- don't use this unless you know what you're doing 