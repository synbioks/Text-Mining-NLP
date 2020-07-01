### These scripts are at /sbksvol/gaurav/BiLSTM-CRF/
<br>
To run:
source /sbksvol/gaurav/tfenv/bin/activate
<br>
python3 get_all_embeddings.py
<br>
python3 top-model.py
<br>

<br>
### TO RUN AS A BATCH JOB:
1. Move the script biobert_job.sh to the location indicated in the script: /sbksvol/gaurav/
2. Run biobert-job.yaml :
<br>
kubectl create -f biobert-job.yaml
<br>
3. See logs to monitor progress:
<br>
kubectl logs -f <name-of-job-pod>
<br>
Follow instructions on nautilus home page for other useful commands.
