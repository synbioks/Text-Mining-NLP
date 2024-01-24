# Attention Visualization

This folder contains the neccessary script to visualize the trained attention weights. The script allows for sampled sentences to be tokenized and visualized using the trained Bert model.


## Requirements:

### System:
```
torch
transformers
bertviz
```


### User-provided:
```
BertLarge (weights/biobert_large_v1.1_pubmed_torch) # can use base model as well
trained model (weights/experiment/checkpoint...) # checkpoint of any trained BioBert model
```

## Getting started

In order to run the notebook in Nautlis, you must first create a job using `viz-gpu.yml`. This job should auto-complete and shut down after two hours, but you should take care to delete the job once you are complete. We need an individual job for this task because we would run out of memory and get an OOMKilled termination.

We have been having issues with Jupyter not automatically starting inside of the job. You need to enter the pod using `kubectl exec podname -it -- /bin/bash`. Once inside the pod, navigate to root and into the relation extraction folder. Run `jupyter notebook --port 9999 --allow-root` to manually start Jupyter and get the link/token to open it.  

You can then port-forward with `kubectl port-forward podname 9999:9999` and access the notebook locally. 


## Example

![attention_viz.png](assets/attention_viz.png)

The above example shows the how the sample sentence "Hello World" is visualized using the Bertviz tool. User can choose the layer number and which attention head to visualize. The mouse hover on [CLS] shows the attentions for this hovered on token. The strength of the attention is seen through the weight of the color.

A more in-depth explanation of how this attention visualization works is written [in this blog post](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1?gi=0205807bbbe7)