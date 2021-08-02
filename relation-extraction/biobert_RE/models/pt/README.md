Annotations and files are stored in Brat server.

Preprocessing and postprocessingfiles are present inside the utils folder.

acs-re-preproess.py       : Takes the ann & txt file as input and creates a re_input.tsv
                            Used before runnding the model.
acs-re-postprocess.py     : Takes re_output & the ann file as input and creates a ann file with the predicted relations by the model.
			    Used after running the model. 
acs-re-postprocess-na.py  : Same as the previous file but this also prints the NoRelation when creating the final ann file.


re-inter-annoatation-na   : Code to evaluate the inter-annotation results.   




