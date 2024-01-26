from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

print("Pytorch Version: {}".format(torch.__version__))

#model = "meta-llama/Llama-2-7b-hf"
#git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
model = "./Llama-2-7b-hf/"

#oserror: we couldn't connect to 'https://huggingface.co/' to load this model and it looks like meta-llama/llama-2-7b-hf is not the path to a directory conaining a config.json file. checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
#huggingface-cli login - fixes OSError
#huggingface-cli whoami


#pip install --user -U transformers (4.30.2) -- resolves ValueError: Tokenizer class LlamaTokenizer does not exist or is not currently imported.
#pip install --user tokenizers==0.13.3 #compatibility when updating HF transformers package - Resolves ImportError: tokenizers>=0.13.3 is required for a normal functioning of this module, but found tokenizers==0.11.6.
#and ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#transformers 4.30.2 requires tokenizers!=0.11.3,<0.14,>=0.11.1, but you have tokenizers 0.14.1 which is incompatible. caused by pip install --user -U tokenizers.
#pip3 freeze > requirements.txt

#tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model)

#tokenizer = AutoTokenizer.from_pretrained("./Llama-2-7b-hf/")
#model = AutoModelForCausalLM.from_pretrained("./Llama-2-7b-hf/")
#
prompt = 'Write a song for me.'
prompt2 = 'What was my previous prompt?'
#
#input_ids = tokenizer(prompt, return_tensors="pt", device_map="auto").input_ids
#outputs = model.generate(input_ids, max_new_tokens=200)
#print(tokenizer.decode(outputs[0]))


llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto" # fixes workload Killed error when using float16 dtype but requires accelerate - pip install --user accelerate
)

#safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge - Download LFS files from repo and place in folder manually - git lfs not installed on Nautilus

#llama_pipeline = pipeline(
#    "text-generation",
#    model=model,
#    torch_dtype=torch.float16
#)

def get_llama_response(prompt: str):
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        sequences[0]['generated_text']: The model's response.
    """

    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=4096
        #max_length=100#,
        )
    #print(sequences)
    #for sequence in sequences:
    #    print("Chatbot:", sequence['generated_text'])
    return sequences[0]['generated_text']

print(get_llama_response(prompt))
print(get_llama_response(prompt2))
