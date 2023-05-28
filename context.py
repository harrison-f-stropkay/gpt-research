import csv

from transformers import AutoModel, AutoTokenizer


def context_vectors(responses_name, model_name):
    with open(f"dataset/{responses_name}.responses", "r") as file:
        responses = file.readlines()

    # load tokenizer and model
    # set use_fast to false because "Note: Make sure to pass use_fast=False when loading OPT’s tokenizer with 
    # AutoTokenizer to get the correct tokenizer." @ https://huggingface.co/docs/transformers/model_doc/opt#transformers.OPTModel
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False) 
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # generate context vectors
    context_vectors = []
    for response in responses:
        encoded_input = tokenizer.encode_plus(
            response,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=512, 
            return_tensors='pt'
        )    
        input_ids = encoded_input['input_ids']
        output = model(input_ids)
        #context_vectors.append(output.last_hidden_state[0, 0, :].tolist())
        context_vectors.append(output.pooler_output.detach().squeeze().numpy().tolist())
    
    # save vectors as CSV
    model_name = model_name.replace('/', '-')
    with open(f"dataset/{responses_name}.{model_name}_POOLED", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(context_vectors)

# driver    
for responses in ['10-prompts_human-GPT-Bard', '10-prompts_human-temps']:
    for model_name in ['bert-base-uncased', 'roberta-base']: #, 'facebook/opt-350m']:
        context_vectors(responses, model_name)
