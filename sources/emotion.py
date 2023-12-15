from datasets import load_dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from collections import Counter
import pickle, re, os, time, random
import json
import argparse

parser = argparse.ArgumentParser()

import warnings

warnings.filterwarnings('ignore')


PROMPT_TEMPLATE_1 = f"""
Classify the following sentence into six categories: [0: Sadness; 1: Joy, 2: Love; 3: Anger; 4: Fear, 5: Surprise].
Provide answer in a structured format WITHOUT additional comments, I just want the numerical label for each sentence.
"""


def main(model, tokenizer, prompts, training_data, args):
    path = os.path.join(args.save_path, str(int(args.current_time)))
    # Create new dir
    if not os.path.exists(path):
        os.makedirs(path)
    # Save configrations
    with open(path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Append to the saved path
    data = []
    for index, prompt in enumerate(prompts):
        preds, entropies = uncertainty_calculation(model, tokenizer, prompt, training_data,
                                                   args.decoding_strategy, args.num_demos,
                                                   args.num_demos_per_class, args.sampling_strategy, 
                                                   args.iter_demos)
        AU, EU = token_uncertainty_calculation_new(preds, entropies)
        print("AU: {}\tEU: {}\tAU_new: {}\tEU_new: {}".format(AU, EU, AU_new, EU_new))
        pred = answer_extraction(preds)
        try:
            pred = Counter(pred).most_common()[0][0]
        except:
            pred = None
        save_res = {"Question": prompt, "Label": labels[index], "Predicted_Label": pred, "AU": AU, "EU_new": EU}
        
        data.append(save_res)
    return data


def post_processing(data, save_path, epochtime, model):
    if not data:
        data = []
        with open(save_path + '{}/{}_sentiment.pkl'.format(int(epochtime), model), 'rb') as fr:
            try:
                while True:
                    data.append(pickle.load(fr))
            except EOFError:
                pass
    # Create Dataframe
    data = pd.DataFrame(data)
    AU, EU, AU_new, EU_new, preds = [], [], [], [], []
    answers, entropies = list(data['Predicted_Label']), list(data['Entropies'])
    for i in range(len(answers)):
        ale_new, epi_new = token_uncertainty_calculation_new(answers[i], entropies[i])
        pred = answer_extraction(answers[i])
        preds.append(Counter(pred).most_common()[0][0])

        AU_new.append(ale_new)
        EU_new.append(epi_new)

    data['AU_new'] = AU_new
    data['EU_new'] = EU_new
    data['Preds'] = preds
    data = data.drop(columns=['Predicted_Label', 'Entropies'])
    data.to_json('./LLM_UQ/results/' + '{}/{}_sentiment_processed.json'.format(int(epochtime), model),
                 orient="records")


if __name__ == '__main__':
    parser.add_argument('--save_path', type=str, default='./LLM_UQ/results/')
    parser.add_argument('--model', type=str, default='7b')
    parser.add_argument('--num_demos', type=int, default=5)
    parser.add_argument('--num_demos_per_class', type=int, default=1)
    parser.add_argument('--sampling_strategy', choices=['random', 'class'], default='random')
    parser.add_argument('--decoding_strategy', choices=['beam_search', 'constractive', 'greedy', 'top_p'],
                        default='beam_search')
    parser.add_argument('--iter_demos', type=int, default=4)
    parser.add_argument('--load8bits', default=False, help='load model with 8 bits')
    parser.add_argument('--current_time', type=str, default=time.time())
    parser.add_argument('--resume_from', type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading Model
    model_path = '/mnt/dsss_data/cling/llama-2-{}-chat-hf'.format(args.model)
    if args.load8bits:
        model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype='auto', load_in_8bit=True)
    else:
        model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Done! Loaded Model: {}".format(args.model))

    # Loading Data
    # training_data, test_data = get_data()
    
    _, test_data = get_data()
    training_data, _ = get_data(dataset_name='glue')
    
    prompts = [i['text'] for i in test_data]
    labels = [i['label'] for i in test_data]
    print("Done! Loaded Data")

    if not args.resume_from:
        data = main(model, tokenizer, prompts, training_data, args)
    else:
        data = main(model, tokenizer, prompts[args.resume_from + 1:], training_data, args)
    
    data = pd.DataFrame(data)
    data.to_json('/mnt/dsss_data/cling/LLM_UQ/results/' + '{}/{}_financial_{}.json'.format(int(args.current_time), 
                args.model, args.sampling_strategy), orient="records")
    # post_processing(data, args.save_path, args.current_time, args.model)
