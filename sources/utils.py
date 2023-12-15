import pandas as pd
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from collections import Counter
import pickle, re, os, time, random
import json
import argparse


def get_data(dataset_name='dair-ai/emotion'):
    if dataset_name=='dair-ai/emotion':
        return (load_dataset(dataset_name)['train'], load_dataset(dataset_name)['test'])
    elif dataset_name=='glue':
        temp_dataset = load_dataset(dataset_name, 'cola')
        return (temp_dataset['train'], temp_dataset['validation'])
    elif dataset_name=='financial_phrasebank':
        temp_dataset = load_dataset(dataset_name, 'sentences_allagree')['train']
        temp_dataset = temp_dataset.train_test_split(test_size=0.5, shuffle=False) # Train/Test Ratio = 1132/1132
        return (temp_dataset['train'], temp_dataset['test'])
    elif dataset_name=='ag_news':
        temp_dataset = load_dataset(dataset_name)['test']
        temp_dataset = temp_dataset.train_test_split(test_size=0.1, shuffle=False) # Train/Test Ratio = 1132/1132
        return (temp_dataset['train'], temp_dataset['test'])

def entropy_calculation(generate_scores):
    logits = torch.stack(generate_scores, dim=1)
    return logits


def create_demonstrations(dataset, k=6, k_per_class=1, sampling_strategy='random'):
    if sampling_strategy == 'random':
        examples = []
        for _ in range(k):
            random_index = int(np.random.random() * len(dataset))
            examples.append(dataset[random_index])
    elif sampling_strategy == 'class':
        label_to_texts = {}
        for text, label in zip(dataset['sentence'], dataset['label']):
            if label not in label_to_texts:
                label_to_texts[label] = []
            label_to_texts[label].append(text)
        # Randomly select one text from each label
        examples = []
        # Randomly select
        for label, texts in label_to_texts.items():
            if len(texts) < k_per_class:
                raise ValueError(f"Label {label} has fewer than {k_per_class} texts.")
            sampled_texts = random.sample(texts, k_per_class)
            for text in sampled_texts:
                examples.append({
                    'sentence': text,
                    'label': label
                })
    # Construct Prompt from sampled demonstrations
    '''
    prompts = PROMPT_TEMPLATE
    for i, x in enumerate(examples):
        temp_msg = """### Example {}\nSentence: "{}"\nCategory: {}\n\n""".format(i, x['sentence'], x['label'])
        prompts += temp_msg
    '''
    prompts = PROMPT_TEMPLATE_1
    for i, x in enumerate(examples):
        temp_msg = """Sentence: "{}" Category: {}\n""".format(x['sentence'], x['label'])
        prompts = temp_msg + prompts
    return prompts


def create_prompt(sentence: str, demonstrations: str) -> str:
    # return demonstrations + "### Test\nWhat is the sentiment of the following sentence? Choose from [0: negative; 1: neutral, 2: positive].\n" + "Sentence: \"{}\"\nCategory:".format(sentence)
    return demonstrations + "Sentence: \"{}\" Category:".format(sentence)


def answer_generation(model, tokenizer, prompt, decoding_method=None):
    if decoding_method == "beam_search":
        outputs = model.generate(**prompt, return_dict_in_generate=True, output_scores=True, max_new_tokens=20,
                                 num_beams=10, num_return_sequences=10, early_stopping=True)
    elif decoding_method == "contrastive":
        outputs = model.generate(**prompt, return_dict_in_generate=True, output_scores=True, max_new_tokens=8,
                                 penalty_alpha=0.6)
    elif decoding_method == "greedy":
        outputs = model.generate(**prompt, return_dict_in_generate=True, output_scores=True, max_new_tokens=8)
    elif decoding_method == "top_p":
        outputs = model.generate(**prompt, return_dict_in_generate=True, output_scores=True, max_new_tokens=8, top_k=40,
                                 top_p=0.8, temperature=0.6)
    generate_scores, generate_ids = outputs.scores, outputs.sequences
    # Calculate average entropy of the batch
    entropy = entropy_calculation(generate_scores)
    # Obtain the predicted answer with majority vote
    temp_answer = []
    for i in range(len(generate_ids)):
        res = tokenizer.batch_decode(generate_ids[i][prompt.input_ids.shape[1]:], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        # res = re.findall(r'\d+\.\d+|\d+', ''.join(res))
        # if res:
        #     temp_answer.append(int(res[0]))
        temp_answer.append(res)
    # if temp_answer:
    #     temp_answer = (Counter(temp_answer).most_common())[0][0]
    # else:
    #     temp_answer = None
    return temp_answer, entropy


def most_frequent_element(lst):
    # Filter out None values
    filtered_lst = [x for x in lst if x is not None]
    if not filtered_lst:
        return None

    counter = Counter(filtered_lst)
    return counter.most_common(1)[0][0]


def uncertainty_calculation(model, tokenizer, prompt, training_data, decoding_strategies, demo_num=5, demo_per_class=1,
                            sampling_strategy='class', demo_iter=4):
    answers, entropies = [], []
    for _ in range(demo_iter):
        demonstrations = create_demonstrations(training_data, demo_num, demo_per_class, sampling_strategy)
        test_prompt = tokenizer(create_prompt(prompt, demonstrations), return_tensors="pt")
        temp_answers, temp_entropies = [], []
        for strategy in [decoding_strategies]:
            output = answer_generation(model, tokenizer, test_prompt, decoding_method=strategy)
            temp_answers.append(output[0])
            temp_entropies.append(output[1])
        answers.append(temp_answers)
        entropies.append(temp_entropies)
    # answers = np.array(answers).flatten('C').tolist()
    # answers = Counter(answers).most_common()[0][0]
    return answers, entropies


def find_first_number_index(lst):
    for idx, item in enumerate(lst):
        if item.isdigit():
            return idx
    return None


def token_uncertainty_calculation(preds, entropies):
    total_token_logits = []
    total_answers = []
    for i in range(len(preds)):
        _temp_token_logits = []
        _answer_token = []
        for j in range(len(preds[i][0])):
            token_idx = find_first_number_index(preds[i][0][j])
            if token_idx:
                _temp_token_logits.append(entropies[i][0][j][token_idx])
                _answer_token.append(preds[i][0][j][token_idx])
            else:
                _temp_token_logits.append(entropies[i][0][j][0])
                _answer_token.append(None)
        total_token_logits.append(torch.stack(_temp_token_logits))
        total_answers.append(np.stack(_answer_token))
    total_answers = np.stack(total_answers)
    total_token_logits = torch.stack(total_token_logits, dim=0)
    prob_demos = []
    for i in range(len(total_answers)):
        prob_demo = []
        for j in range(len(total_answers[0])):
            if total_answers[i][j]:
                prob_demo.append(total_token_logits[i][j])
            prob_demos.append(prob_demo)
    AU = [torch.mean(torch.stack(i), dim=0).softmax(dim=0) for i in prob_demos]
    AU = np.mean([-torch.sum(i * torch.log(i)).item() for i in AU])
    TU = [torch.mean(torch.stack(i), dim=0) for i in prob_demos]
    TU = torch.mean(torch.stack(TU), dim=0).softmax(dim=0)
    TU = -torch.sum(TU * torch.log(TU)).item()
    return AU, TU - AU


def token_uncertainty_calculation_new(preds, entropies, num_classes=2):
    total_token_logits = []
    total_answers = []
    for i in range(len(preds)):
        _temp_token_logits = []
        _answer_token = []
        for j in range(len(preds[i][0])):
            token_idx = find_first_number_index(preds[i][0][j])
            if token_idx:
                _temp_token_logits.append(entropies[i][0][j][token_idx])
                _answer_token.append(preds[i][0][j][token_idx])
            else:
                _temp_token_logits.append(entropies[i][0][j][0])
                _answer_token.append(None)
        total_token_logits.append(torch.stack(_temp_token_logits))
        total_answers.append(np.stack(_answer_token))
    total_token_logits = torch.stack(total_token_logits, dim=0)
    # Calculate Total Uncertainty
    total = total_token_logits.softmax(dim=-1).max(dim=-1).values.cpu().numpy()
    total_answers = np.stack(total_answers)
    # Calculate probabilities
    prob_demos = []
    for i in range(len(total_answers)):
        prob_demo = np.zeros(num_classes)
        for j in range(len(total_answers[0])):
            if total_answers[i][j] and int(total_answers[i][j]) < num_classes:
                prob_demo[int(total_answers[i][j])] += total[i][j]
        prob_demos.append(torch.from_numpy(prob_demo))
    prob_demos = torch.stack(prob_demos)
    # Total Uncertainty
    # TU = torch.sum(prob_demos, dim=0).softmax(dim=0)
    # TU = -torch.sum(TU * torch.log(TU)).item()
    TU = (torch.sum(prob_demos, dim=0) + 10 ** -7)
    TU = TU / torch.sum(TU)
    TU = -torch.sum(TU * torch.log(TU)).item()
    # Aleatoric Uncertainty
    AU = prob_demos + 10 ** -7
    uncertainty_list_AU = []
    for i in range(len(AU)):
        temp = AU[i] / torch.sum(AU[i])
        uncertainty_list_AU.append(-torch.sum(temp * torch.log(temp)).item())
    AU = np.mean(uncertainty_list_AU)
    return AU, TU - AU


def answer_extraction(preds):
    answers = []
    for i in range(len(preds)):
        for j in range(len(preds[i][0])):
            res = re.findall(r'\d+\.\d+|\d+', ''.join(preds[i][0][j]))
            if res:
                answers.append(int(float(res[0])))
    return answers