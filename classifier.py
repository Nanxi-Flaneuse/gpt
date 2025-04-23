import json
import openai
from openai import OpenAI
import os
import pandas as pd
import sys
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_random_exponential
import multiprocessing
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from seqeval.metrics.sequence_labeling import get_entities
import torch
from cjkfuzz import fuzz 
from cjkfuzz import process
import collections

# add api keys here


def is_answer_in_valid_form(answer):
    """Check if the GPT's answer is in the expected format.

    This is the format we want:
        Readability: 1

    Note: 4.5 will be extracted as 4.
    """
    answer = answer.strip("\n").strip()
    if re.search("^[0-1]+$", answer):
        return True
    return False

# running single query
def run_gpt4_query(filled_prompt, lang, model):
    print('running gpt -----------------------')
    if lang == "en":
        sys = "Your job is a computational social scientist interested in the names of Chinese restaurants in the U.S. "
    else:
        sys = "您的工作是一名计算社会科学家，对美国的中餐馆名称感兴趣。"
    # print(sys)
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": sys},
        {"role": "user", "content": filled_prompt},
    ],
    temperature=0)
    print('getting response')
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))

### this function prompts the GPT model to generate a response when given a prompt and checks if the response is correct
def generate_categorization(restaurant_en, restaurant_cn, prompt_file, language, model,category = None):
    """Explains a given text for a specific audience.

    Args:
        text (str): The input text to be explained.
        prompt_file (str): The file path to the prompt file.

    Returns:
        str: The explanation of the input text.

    """
    # Read prompt template
    prompt_template = open(prompt_file).read()
    print('prompt read')
    if language == "en":
        prompt = prompt_template.replace("{EN_NAME}", restaurant_en)
       
    else:
        prompt = prompt_template.replace("{CN_NAME}", restaurant_cn)
    # prompt = prompt.replace("{CN_NAME}", restaurant_cn)
    prompt = prompt.strip("\n").strip()
    prompt = prompt + "\n"
    print(prompt)
    print('prompt generated')
    ## Nanxi testing
    # return True
    # comment out below for prompt generation testing
    while True:
        # print('here---------------')
        response = run_gpt4_query(prompt, lang = language, model = model)
        print('response generated ----------------------')
        response = response.choices[0].message.content.strip("\n")
        print(response)
        # return response
        if is_answer_in_valid_form(response):
            # print(response)
            return response
        else:
            print("====>>> Answer not right, re-submitting request...")
            print(response)

# preparing the parameters and obtain GPT response for the entire file
def main():

    # question_type = "prompt_en_Positivity_json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../data_cleaning/output/validation_en.csv") #model_training/
    parser.add_argument("--prompt_file_path", type=str)
    parser.add_argument("--output_folder", type=str) #/model_training
    parser.add_argument("--model", default="gpt-4-0125-preview", type=str)
    parser.add_argument('--output_file',type=str)
    parser.add_argument("--category",type=str)
    # parser.add_argument("--category_cn",type=str,default="氛围")
    parser.add_argument("--language", type = str, default="en")
    parser.add_argument("--prompt_type",type = str)
    args = parser.parse_args()
    ### QUESTION: IS df_test the trianing file?
    df_text = pd.read_csv(args.input_file)#, encoding="utf-8", delimiter="\t")
    df_text = df_text.iloc[:]
    print(df_text.shape)
    # print(df_text)
    output_folder = args.output_folder

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    Path(os.path.join(output_folder)).mkdir(parents=True, exist_ok=True)


    pool = multiprocessing.Pool()

    responses = []
    results = pd.DataFrame(columns=['sample_id','national_id','English_Name','Chinese_Name',args.category])

    for restaurant_en, restaurant_cn, sample_id, national_id in tqdm(zip(df_text.English_Name.to_list(), df_text.Chinese_Name.to_list(), df_text.sample_id.to_list(),df_text.national_id.to_list())):
        # concept_name_string = " ".join(concept_name.split("_"))
        if args.language == "en":
            prompt = "./prompts/English/binary/{}.txt".format(args.prompt_type+args.prompt_file_path)
        else:
            prompt = "./prompts/Chinese/binary/{}.txt".format(args.prompt_type+args.prompt_file_path)
        # prompt = "./prompts/English/{}.txt".format(args.prompt_file_path)
        response = pool.apply_async(generate_categorization, args=(restaurant_en, restaurant_cn, prompt, args.language, args.model,args.category))
        # print(response)
        responses.append([sample_id, national_id, restaurant_en, restaurant_cn, response])
        # print('raw response -------------------------------------')
        # print(responses)

    for sample_id, national_id, restaurant_en, restaurant_cn, response in tqdm(responses):

        results.loc[len(results.index)] = [sample_id, national_id, restaurant_en, restaurant_cn, response.get()]


    ########## SWITCH THE CODE WHEN NEEDED. THE SECOND LINE IS FOR USING THE BEST MODEL FOR EACH CATEGORY
    results.to_csv(os.path.join(output_folder, args.prompt_type+args.output_file),encoding='utf-8')

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
