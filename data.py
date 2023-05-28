
import json
import random
import csv
import re 


from nltk.tokenize import word_tokenize


prompts = [
    "[ WP ] Killing Hitler has become a sport amongst time travelers . Points are awarded for creativity and difficulty . You are last year 's champion , how did you win ?",
    "[ WP ] There is no prompt . Just write a story you 've always been thinking about or one you 've been thinking about sharing . Anything goes .",
    "[ WP ] `` She said she loved him . '' Insert the word `` only '' anywhere in this sentence . It must be the final sentence of your story .",
    "[ WP ] You are born without emotions ; to compensate this , you started a donation box where people could donate their unwanted emotions . You 've lived a life filled with sadness , fear and regret until one day , someone donates happiness .",
    "[ WP ] You live in a city full of people with powers ( telekinesis , electro kinesis , sensors , etc ) and everyone is ranked according to how powerful they but they can kill someone of higher rank and obtain their rank . You are rank # 1 but no one knows what your power is",
    "[ CW ] Write the first and last paragraph of a story and make me want to know what happened in between .",
    "[ WP ] Write a short story where the first sentence has 20 words , 2nd sentence has 19 , 3rd has 18 etc . Story ends with a single word .", 
    "[ WP ] You are a kid 's imaginary friend . They 're growing up . You 're fading away .",
    "[ WP ] This is the prologue ( or the first chapter ) of the novel you 've always wanted to write .",
    "[ WP ] To get in Heaven , you have to confront the person who you hurt the most . You were expecting an ex , your parents/relatives , or a friend . You did n't expect to see yourself ."
]


"""
with open('reddit_dataset/train.wp_source', 'r') as file:
    prompt_lines = file.readlines()
with open('reddit_dataset/train.wp_target', 'r') as file:
    response_lines = file.readlines()



### compile temp agnostic data ###

all_responses = list()
for i, prompt in enumerate(prompts):

    # human responses
    indices = [index for index, line in enumerate(prompt_lines) if (line.strip() == prompt)]
    random.shuffle(indices)
    indices = indices[:80]
    human_responses = [re.sub(' +', ' ', response_lines[index].replace("<newline>", " ")).strip() for index in indices]

    # GPT responses
    i += 1
    with open(f'gpt_dataset/P{i}_1.0.json', 'r') as file:
        gpt_responses = json.load(file)
    gpt_responses = [" ".join(word_tokenize(response)).strip() for response in gpt_responses]
    
    # Bard responses 
    with open(f'bard_dataset/Bard_P{i}.txt', 'r') as file:
        bard_responses = file.read()
    bard_responses = bard_responses.split("\n\n\n\n")
    bard_responses = [" ".join(word_tokenize(response)).strip() for response in bard_responses]
    
    # combine responses
    all_responses.append([human_responses, gpt_responses, bard_responses])

# write responses
with open('dataset/10-prompts_human-GPT-Bard.responses', 'w') as file:
    for human_responses, gpt_responses, bard_responses in all_responses:
        for response in human_responses:
            file.write(response + "\n")
        for response in gpt_responses:
            file.write(response + "\n")
        for response in bard_responses:
            file.write(response + "\n")

# write labels
all_labels = list()
for prompt_number, (human_responses, gpt_responses, bard_responses) in enumerate(all_responses):
    for response in human_responses:
        all_labels.append([prompt_number, 0])
    for response in gpt_responses:
        all_labels.append([prompt_number, 1])
    for response in bard_responses:
        all_labels.append([prompt_number, 2])
        
with open('dataset/10-prompts_human-GPT-Bard.labels', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow("PromptNumber,Label (0=human 1=GPT 2=Bard)".split(","))
        writer.writerows(all_labels)


### compile temp data ###

all_responses_2d = list()
for prompt in prompts:
    all_responses = list()
    # human responses
    indices = [index for index, line in enumerate(prompt_lines) if (line.strip() == prompt)]
    random.shuffle(indices)
    indices = indices[:20]
    # clean responses by removing <newline> tokens and extra spaces
    human_responses = [re.sub(' +', ' ', response_lines[index].replace("<newline>", " ")).strip() for index in indices]
    all_responses.append(human_responses)
        
    # GPT responses
    for temp in [round(x * 0.1, 1) for x in range(1, 16)]:
        with open(f'gpt_dataset/GPT_top10_temp_{temp}.json', 'r') as file:
            gpt_responses = json.load(file)
        random.shuffle(gpt_responses)
        gpt_responses = gpt_responses[:20]
        gpt_responses = [" ".join(word_tokenize(response)).strip() for response in gpt_responses]
        all_responses.append(gpt_responses)
    
    all_responses_2d.append(all_responses)

# write responses
with open('dataset/10-prompts_human-temps.responses', 'w') as file:
    for all_responses in all_responses_2d:
        for responses in all_responses:
            for response in responses:
                file.write(response + "\n")

# compile labels
all_labels = list()
for prompt_number, all_responses in enumerate(all_responses_2d):
    for temp, responses in enumerate(all_responses):
        for response in responses:
            all_labels.append([prompt_number, temp])

# write labels
with open('dataset/10-prompts_human-temps.labels', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow("PromptNumber,Temp (0=human 1=0.1 2=0.2 ...)".split(","))
    writer.writerows(all_labels)
"""
        
### save standardized prompts ###
""" 
standardized_prompts = [
    "Killing Hitler has become a sport amongst time travelers. Points are awarded for creativity and difficulty. You are last year's champion, how did you win?",
    "There is no prompt. Just write a story you've always been thinking about or one you've been thinking about sharing. Anything goes.",
    "``She said she loved him.'' Insert the word ``only'' anywhere in this sentence. It must be the final sentence of your story.",
    "You are born without emotions; to compensate this, you started a donation box where people could donate their unwanted emotions. You've lived a life filled with sadness, fear and regret until one day, someone donates happiness.",
    "You live in a city full of people with powers (telekinesis, electro kinesis, sensors, etc) and everyone is ranked according to how powerful they but they can kill someone of higher rank and obtain their rank. You are rank # 1 but no one knows what your power is",
    "Write the first and last paragraph of a story and make me want to know what happened in between.",
    "Write a short story where the first sentence has 20 words, 2nd sentence has 19, 3rd has 18 etc. Story ends with a single word.",
    "You are a kid's imaginary friend. They're growing up. You're fading away.",
    "This is the prologue (or the first chapter) of the novel you've always wanted to write.",
    "To get in Heaven, you have to confront the person who you hurt the most. You were expecting an ex, your parents/relatives, or a friend. You didn't expect to see yourself."
]
with open('dataset/prompts', 'w') as file:
    for prompt in standardized_prompts:
        file.write(prompt + "\n")
"""