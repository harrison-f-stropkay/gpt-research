import requests
import time

with open("dataset/10-prompts_human-GPT-Bard.responses", 'r') as file:
    writing_responses = file.readlines()

with open("dataset/10-prompts_human-GPT-Bard.labels", 'r') as file:
    labels = file.readlines()

with open("api_key.txt", "r") as f:
    api_key = f.read()

scan_endpoint = "https://api.originality.ai/api/v1/scan/ai"

headers = {
    "X-OAI-API-KEY": api_key,
}

scores = []
for writing_response in writing_responses:
    data = {
        "content": writing_response
    }

    response = requests.post(scan_endpoint, headers=headers, data=data)

    if response.status_code == 200:
        scores.append(response.json()['score']['ai'])
    # if API fails
    else:
        scores.append(-1)
        print(-1)
    
    # "Currently the maximum number of requests you can make per minute is 100"
    time.sleep(0.6)


with open("dataset/10-prompts_human-GPT-Bard.originality", 'w') as file:
    for score in scores:
        file.write(str(score) + "\n")

