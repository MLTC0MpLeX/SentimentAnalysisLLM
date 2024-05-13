
DATA_PATH = "training.1600000.processed.noemoticon.csv"
SAMPLE_SIZE = 100
MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
SYSTEM_PROMPT = "You are a sentiment analysis robot. Judge the sentiment of the upcoming text if it is positive reply '4' is it is negative reply '0'. BE CRITICAL. DO NOT HESITATE BE AS OBJECTIVE AS POSSBILE. ONLY respond with 0 or 4. NO OTHER RESPONSES ARE ALLOWED. DO NOW ELABORATE. YOU ARE ONLY ALLOWED TO USE ANY OTHER CHARACTERS APART FROM 1 and 4. NEGATIVE = 0 || POSITIVE = 4"
PROMPT_TEMPLATE = "USER: {0}\nASSISTANT:"