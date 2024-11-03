import torch
from transformers import BartTokenizer,BartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
checkpoint_path = "/home/user3/workplace/results/checkpoint-59500"
model_name = "facebook/bart-large-cnn" 
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(checkpoint_path).to(device)

def generate_email(text,politeness_level):
    input_text_formatted = f"politeness: {politeness_level} / email content:{text}"
    inputs = tokenizer(input_text_formatted,return_tensors='pt',max_length = 512, truncation=True).to(device)
    input_ids = inputs.input_ids
    print(f"[debug] input_ids:{input_ids}")
    summary_ids = model.generate(
        input_ids,
        max_length=512,
        length_penalty=2.0,
        num_beams=6, 
        early_stopping=True,
        no_repeat_ngram_size=3, #prevent repeated sentences.
        do_sample = True, # random sampling. if not -> always show same output via greedy method/beams search
        temperature=1 # higher value makes senteces more creative. word statistics
         )
    print(f"[debug] summary_ids:{summary_ids}")
    summary = tokenizer.decode(summary_ids[0],skip_special_tokens=True)
    print(f"[debug] summary:{summary}")
    return summary

test_text = "Dear dewin,Thanks for instruction for job posting. I attached a poster including job openings and company profile.I have a few more questions about the charge. We want to upload our job vacancies on the UGM website and advertise through email. We have total 4 job openings. I would like to know how much it will take.Please let me know if you need more information.Thanks. Sincerely yours,Jun"
pl_level = "P_9" #input("politeness level: ")
generated_email = generate_email(test_text,pl_level)
print(f"generated content as pl_level: {pl_level}\n {generated_email}")