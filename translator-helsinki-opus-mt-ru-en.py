from pytorch_transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch import device, cuda

EN_FILE = './data/corpus.en_ru.1m.en'
RU_FILE = './data/corpus.en_ru.1m.ru'
BATCH_SIZE = 32

device = device('cuda' if cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(device)


data_in = open('eval-ru-100.txt', 'r').readlines()
input_ids = tokenizer.batch_encode_plus(data_in, return_tensors="pt", padding=True).data['input_ids']
outputs = model.generate(input_ids.to(device))
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

data_out = open('answer.txt', 'w')
data_out.write('\n'.join(decoded))
data_out.close()