# Use yandex dataset (https://translate.yandex.ru/corpus?lang=en)
from tqdm import tqdm
from transformers import FSMTForConditionalGeneration, FSMTTokenizer, \
    AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamW
from torch import device, cuda
from nltk.translate.bleu_score import corpus_bleu
device = device('cuda' if cuda.is_available() else 'cpu')

EN_FILE = './data/corpus.en_ru.1m.en'
RU_FILE = './data/corpus.en_ru.1m.ru'
BATCH_SIZE = 8
EPOCH_COUNT = 4


def batcher(size, en, ru):
    en_file = open(en)
    ru_file = open(ru)

    en_batch = []
    ru_batch = []
    for en_line, ru_line in zip(en_file, ru_file):
        en_batch.append(en_line)
        ru_batch.append(ru_line)

        if len(en_batch) == size:
            yield en_batch, ru_batch
            en_batch = []
            ru_batch = []


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=1e-5)

data_in = open('eval-ru-100.txt', 'r').readlines()

for epoch in range(EPOCH_COUNT):
    for batch in tqdm(batcher(BATCH_SIZE, EN_FILE, RU_FILE), total=1000000//BATCH_SIZE):
        en_sample = tokenizer.batch_encode_plus(
            batch[0], return_tensors="pt", padding=True).data

        ru_sample = tokenizer.batch_encode_plus(
            batch[1], return_tensors="pt", padding=True).data

        loss = model(
            ru_sample['input_ids'].to(device),
            labels=en_sample['input_ids'].to(device)
        )

        loss = loss.loss
        loss.backward()
        optimizer.step()

    print(f'Evaluation, epoch {epoch} ...', end='')
    input_ids = tokenizer.batch_encode_plus(
        data_in, return_tensors="pt", padding=True).data['input_ids'].to(device)
    outputs = model.generate(input_ids)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    score = corpus_bleu(
        [x.split(' ') for x in data_in],
        [x.split(' ') for x in decoded])

    print(f'bleu score: {score:3f}')

    if input('Continue? y/n').lower() == 'n'.lower():
        break

model.eval()
input_ids = tokenizer.batch_encode_plus(
    data_in, return_tensors="pt", padding=True).data['input_ids'].to(device)
outputs = model.generate(input_ids)
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
data_out = open('answer.txt', 'w')
data_out.write('\n'.join(decoded))
data_out.close()