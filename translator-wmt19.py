from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from torch import device, cuda
device = device('cuda' if cuda.is_available() else 'cpu')
mname = "facebook/wmt19-ru-en"
model = FSMTForConditionalGeneration.from_pretrained(mname).to(device)
tokenizer = FSMTTokenizer.from_pretrained(mname)

data_in = open('eval-ru-100.txt', 'r').readlines()
input_ids = tokenizer.batch_encode_plus(data_in, return_tensors="pt", padding=True).data['input_ids'].to(device)
outputs = model.generate(input_ids)
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

data_out = open('answer.txt', 'w')
data_out.write('\n'.join(decoded))
data_out.close()