import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import ElectraTokenizer
from tqdm.auto import tqdm as tq

from config import *

data = pd.read_csv(train_data)


tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
tagged_data = [
    TaggedDocument(
        words=tokenizer.tokenize(txt[:1024]), 
        tags=[str(i)]
    ) for i, txt in enumerate(data["overview"])
]
    
max_epochs = 100
vec_size = CFG["embedding_dim"]
alpha = 0.025
model = Doc2Vec(tagged_data,
                vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                seed = 33,
                dm =1)

for epcoch in range(10):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha


# for index, row in tq(data.iterrows(), total=len(data)):
# 	if feature == 'text':
#         tokens = tokenizer.tokenize(row['text'])
#         tags = [row['title']]
#     elif feature == 'tag':
#         tokens = tokenizer.tokenize(row['text'])
#         tags = row['category']
# 	else:
# 		raise "Feature not defined"

# 	tagged_docs.append(TaggedDocument(tokens, tags))