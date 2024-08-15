from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

docs = [
    'AI in Texas',
    'AI in New York',
    'AI in California',
]

embeddings = model.encode(docs)

query = 'Large Language Models in San Antonio'

query_embedding = model.encode(query)

similarities = cos_sim(query_embedding, embeddings).flatten()

print(similarities)