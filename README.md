This is (approximately) the code I used for a talk I gave at the San Antonio AITX Meetup in August 2024, "Build Your Own RAG from Common Household Objects."

Here the "common household objects" are

1. [Ollama](https://ollama.com/)
2. [Sentence Transformers](https://www.sbert.net/index.html)
3. a dataset (I used the page listing the [Ig Nobel Prize winners](https://improbable.com/ig/winners/)

# The Idea

An LLM can generate text like

```python
import ollama

result = ollama.generate(
    # This model is reasonably fast even on my crappy laptop
    'phi3:mini-128k',
    'What is the best deep learning framework?',
)

print(result['response'])
```

But it only knows whatever information it was trained on.
If you want to it use other information (or only specific information), 
you could fine-tune it with that information, but that's a lot of work.
Or you could just feed it that information as part of the prompt:

```python
best_framework = "joelnet"

result = ollama.generate(
    # This model is reasonably fast even on my crappy laptop
    'phi3:mini-128k',
    'What is the best deep learning framework? '
    f'(hint: the answer is {best_framework})',
)

print(result['response'])
```

The idea behind Retrieval-Augmented Generation ("RAG")
is that we use some kind of information retrieval (here, embedding similarity)
to find documents related to a query, shove them into a prompt, and then get 
the LLM to craft an answer.

# The Household Objects

## Ollama

Ollama allows you to run a LLM on your local machine. 
This is more convenient than using (say) the OpenAI API,
especially if (say) you work somewhere where you are not allowed to use the OpenAI API.

If you have a powerful MacBook or a GPU, `llama3.1` is a very good model.
If you have a wimpier computer, `phi3:mini-128k` is ... good for its size?

## Sentence-Transformers

Here we use sentence-transformers to create embeddings for our documents and query.
The `all-MiniLM-L6-v2` model is not ideal for this, but it's fast and good enough.

## The Dataset

Use your own!

# Running the RAG

1. Make sure you have Ollama running and have pulled whatever model you're going to use.
2. Make sure you have a recent Python and create a virtualenv in your favorite way.
3. `pip install` the requirements
4. `python main.py`

And then you get a little interactive session:

```
Ask a question: what sleep apnea treatment won a prize
The group that won a prize for their contribution in treating sleep apnea was the team consisting of Milo A. Puhan, Alex Suarez, Christian Lo Cascio, Alfred Zahn, Markus Heitz, and Otto Braendli. They were recognized jointly with Fernanda Ito, Enrico Bernard, Rodrigo Aneros (Tyron Anero), for their research on the effectiveness of using a didgeridoo as treatment against obstructive sleep apnea syndrome. The prize they received was from France and UK's Medicine Prize [FRANCE, UK]. They delivered their acceptance speech via recorded video during the award ceremony that took place in December 2016 at Acta Chiropterologica journal publishing house
```

Anyway, it is not the world's best RAG, 
but it's also only 50 lines of code, 
and the goal is mainly to show you how a RAG works.
You could make it better if you wanted.