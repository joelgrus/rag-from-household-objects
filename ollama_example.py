import ollama

stream = ollama.generate(
    # This model is reasonably fast even on my crappy laptop
    'phi3:mini-128k',
    'What is the best deep learning framework?',
    stream=True
)

# The streaming is not strictly necessary,
# but it makes things feel snappier.
for chunk in stream:
  print(chunk['response'], end='', flush=True)
print()