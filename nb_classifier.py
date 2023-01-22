# Naive Bayes Classifier
import nltk


samples = [
  ["Chinese Beijing Chinese".split(), "Chinese"],
  ["Chinese Chinese Shanghai".split(), "Chinese"],
  ["Chinese Macao".split(), "Chinese"],
  ["Tokyo Japan Chinese".split(), "Japanese"]
]

classes = ["Chinese", "Japanese"]

def bag_of_words_for_class(samples, cls):
  bag_of_words = []
  for (words, c) in samples:
    if c == cls:
      bag_of_words.extend(words)
  return bag_of_words

def bag_of_words_from_samples(samples):
  bag_of_words = []
  for (words, cls) in samples:
    bag_of_words.extend(words)
  return bag_of_words

class Classifier:
  def __init__(self, samples, classes):
    self.samples = samples
    self.classes = classes
    self.bag_of_words = bag_of_words_from_samples(samples)
    self.vocab = set(self.bag_of_words)
    self.vocab_for_class = {}
    self.bag_of_words_for_class = {}
    for cls in classes:
      self.bag_of_words_for_class[cls] = bag_of_words_for_class(samples, cls)
      self.vocab_for_class[cls] = set(self.bag_of_words_for_class[cls])

  def __repr__(self):
    return "Classifier(%r, %r)" % (self.vocab, self.classes)


training = samples
testing = ["Chinese Chinese Chinese Tokyo Japan".split()]

print(training)

my_classifier = Classifier(training, classes)







