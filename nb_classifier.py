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

def count_of_class_from_sample(cls, samples):
  classes = [c for words, c in samples]
  return classes.count(cls)

class Classifier:
  def __init__(self, samples, classes):
    self.samples = samples
    self.classes = classes
    self.bag_of_words = bag_of_words_from_samples(samples)
    self.vocab = set(self.bag_of_words)
    self.vocab_for_class = {}
    self.bag_of_words_for_class = {}
    self.prob_of_class = {}
    
    for cls in classes:
      self.bag_of_words_for_class[cls] = bag_of_words_for_class(samples, cls)
      self.vocab_for_class[cls] = set(self.bag_of_words_for_class[cls])
      self.prob_of_class[cls] = count_of_class_from_sample(cls, samples)/len(samples) 

  def word_prob(self, word, cls):
    count_of_word_in_class = self.bag_of_words_for_class[cls].count(word)
    count_of_words_in_bag = len(self.bag_of_words_for_class[cls])
    vocabulary_size = len(self.vocab)
    prob = (count_of_word_in_class + 1) / (count_of_words_in_bag + vocabulary_size)
    return prob

  def class_prob(self, cls, doc):
    prob = self.prob_of_class[cls]
    for word in doc:
      prob = prob * self.word_prob(word, cls)
    return prob

  def __repr__(self):
    return "Classifier(%r, %r)" % (self.vocab, self.classes)


training = samples
testing = ["Chinese Chinese Chinese Tokyo Japan".split()]

print(training)

my_classifier = Classifier(training, classes)

print("word_prob P(Chinese, J)", my_classifier.word_prob("Chinese", "Japanese"))
print("word_prob P(Chinese, C)", my_classifier.word_prob("Chinese", "Chinese"))

print("Chinese", my_classifier.class_prob("Chinese", testing[0]))
print("Japanese", my_classifier.class_prob("Japanese", testing[0]))







