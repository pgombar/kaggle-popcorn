# Bag of Words Meets Bags of Popcorn

**Please commit any changes so we can merge easily**

## **TODO**
(notes from the meeting with Jesse)

* Feature detectors:
  - N-grams (Dennis), Done
  - No-grams (Dennis)
  - Word2Vec
  - Regex (Panni)

* Classifiers:
  - Vawpal Wabbit
  - Semi supervised learning (Csaba: in progress)

## About n-gram:

The n-gram feature combines 1-grams(=BoW), 2-grams ...., n-grams for feature creation.
Now I set the minimum document-frequency to 1/10000 and it improved the BoW for about 2%.
Yet it might be beneficial to indeed include a lot of features when involving 2- and 3-grams. Do you have an idea for a good classifier that can handle a lot of features ?

## About no-grams idea:

Creating 2-grams of the form: no+adjective and then:
if adj = positive --> 2-gram is negative 2-gram
if adj = negative --> 2-gram is positive 2-gram
And then simply count amount of positive vs. negative 2-grams

## Error to fix:

Error fixed thanks to Dennis. There's another
error, I (Csaba) will solve it tomorrow night. Until then,
do not use the `SemiSupervised` class.
