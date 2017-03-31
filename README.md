# kaggle-popcorn

## **TODO**
(notes from the meeting with Jesse)

* Feature detectors:
  - N-gramms (Dennis)
  - Word2Vec
  - Regex (Panni)

* Classifiers:
  - Vawpal Wabbit
  - Semi supervised learning (Csaba: in progress)

## Error to fix: Please help!

So python (BeautifulSoup) throws an error while preprocessing reviews at
almost at the end of the file. I have no idea why this happens, but here
comes what I do know.

It crashes with an error at the 45092nd review while cleaning the file
`unlabelledTrainData.tsv`. See call stack in `error.txt`. Since numbering started
with 0, but the file line labeling starts with 1, and there is an extra
line in the file, the error should be on the 45094th line.

Copy-pasting the line to the interpreter and running the BeautifulSoup
html tag cleaning command yields no error. All Other line in the
neighbourhood work just fine as well. Googleing the error does not help
either.

The error info is in `error.txt`.
