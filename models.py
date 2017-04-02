from preprocessing import Get_Unlabelled_Data
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
# This classifier composes two models and uses
# Semi-supervised learning on the second model
# using the first's output

class Mnb:
	name, model = "Multinomial Naive Bayes", MultinomialNB()
	def fit(self, X, Y):
		self.model.fit(X,Y)
		
	def predict(self, X):
		return self.model.predict(X)
		
class Bnb:
	name, model = "Bernoulli Naive Bayes",	BernoulliNB()
	def fit(self, X, Y):
		self.model.fit(X,Y)
	def predict(self, X):
		return self.model.predict(X)

class RForest:
	name, model = "Random Forest Classifier", RandomForestClassifier(n_estimators = 100)
	def fit(self, X, Y):
		self.model.fit(X,Y)
	def predict(self, X):
		return self.model.predict(X)

# This is not yet working. I will try to make it work tomorrow night.
class SemiSupervised:
	model1, model2 = 0,0
	name = 0;
	def __init__(self, Model1, Model2 = None):
		if Model2 is None:
			Model2 = Model1
		self.model1, self.model2 = Model1, Model2
		if self.model1 is self.model2:
			self.name = "Semi-supervised " + self.model1.name
		else:
			self.name = "Semi-supervised " + self.model1.name \
					  + " WITH " + self.model2.name;
	
	def fit(self, X1, Y1):
		self.model1.fit(X1,Y1)
		X2 = Get_Unlabelled_Data()
		Y2 = self.model1.predict(X2)
		if self.model1 is not self.model2:
			del self.model1;
		self.model2.fit(X1+X2, Y1+Y2)
		del X2, Y2
	
	def predict(self, X):
		return self.model2.predict(X)