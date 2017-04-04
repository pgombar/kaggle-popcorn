from preprocessing import Get_Unlabelled_Data
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from numpy import concatenate
from scipy.sparse import vstack
# This classifier composes two models and uses
# Semi-supervised learning on the second model
# using the first's output

class Mnb:
	name, model = "Multinomial Naive Bayes", MultinomialNB()
	feature = None;	# Semi-supervised model needs to know the feature as well
	def fit(self, X, Y):
		self.model.fit(X,Y)
		
	def predict(self, X):
		return self.model.predict(X)
		
class Bnb:
	name, model = "Bernoulli Naive Bayes",	BernoulliNB()
	feature = None;
	def fit(self, X, Y):
		self.model.fit(X,Y)
	def predict(self, X):
		return self.model.predict(X)

class RForest:
	name, model = "Random Forest Classifier", RandomForestClassifier(n_estimators = 100)
	feature = None;
	def fit(self, X, Y):
		self.model.fit(X,Y)
	def predict(self, X):
		return self.model.predict(X)

# This is not yet working. I will try to make it work tomorrow night.
class SemiSupervised:
	model1, model2 = 0,0
	name = 0;
	feature = 1; # This is the feature needed to fit the data
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
		X2 = self.feature.transform(Get_Unlabelled_Data())
		Y2 = self.model1.predict(X2)
		if self.model1 is not self.model2:
			del self.model1;
		X = vstack([X1,X2])
		Y = concatenate((Y1,Y2))
		self.model2.fit(X, Y)
		del X, Y, X2, Y2
	
	def predict(self, X):
		return self.model2.predict(X)