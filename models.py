from preprocessing import Get_Unlabelled_Data
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from numpy import concatenate, repeat
from scipy.sparse import vstack
# This classifier composes two models and uses
# Semi-supervised learning on the second model
# using the first's output
class SemiSupervised:
	model1, model2 = 0,0
	feature = 1; # This is the feature needed to fit the data
	def __init__(self, Model1, Model2 = None):
		if Model2 is None:
			Model2 = Model1
		self.model1, self.model2 = Model1, Model2
	
	def __str__(self):
		if self.model1 is self.model2:
			return "Semi-supervised " + str(self.model1);
		else:
			return "Semi-supervised " + str(self.model1) \
				 + " WITH " + str(self.model2);

	def fit(self, X1, Y1):
		self.model1.fit(X1,Y1)
		X2 = self.feature.transform(Get_Unlabelled_Data())
		Y2 = self.model1.predict(X2)
		if self.model1 is not self.model2:
			del self.model1;
		self.model2.partial_fit(X2, Y2, None, repeat(0.2, len(Y2)))
		del X2, Y2
	
	def predict(self, X):
		return self.model2.predict(X)