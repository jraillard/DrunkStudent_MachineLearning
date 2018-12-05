# Machine Learning on Student Alcohol Comsuption

Progress File of our project. 

Dataset reminder : https://www.kaggle.com/uciml/student-alcohol-consumption

## 21/11/2018

```
Git Init
Adding :
	- Dataset csv file
	- Get TD's programs : 
		Linear regression
		Supervised Classifier
			=> DummyClassfier
			=> KNeighbors
			=> Bayes
			=> DecisionTree
			=> GroupLearning
				~ VotingClassifier
				~ RandomForestClassifier
			=> SVM
		Non-Supervised Classifier 
			=> KMeans
			=> GaussianMixture

	- Used the following link to get the matrice correlation between the different factors instead of using TD's programs to determine it : https://www.kaggle.com/kanncaa1/does-alcohol-affect-success/notebook

		=> Walc & Dalc to determine 
		=> most correlate factors (> 0.08 , factor's correlation rate start from -0.25 to 1) : age, traveltime, failures, freetime, goout (going out), health, absences

	- First python program 
		=> deleting unrelated factors :
			~ 1-School
			~ 11-Reasons (to choose school)
		=> replace non integer values (binary and nominals)
```

## 05/12/2018

```
Classifier choice : 
	=> Linear Regression : maybe ? might have too much descriptor
	=> Supervised : best fit
	=> Non-Supervised : got descriptors , so useless

Dalc and Walc are scored between 0 and 5

Start estimations
Linear regression programs
	Dalc => error : 0.666 (13.32%)
	Walc => error : 0.896 (17.92%)
	test fit with Dalc & Walc then test without Dalc column => error : 1.91 (38,2%) might be over-learning

```