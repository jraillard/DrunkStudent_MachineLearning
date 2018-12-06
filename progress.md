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
			=> DummyClassfier (useless)
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

train_data = 80% data
test_data = 20% data

Start estimations
Linear regression program
	- With all descriptors :
		Dalc => error : 0.666 (13.32%)
		Walc => error : 0.896 (17.92%)
		test fit with Dalc & Walc then test without Dalc column => error : 1.91 (38,2%) might be over-learning
		
```

## 06/12/2018

```
Retest without descriptors given by correlation matrice 
Linear regression program
	- With chosen descriptors :
		Dalc => error : 0.682 (13.64%)
		Walc => error : 0.952 (19.04%)
		=> better with all descriptors

KNeighbors program
	- With all descriptors :
						k=3 	k=7		k=8		k=10  	k=15			
		Dalc => error : 27.8%	24.0%	22,7%	22.7%	24.0%			
		Walc => error : 64.5%	63.2%	64.5%	64.5%	58.2%

	- With chosen descriptors :
						k=3 	k=7		k=8		k=10  	k=15					
		Dalc => error : 30.3%	24.0%	25.3%	24.0%	22.7%	
		Walc => error : 65.8%	58.2%	56.9%	63.2%	60.8%

Bayes program
	- With all descriptors :						
		Dalc => error : 58.2%
		Walc => error : 49.4%

	- With chosen descriptors :						
		Dalc => error : 27.8%
		Walc => error : 39.2%

Decision tree program
	k = tree max_depth
	- With all descriptors :	
						k=2 	k=3		k=5 	k=10  	k=15
		Dalc => error : 24.0%	27.8%	27.8%	22.7%	24.0%
		Walc => error : 45.6%	43.0%	53.2%	69.6%	64.5%

	- With chosen descriptors :	
						k=2 	k=3		k=5 	k=10  	k=15
		Dalc => error : 24.0%	25.3%	26.6%	32.9%	31.6%
		Walc => error : 45.5%	41.7%	58.2%	59.4%	65.8%

Voting  program
	=> Kneighbors (k=8)
	=> Bayes
	=> Decision Tree (k=3)
	k = voting mode
	- With all descriptors :	
						k=soft	k=hard
		Dalc => error : 24.1%	24.1%
		Walc => error : 53.2%	53.2%

	- With chosen descriptors :	
						k=soft	k=hard
		Dalc => error : 26.6%	25.3%
		Walc => error : 40.5%	45.5%

RandomForest program
	k = estimator counter
	- With all descriptors :	
						k=2 	k=3		k=5 	k=10  	k=15
		Dalc => error : 31.6%	22.8%	25.3%	22.8%	25.3%
		Walc => error : 46.8%	62.2%	55.7%	50.6%	57.0%

	- With chosen descriptors :	
						k=2 	k=3		k=5 	k=10  	k=15
		Dalc => error : 26.6%	30.4%	25.3%	29.1%	22.8%
		Walc => error : 59.5%	62.0%	55.7%	59.5%	50.7%

SVM programs
	- LinearSVC
		k = C parameter 
		- With all descriptors :	
							k=2 	k=3		k=5 	k=10  	k=15
			Dalc => error : 31.6%	22.8%	25.3%	22.8%	25.3%
			Walc => error : 46.8%	62.2%	55.7%	50.6%	57.0%

		- With chosen descriptors :	
							k=2 	k=3		k=5 	k=10  	k=15
			Dalc => error : 26.6%	30.4%	25.3%	29.1%	22.8%
			Walc => error : 59.5%	62.0%	55.7%	59.5%	50.7%
	
	- SVC 
		k = kernel chosen 
		- With all descriptors :	
							k='linear'		k='poly' 	k='rbf'		k='sigmoid'  	k='precomputed'
			Dalc => error : 
			Walc => error : 

		- With chosen descriptors :	
							k='linear'		k='poly' 	k='rbf'		k='sigmoid'  	k='precomputed'
			Dalc => error : 
			Walc => error :
		
```