**Final Assignment**


Your company, DS Pros, would like to win a contract with a big city council as it would give us great PR. To do so you think it would be a great idea to proactively browse in the open data sets of this city (the one you choose, total freedom here) identify a situation that could be solved or improved using classification algorithms and present it to the technical office of that city council.
You need to prepare the following:

	• A presentation describing the solution you try to solve, how classification will solve it and a summary of the solution proposed
	• A well documented and visually appealing notebook where you try different models, explain the steps followed and chose one particular algorithm and hyperparameters (explaining why)
	• You should also export that model, once trained, using pickle or similar so it can be reused.
	• You should implement a .py script that loads the exported model, accepts a file with samples to classify (identified with an id) and stores the results in a DDBB table (SQLlite) with fields id and class.
	• You should provide the files to test the .py script and clear instructions on how to run it.
 
Happy coding!!

**PD**: *Steps by Steps:*

	• pre_processing.ipynb for cleaning the dataset, EDA, and some traditionals algorithms.
	• models.ipynb for modelling and getting the best model.
	• best_model.py for testing the best model (ANN) using the sample from the original dataset
		• This .py exports the classification_request.db with the requests_predictions table created
		• to execute this script, please use the following code: python best_model.py models/models_testing.csv

