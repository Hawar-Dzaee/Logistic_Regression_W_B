# Logistic_Regression_W_B

This web application visualizes a logistic regression model with one feature and its corresponding loss landscape. Users can interact with the model by adjusting weight **w**  and bias **bias** using widgets, and observe how changes affect the loss function and help reach the global minima. In addition to that you can check the confusion matrix on training data. 


## Features

* Plot of logistic Regression Model: Visualizes how the model fits a given dataset with one feature.

* Loss Landscape Plot: Shows the Binary Cross Entropy Loss (BCE Loss) landscape for different values of weight and bias.

* Dataset Generation using widgets : you can select sample number of each class, and add noise to it.

* Threshold Select : you can select the decision boundary between class orange and class purple.

* Interactive Widgets: Adjust weight (w) & bias (b) in real-time and see the effect on the loss function and  sigmoid curve.


* Equations Display: Displays the equations used in the plots:
      sigmoid: Used in the first plot, and the probability of each input(y_hat).
      Binary Cross Entropy (BCE) Equation: Used in the second plot.
      Confusion matrix: Third plor. 




## Installation

To run this application, you need to have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository:

    `git clone https://github.com/Hawar-Dzaee/Logistic_Regression_W_V2.git`


2. Install the required packages:

    `pip install -r requirements.txt`

## Usage

To run the application, use the following command:

  `streamlit run main.py`



## Files

  main.py: Contains the code for the web application.
  LO_W_B.ipynb : a notebook that follows main.py (to some extend)
  requirements.txt: Lists the required Python packages.
  LICENSE : MIT License



## Example

Once the application is running, you will see three plots:

1. sigmoid curve fitting the data :

  * Shows the S-shaped curve fitting the dataset with one feature(two classes 0 & 1).


2. Loss Landscape Plot:

  * Displays Binary Cross Entropy (BCE) landscape.

3. Confusion matrix plot: 
  
  * Confusion matrix for orange & purple classes. 





## Requirements 

The application requires the following Python packages:

* Streamlit
* Torch
* Numpy
* Plotly



## License

This project is licensed under the MIT License. See the **LICENSE** file for more details.


Contact
For any questions or suggestions, please open an issue or contact [hawardizayee@gmail.com].