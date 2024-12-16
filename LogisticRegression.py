#

#Titanic Data Analysis.
#Explore the Titanic Dataset and explore about the people, both those who survived and those who did not. With today's technology, answering questions through data analysis is now easier than ever.
#What factors made people more likely survive the sinking of the Titanic?


#Collect Data: Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
% matplotlib inline

titanic_data = pd.read_csv('Titanic.csv')


#SUV Data Analysis 
# A Car Company has released a new SUV in the market. Using the previous data about the sales of their SUV's, they want to predict the category of peopel who might be interested in buying this.
# What factors made people more interested in buying the SUV?
