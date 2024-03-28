# Loan Approval Prediction

Loan Approval Prediction is a project that aims to predict the approval status of loan applications based on various features such as applicant information, loan amount, credit score, and more. The project employs machine learning models for this purpose.

## Project Files and Folders

- `Project 3.3.ipynb`: Jupyter Notebook containing the project code.
- `usshapefiles/`: Folder with shapefiles for mapping the US country and states.
- `4models/`: Folder containing machine learning models in .pkl format, as well as data preprocessing components such as scaler. Also, includes a FastAPI application and Dockerfile for creating a docker image.
- `testdata`: Folder containing testing data for decision and grade/subgrade/int.rate models.
- `functions.py` External functions used in analysis

#### Usage
Running the Jupyter Notebook 
1. Open the Project 3.3.ipynb Jupyter Notebook.
2. Follow the instructions within the notebook to explore the project, conduct data analysis, and build machine learning models.

#### Using the FastAPI Application
1. Open https://tcs33-iflehjyjlq-uc.a.run.app/docs on a browser.
2. For decision model choose /decision end point select button "Try it out" and upload the file located on /testdata/X_testdata_decision.csv and click button "Execute".
3. For grade/subgrade/interest_rate model choose /gradesnrate end point select button "Try it out" and upload the file located on /testdata/X_testdata_gr_sg_intr.csv and click button "Execute".


#### Note:
I was not able to upload model_subgrade.pkl file into github, because it weights more than 25MB. Please write me a message on discord, so I could share it there, if needed. Thanks.