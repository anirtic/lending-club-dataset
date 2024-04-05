# Loan Approval Classification: Grade / Subgrade / Interest Rate Prediction

Loan Approval Prediction is a project aimed at predicting the approval status of loan applications based on various features such as applicant information, loan amount, credit score, and more. In addition to approval classification, the model also classifies loans by grade/subgrade and predicts the interest rate for accepted loans.
## Project Files and Folders

- `models_comparison_output/`: Contains saved .txt files with various models' performance metrics. This is used in the notebook to prevent the need to re-run all the code every time.
- `usshapefiles/`: Folder with shapefiles for mapping the United States and its states.
- `app/`: Contains the FastAPI application and a Dockerfile for creating a Docker image. The app loads models and scalers from a GCP bucket.
- `testdata`: Contains testing data for batch endpoints for decision and grade/subgrade/interest rate models.
- `functions.py`: External functions used in the analysis.
- `Project 3.3.ipynb`: Jupyter Notebook containing the project code.
- `stream_mode_test`: Notebook used for testing stream mode endpoints by passing a single line of JSON data.

#### Usage
Running the Jupyter Notebook 
1. Open the Project 3.3.ipynb Jupyter Notebook.
2. Follow the instructions within the notebook to explore the project, conduct data analysis, and build machine learning models.

#### Using the FastAPI Application
#### Batch mode:
1. Open https://lendingc33-26ulqxbbaq-uc.a.run.app/docs in a browser.
2. For the decision model, choose the /decision-batch endpoint, select the "Try it out" button, and upload the file located at /testdata/X_testdata_decision.csv, then click the "Execute" button.
3. For the grade/subgrade/interest rate model, choose the /gradesnrate endpoint, select the "Try it out" button, and upload the file located at /testdata/X_testdata_gr_sg_intr.csv, then click the "Execute" button.

##### Stream mode:
1. Open `stream_mode_test` notebook and follow the instructions there.

### Performance:
#### Loan Decision Model:
Able to handle a single request on average in 210ms, resulting in 285 requests per minute.
#### Grade / Subgrade / Interest rate Model:
Able to handle a single request on average in 300ms, resulting in 200 requests per minute.
