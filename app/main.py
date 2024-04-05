import io
import os
import joblib
import pickle
import warnings
import xgboost as xgb
import numpy as np
import pandas as pd
import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from fastapi.encoders import jsonable_encoder
from google.cloud import storage
from io import BytesIO
warnings.filterwarnings("ignore")
app = FastAPI()


def load_file(path):
    storage_client = storage.Client()
    bucket = storage_client.bucket("bucket-for-lending")
    blob = bucket.blob(f"{path}")

    byte_stream = BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)

    file = joblib.load(byte_stream)
    return file


def transform_dec(data):
    cols_scale_dec = ["Amount Requested", "Risk_Score", "Debt-To-Income Ratio"]
    data[cols_scale_dec] = scaler_dec.transform(data[cols_scale_dec])
    return data


def decision_unmap(decision):
    decision_unmap = { 1: "Accepted", 0: "Rejected"}
    return decision.map(decision_unmap)


def transform_gradesnrate(data, scaler):
    cols_to_scale = [
        "total_bal_ex_mort",
        "total_bc_limit",
        "bc_util",
        "annual_inc",
        "dti",
        "revol_bal",
        "revol_util",
        "total_acc",
        "avg_cur_bal",
        "bc_open_to_buy",
        "mo_sin_old_il_acct",
        "mo_sin_old_rev_tl_op",
        "mo_sin_rcnt_rev_tl_op",
        "mo_sin_rcnt_tl",
        "mths_since_recent_bc",
        "mths_since_recent_inq",
        "num_rev_accts",
        "tot_hi_cred_lim",
        "total_il_high_credit_limit",
        "fico",
        "funded_amnt",
        "installment",
        "monthly_load",
        "loan_amnt",
    ]
    data[cols_to_scale] = scaler.transform(data[cols_to_scale].values)
    return data


def grade_unmap(grade):
    grade_unmap = {6: "A", 5: "B", 4: "C", 3: "D", 2: "E", 1: "F", 0: "G"}
    return grade.map(grade_unmap)


def subgrade_unmap(subgrade):
    subgrade_unmap = {
        0: "A1",
        1: "A2",
        2: "A3",
        3: "A4",
        4: "A5",
        5: "B1",
        6: "B2",
        7: "B3",
        8: "B4",
        9: "B5",
        10: "C1",
        11: "C2",
        12: "C3",
        13: "C4",
        14: "C5",
        15: "D1",
        16: "D2",
        17: "D3",
        18: "D4",
        19: "D5",
        20: "E1",
        21: "E2",
        22: "E3",
        23: "E4",
        24: "E5",
        25: "F1",
        26: "F2",
        27: "F3",
        28: "F4",
        29: "F5",
        30: "G1",
        31: "G2",
        32: "G3",
        33: "G4",
        34: "G5",
    }
    return subgrade.map(subgrade_unmap)


scaler_dec = load_file("scalers/scaler_decision.pkl")
scaler_gr_sg_int = load_file("scalers/scaler_grade.pkl")

model_dec = load_file("models/model_decision.pkl")
model_gr = load_file("models/model_grade.pkl")
model_sg = load_file("models/model_subgrade.pkl")
model_intr = load_file("models/model_intr.pkl")


@app.post("/decision-batch")
async def transform_predict_batch(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    input_data = pd.read_csv(io.BytesIO(contents))
    transformed_data = transform_dec(input_data)
    prediction = model_dec.predict(transformed_data)
    prediction = decision_unmap(pd.Series(prediction))
    return prediction


@app.post("/decision-stream")
async def transform_predict_stream(request: Request):
    contents = await request.json()
    input_data = pd.DataFrame([contents])
    transformed_data = transform_dec(input_data)
    prediction = model_dec.predict(transformed_data)
    prediction = decision_unmap(pd.Series(prediction))
    return jsonable_encoder(prediction)


@app.post("/gradesnrate-batch")
async def gradesnrate_predict_batch(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    input_data = pd.read_csv(io.BytesIO(contents))

    grade_data = transform_gradesnrate(input_data, scaler_gr_sg_int)
    pred_gr = pd.Series(model_gr.predict(grade_data))

    subgrade_data = pd.concat([grade_data, pd.DataFrame(pred_gr, columns=["grade"])], axis=1)
    pred_sg = pd.Series(model_sg.predict(subgrade_data))

    subgrade_data.insert(len(subgrade_data.columns)-1, "sub_grade", pred_sg)
    int_data = subgrade_data.copy()
    pred_intr = model_intr.predict(int_data)

    grade = grade_unmap(pred_gr)
    subgrade = subgrade_unmap(pred_sg)

    result = pd.DataFrame({
        "grade": grade,
        "subgrade": subgrade,
        "int_rate": pred_intr.tolist(),
    })
    return result


@app.post("/gradesnrate-stream")
async def gradesnrate_predict_stream(request: Request):
    data = await request.json()
    input_df = pd.DataFrame([data])

    print(input_df)
    print(input_df.columns)

    grade_data = transform_gradesnrate(input_df, scaler_gr_sg_int)
    pred_gr = pd.Series(model_gr.predict(grade_data))

    subgrade_data = pd.concat([grade_data, pd.DataFrame(pred_gr, columns=["grade"])], axis=1)
    pred_sg = pd.Series(model_sg.predict(subgrade_data))

    subgrade_data.insert(len(subgrade_data.columns)-1, "sub_grade", pred_sg)
    int_data = subgrade_data.copy()
    pred_intr = model_intr.predict(int_data)

    grade = grade_unmap(pred_gr)
    subgrade = subgrade_unmap(pred_sg)

    result = {
        "grade": grade[0],
        "subgrade": subgrade[0],
        "int_rate": float(pred_intr[0]),
    }

    return jsonable_encoder(result)


@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI application!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
