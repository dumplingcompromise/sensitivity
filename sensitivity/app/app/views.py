from app import app
from flask import render_template
from flask import request, redirect
import pandas as pd
import pickle
import numpy as np

filename = 'app/static/model/rf_model.pkl'

with open(filename,'rb') as f:
	rfm,x_train_columns,feature_importances = pickle.load(f)




@app.route("/")
def index():
    return render_template("index.html")


@app.route("/form", methods=["GET", "POST"])
def sign_up():
	req = request.form
	if request.method == "POST":
		
		#e2e = req["end_to_end"]
		#label = req["label"]
		#co_tt = req["co_tt"]
		#print(int(e2e)+int(label)+int(co_tt))

		#e2e_new = req["end_to_end_new"]
		#label_new = req["label_new"]
		#co_tt_new = req["co_tt_new"]

		Baseline_Dict = {}
		Baseline_Dict["end_to_end_turn_time"] = [req["end_to_end"]]
		Baseline_Dict["crt_cust_response_tt"] = [req["crt_cust_response_tt"]]
		Baseline_Dict["customer_origination_tt"] = [req["co_tt"]]
		#Baseline_Dict["repair_tt"] = 
		Baseline_Dict["package_del_tt"] = [req["label"]]
		Baseline_Dict["dmg_tt"] =  [req["dmg_tt"]]
		Baseline_Dict["job_creation_tt"] = [req["job_creation_tt"]]


		Proposed_Dict = {}
		Proposed_Dict["end_to_end_turn_time"] = [req["end_to_end_new"]]
		Proposed_Dict["crt_cust_response_tt"] = [req["crt_cust_response_tt_new"]]
		Proposed_Dict["customer_origination_tt"] = [req["co_tt_new"]]
		#Proposed_Dict["repair_tt"] = 
		Proposed_Dict["package_del_tt"] = [req["label_new"]]
		Proposed_Dict["dmg_tt"] =  [req["dmg_tt_new"]]
		Proposed_Dict["job_creation_tt"] = [req["job_creation_tt_new"]]


		baseline_df = pd.DataFrame.from_dict(Baseline_Dict,orient='columns')
		my_zero_df = pd.DataFrame(data=[np.zeros(len(x_train_columns))],columns=x_train_columns)
		my_zero_df[my_zero_df.columns & baseline_df.columns] = baseline_df
		
		baseline_osat = rfm.predict_proba(my_zero_df)[0][1]

		proposed_df = pd.DataFrame.from_dict(Proposed_Dict,orient='columns')
		my_zero_df_2 = pd.DataFrame(data=[np.zeros(len(x_train_columns))],columns=x_train_columns)
		my_zero_df_2[my_zero_df_2.columns & proposed_df.columns] = proposed_df

		proposed_osat = rfm.predict_proba(my_zero_df_2)[0][1]
		
		expected_delta = (proposed_osat - baseline_osat) / baseline_osat



		return render_template("result.html",result=req,baseline_osat=baseline_osat,proposed_osat=proposed_osat,expected_delta=expected_delta)
	else:
		return render_template("form.html",result=req)



