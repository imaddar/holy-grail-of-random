customer_id,age,occupation_status,years_employed,annual_income,credit_score,credit_history_years,savings_assets,current_debt,defaults_on_file,delinquencies_last_2yrs,derogatory_marks,product_type,loan_intent,loan_amount,interest_rate,debt_to_income_ratio,loan_to_income_ratio,payment_to_income_ratio,loan_status
customer_id,age,occupation_status,years_employed,annual_income,credit_score,credit_history_years,savings_assets,current_debt,defaults_on_file,delinquencies_last_2yrs,derogatory_marks,product_type,loan_intent,loan_amount,interest_rate,debt_to_income_ratio,loan_to_income_ratio,payment_to_income_ratio,loan_status

sample for the first 10 data points:

CUST100000,40,Employed,17.2,25579,692,5.3,895,10820,0,0,0,Credit Card,Business,600,17.02,0.423,0.023,0.008,1
CUST100001,33,Employed,7.3,43087,627,3.5,169,16550,0,1,0,Personal Loan,Home Improvement,53300,14.1,0.384,1.237,0.412,0
CUST100002,42,Student,1.1,20840,689,8.4,17,7852,0,0,0,Credit Card,Debt Consolidation,2100,18.33,0.377,0.101,0.034,1
CUST100003,53,Student,0.5,29147,692,9.8,1480,11603,0,1,0,Credit Card,Business,2900,18.74,0.398,0.099,0.033,1
CUST100004,32,Employed,12.5,63657,630,7.2,209,12424,0,0,0,Personal Loan,Education,99600,13.92,0.195,1.565,0.522,1
CUST100005,32,Employed,13.4,32015,570,7.3,253,1120,0,0,2,Credit Card,Personal,37000,22.92,0.035,1.156,0.385,0
CUST100006,53,Employed,22.9,44989,674,11.1,19667,19298,0,0,0,Personal Loan,Home Improvement,45600,11.02,0.429,1.014,0.338,1
CUST100007,44,Self-Employed,4.2,80603,625,18.5,830,38382,0,0,0,Credit Card,Personal,51700,19.42,0.476,0.641,0.214,1
CUST100008,29,Employed,5.9,28416,569,2.6,1334,22668,1,2,0,Credit Card,Education,33800,22.72,0.798,1.189,0.396,0


Some notes on interesting facts about columns

credit_score appears gaussian in nature, with a mean at ~643

1. will try an initial implementation/train without dealing with multicollinearity
  then compare this with dealing with multicollinearity (removing some features, PCA, etc)
  
next steps
1. implement homemade decision tree
2. compare it with a decision tree package
3. build out appropriate evals
  accuracy, precision, recall, etc.
4. implement random forest by scratch
5. compare with random forest package
6. do some hyperparameter tuning
7. try to implement XGBoost
8. compare with baseline
9. implement very basic neural network, afterwards check kaggle approach
10. ++ see if I can make this production ready
