from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
app.secret_key = 'project_anusha'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form inputs
        print("Input Received")
        
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Access age from the session
    name = request.form.get('name')
    age = request.form.get('age')
    income = request.form.get('income')
    loan_amount = request.form.get('loan_amount')
    credit_score = request.form.get('credit_score')
    months_employed = request.form.get('months_employed')
    num_credit_lines = request.form.get('num_credit_lines')
    interest_rate = request.form.get('interest_rate')
    loan_term = request.form.get('loan_term')
    dti_ratio = request.form.get('dti_ratio')
    education = request.form.get('education')
    employment_type = request.form.get('employment_type')
    marital_status = request.form.get('marital_status')
    has_mortgage = request.form.get('has_mortgage')
    has_dependents = request.form.get('has_dependents')
    loan_purpose = request.form.get('loan_purpose')
    has_cosigner = request.form.get('has_cosigner')

    session['name'] = name
    session['age'] = age
    session['income'] = income
    session['loan_amount'] = loan_amount
    session['credit_score'] = credit_score
    session['months_employed'] = months_employed
    session['num_credit_lines'] = num_credit_lines
    session['interest_rate'] = interest_rate
    session['loan_term'] = loan_term
    session['dti_ratio'] = dti_ratio
    session['education'] = education
    session['employment_type'] = employment_type
    session['marital_status'] = marital_status
    session['has_mortgage'] = has_mortgage
    session['has_dependents'] = has_dependents
    session['loan_purpose'] = loan_purpose
    session['has_cosigner'] = has_cosigner

    name = session.get('name', 'N/A')
    age = session.get('age', 'N/A')
    income = session.get('income', 'N/A')
    loan_amount = session.get('loan_amount', 'N/A')
    credit_score = session.get('credit_score', 'N/A')
    months_employed = session.get('months_employed', 'N/A')
    num_credit_lines = session.get('num_credit_lines', 'N/A')
    interest_rate = session.get('interest_rate', 'N/A')
    loan_term = session.get('loan_term', 'N/A')
    dti_ratio = session.get('dti_ratio', 'N/A')
    education = session.get('education', 'N/A')
    employment_type = session.get('employment_type', 'N/A')
    marital_status = session.get('marital_status', 'N/A')
    has_mortgage = session.get('has_mortgage', 'N/A')
    has_dependents = session.get('has_dependents', 'N/A')
    loan_purpose = session.get('loan_purpose', 'N/A')
    has_cosigner = session.get('has_cosigner', 'N/A')

    print(age)

    train_df = pd.read_csv("C:\\Users\\giris\\OneDrive\\Documents\\Python_SS\\python_env\\train.csv")
    
    df = train_df

    le = LabelEncoder()


    new_row = {'LoanID' : 'KNO123' , 'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score, 'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate, 'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': education, 'EmploymentType': employment_type, 'MaritalStatus': marital_status, 'HasMortgage': has_mortgage, 'HasDependents': has_dependents, 'LoanPurpose': loan_purpose ,'HasCoSigner': has_cosigner , 'Default': 0}
    df.loc[len(df)] = new_row
    print(new_row)
    print(1)

    df['Education'] = le.fit_transform(df['Education'])
    df['EmploymentType'] = le.fit_transform(df['EmploymentType'])
    df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
    df['HasMortgage'] = le.fit_transform(df['HasMortgage'])
    df['HasDependents'] = le.fit_transform(df['HasDependents'])
    df['LoanPurpose'] = le.fit_transform(df['LoanPurpose'])
    df['HasCoSigner'] = le.fit_transform(df['HasCoSigner'])

    X_train = df.drop('LoanID', axis=1)   
    X_train = X_train.drop('Default', axis = 1)

    predictor = X_train.iloc[-1:]
    X_train = X_train.iloc[:-1]
    y_train = df['Default']
    y_train = y_train.iloc[:-1]


    model = RandomForestClassifier(random_state=39)

    print("model fit")

    model.fit(X_train, y_train)

    print("Done fit")

    y_pred = model.predict(predictor)

    final = y_pred[0]

    if final == 0:
        a = "ELIGIBLE"
    else:
        a = "NOT ELIGIBLE"
    

    return render_template('result.html', a = a , name = name)


if __name__ == '__main__':
    app.run(debug=True)
