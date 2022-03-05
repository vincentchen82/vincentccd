#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
import joblib

app = Flask(__name__)

from flask import request, render_template
import joblib

@app.route("/", methods=["GET","POST"])
def index():
    if request.method =="POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        
        income = float(income)
        age = float(age)
        loan = float(loan)
        print(income, age, loan)
        
        model1 = joblib.load("CCD_DT")
        pred1 = model1.predict([[income, age, loan]])
        s1 = "The score of credit card default base on decision tree is " + str(pred1)
        
        model2 = joblib.load("CCD_Reg")
        pred2 = model2.predict([[income, age, loan]])
        s2 = "The score of credit card default base on regression is " + str(pred2)
        
        model3 = joblib.load("CCD_NN")
        pred3 = model3.predict([[income, age,loan]])
        s3 = "The score of credit card default base on neural network is " + str(pred3)
        
        model4 = joblib.load("CCD_RF")
        pred4 = model4.predict([[income, age, loan]])
        s4 = "The score of credit card default base on random forest is " + str(pred4)
        
        model5 = joblib.load("CCD_GB")
        pred5 = model5.predict([[income, age, loan]])
        s5 = "The score of credit card default base on gradient boosting is " + str(pred5)
        
        return(render_template("index.html",result1=s1, result2=s2, result3=s3, result4=s4, result5=s5))
    else:
        return(render_template("index.html",result1="2", result2="2", result3="2", result4="2", result5="2"))

if __name__ == "__main__":
    app.run()


# In[ ]:




