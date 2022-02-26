import streamlit as st
import pandas as pd
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from numpy import absolute, mean, std
from sklearn import metrics
import matplotlib.patches as mpatches

data_fold = "data/"
df = pd.read_csv(data_fold+"Train_Data.csv")
orig_df = pd.read_csv(data_fold+"Train_Data.csv")

#convert categorical to numerical
df["smoker"].replace({"yes": 1, "no": 0}, inplace=True)
dummy_gender = pd.get_dummies(df.sex, prefix='gender')
dummy_region = pd.get_dummies(df.region, prefix = "region")
# print(dummy_gender)

df = df.drop(columns=['region', 'sex'])
region_columns = ["region_southwest", "region_southeast", "region_northwest", "region_northeast"]
gender_columns = ["gender_female", "gender_male"]
for c in region_columns:
    df[c] = dummy_region[c]
for c in gender_columns:
    df[c] = dummy_gender[c]


filename = "data/insurance_model"
infile = open(filename,'rb')
model = pickle.load(infile)
infile.close()

cols = ["age", "bmi", "smoker", "children", "region_southwest", "region_southeast", "region_northwest", "region_northeast", "gender_female", "gender_male"]
def prediction(Gender, Smoker, BodyMass, Region, Children, Age):
    df = pd.DataFrame(columns = cols)
    row = []
    row.append(Age)
    row.append(BodyMass)
    if Smoker == "Yes":
        row.append(1)
    elif Smoker == "No":
        row.append(0)
    row.append(Children)
    if Region =="Southwest":
        row.append(1)
        row.append(0)
        row.append(0)
        row.append(0)
    elif Region =="Southeast":
        row.append(0)
        row.append(1)
        row.append(0)
        row.append(0)
    elif Region =="Northwest":
        row.append(0)
        row.append(0)
        row.append(1)
        row.append(0)
    elif Region =="Northeast":   
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(1) 
    if Gender == "Male":
        row.append(0)
        row.append(1)
    elif Gender == "Female":
        row.append(1)
        row.append(0)

    df.loc[0] = row

    pred = model.predict(df)
    
    return pred




header = st.container()

data = st.container()

eda = st.container()

predictor = st.container()

with header:
    st.title("Personal Medical Costs App")
    st.write("""This web app explores annual personal insurance premium charges and allows for users to estimate their premium cost based on a few features. 
    The dataset is from the book Machine Learning with R by Brett Lantz and uses simulated data extracted from 2016 Census Records""")

with data:
    st.header("The Data")
    st.write("The dataset contains 7 columns giving details of the subject and 3630 subject entries. Below is a snippet of the dataset.")
    st.dataframe(orig_df.head(45))

with eda:
    st.header("Exploratory Data Analysis")
    st.subheader("Charges Distribution")
    st.write("""First we will explore the distribution of the charges. The charges indicate the total of annual premiums each subject paid for in medical expenses.
    From the histogram we can observe that the data is positively skewed. The median values lay near the $10,000 mark.""")
    fig, ax = plt.subplots(2,1)
    # charges_stats = orig_df.charges.describe()
    ax[0].hist(orig_df.charges)
    # print(charges_stats)
    ax[1].boxplot(orig_df.charges, vert=False)
    st.pyplot(fig)
    st.subheader("Smoker Analysis")
    st.write(""" Next we will explore the smoker variable. This variable asks if a subject is a smoker or not (answers being yes or no).  """)

    fig = plt.figure()
    total = orig_df.copy()
    total['count'] = np.ones(shape =(total.shape[0]))
    total = total.groupby('smoker')['count'].count().reset_index()
    # st.dataframe(total)
    sns.barplot(x="smoker", y="count", data =total, color="lightblue")
    st.pyplot(fig)

    st.write(""" The majority of subjects are non-smokers and smokers make up a small percentage of the total dataset. Digging deeper, we'll compare the number of smokers depending on their genders.""")
    
    fig = plt.figure()
    total = orig_df.copy()
    total['count'] = np.ones(shape =(total.shape[0]))
    total = total.groupby('sex')['count'].count().reset_index()
    sns.barplot(x="sex",  y="count", data=total, color='darkblue')
    smoker = orig_df[orig_df.smoker=='yes']
    smoker['count'] = np.ones(shape =(smoker.shape[0]))
    # # bar chart 2 -> bottom bars (group of 'smoker=Yes')
    sns.barplot(x="sex", y="count", data=smoker, estimator=sum, ci=None,  color='lightblue')
    top_bar = mpatches.Patch(color='darkblue', label='smoker = no')
    bottom_bar = mpatches.Patch(color='lightblue', label='smoker = yes')
    plt.legend(handles=[top_bar, bottom_bar])
    st.pyplot(fig)

    st.write(""" Here we observe that the their are 1/2 as many female smokers as male smokers. We can also observe that the total overall gender split is not 50:50 like the smoker genders.
    We can see that there is approximately a 45:55 gender split of men to women. Although slightly different, 
    the distribution of males and females in the smoker subset are still relatively close to the actual distribution.""")

    st.subheader("BMI Distribution")
    st.write("BMI stands for Body Mass Index and is calculated from an individual's weight and height. We plot the distribution of BMI's:")
    fig, ax = plt.subplots(2,1)
    ax[0].hist(orig_df.bmi, bins = 20)
    ax[1].boxplot(orig_df.bmi, vert=False)
    st.pyplot(fig)

    st.write(""" The observed distribution is normal with its median and mean approximately at 30""")

    st.subheader("Age Distribution")
    st.write("""Lastly we'll observe the distribution of ages amongst the dataset. The ages are somewhat irregularly distributed with a 
    large peak at around the late teens (17-20) and late 40's""")
    fig, ax = plt.subplots(2,1)
    ax[0].hist(orig_df.age, bins = 20)
    ax[1].boxplot(orig_df.age, vert=False)
    st.pyplot(fig)

    st.write("""From here we have strong foundations to compare charges to BMI and age. """)
    st.subheader("Charges vs. BMI")
    st.write(""" We plot BMI vs charges below on a scatter plot:""")

    fig = plt.figure()
    plt.scatter(orig_df.bmi, orig_df.charges)
    st.pyplot(fig)

    st.write(""" There is not a strong relationship between the two factors but we can analyze the data further by using colors to differentiate some categorical variables on the scatter plot.""")

    fig, ax = plt.subplots(2,1)
    smoker_bin = orig_df.smoker.map({'yes': "green", 'no': "red"})
    sex_color = orig_df.sex.map({"male": "blue", "female": "pink"})
    ax[1].scatter(orig_df.bmi, orig_df.charges, c=smoker_bin)
    ax[0].scatter(orig_df.bmi, orig_df.charges, c=sex_color)
    st.pyplot(fig)
    st.write("""Not a lot of conclusions can be extracted from the scatter plot colored by sex, but when we observe the plot colorwed by 
    smoker status we can see a clear relationship between charges and BMI for smokers. """)

    st.subheader("Charges vs Age")
    st.write(""" We want to observe the relationship between charges and age. We will color subjects by their sex and smoker status to see if we can define a relationship in these categories also.""")

    fig, ax = plt.subplots(3,1)
    ax[0].scatter(orig_df.age, orig_df.charges)
    ax[2].scatter(orig_df.age, orig_df.charges, c = smoker_bin)
    ax[1].scatter(orig_df.age, orig_df.charges, c = sex_color)

    st.pyplot(fig)
    st.write(""" Again, there is not a strong relationship between overall charges and age even when colored by sex. When we color by
    by smoker status however, we can see a relationship forming. This means we can try to understand the relationship between charges and 
    smoker status. """)

with predictor:
    st.header("Annual Medical Insurance Cost Predictor")
    Gender = st.radio('Gender',["Male","Female"])
    Smoker= st.radio('Smoker',["Yes","No"]) 
    BodyMass = st.number_input("Body Mass Index", step=0.1) 
    Region= st.radio('Region (US)',["Northeast","Northwest", "Southeast", "Southwest"])
    Children = st.number_input('Number of children', min_value=0, max_value=5, value=0, step=1)
    Age = st.number_input('Age', min_value=0, max_value=80, value=0, step=1)
    result =""
    if st.button("Predict"):
        st.header(f"Insurance Charge: ${prediction(Gender, Smoker, BodyMass, Region, Children, Age)[0]:,.2f}")
