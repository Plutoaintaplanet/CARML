import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

print("Loading dataset...")

# Load data
df = pd.read_csv("car_prediction_data.csv")

print("Dataset loaded successfully")
print("Total records:", len(df))

df = df[['Car_Name',
         'Year',
         'Fuel_Type',
         'Kms_Driven',
         'Transmission',
         'Selling_Price']]

# Encoding
print("\nEncoding categorical features...")

fuel_encoder = LabelEncoder()
trans_encoder = LabelEncoder()

df['Fuel_Type'] = fuel_encoder.fit_transform(df['Fuel_Type'])
df['Transmission'] = trans_encoder.fit_transform(df['Transmission'])

X = df[['Year',
        'Fuel_Type',
        'Kms_Driven',
        'Transmission']]

y = df['Selling_Price']

print("Features selected:")
print(X.columns.tolist())

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("\nTraining samples:",len(X_train))
print("Testing samples:",len(X_test))

print("\nTraining model...")

model=LinearRegression()
model.fit(X_train,y_train)

print("Training completed")

# Metrics
y_pred=model.predict(X_test)

print("\n===== TRAINING LOGS =====")

print("Intercept:",
      round(model.intercept_,3))

print("\nFeature Coefficients:")

for feature,coef in zip(X.columns,
                         model.coef_):
    print(feature,":",
          round(coef,4))

print("\nModel Accuracy")

print("R2 Score:",
      round(r2_score(y_test,y_pred),3))

print("MAE:",
      round(mean_absolute_error(y_test,y_pred),3))

print("\nSample predictions:")

for i in range(5):
    print("Actual:",
          round(y_test.iloc[i],2),
          "| Predicted:",
          round(y_pred[i],2))

# Predict for all cars
df['Predicted_Price']=model.predict(X)

print("\n===== VEHICLE RECOMMENDATION =====")

min_price=float(input("Minimum budget: "))
max_price=float(input("Maximum budget: "))

fuel=input("Fuel Type(Petrol/Diesel/CNG): ")
trans=input("Transmission(Manual/Automatic): ")

fuel=fuel_encoder.transform([fuel])[0]
trans=trans_encoder.transform([trans])[0]

results=df[
(df['Predicted_Price']>=min_price)&
(df['Predicted_Price']<=max_price)&
(df['Fuel_Type']==fuel)&
(df['Transmission']==trans)
]

results=results[['Car_Name',
                  'Predicted_Price']].drop_duplicates()

if len(results)>0:

    print("\nTop vehicle suggestions:\n")

    for i,row in results.head(5).iterrows():

        print(
            row['Car_Name'],
            "-",
            round(row['Predicted_Price'],2),
            "Lakhs"
        )

else:
    print("No matching vehicles found")