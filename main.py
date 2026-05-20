import pandas as pd
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

#pip install scikit-learn
# Progress Bar
def progress_bar(task, steps=30, delay=0.03):

    print(f"\n{task}")

    for i in range(steps + 1):

        percent = int((i / steps) * 100)

        bar = "█" * i + "-" * (steps - i)

        sys.stdout.write(
            f"\r[{bar}] {percent}%"
        )

        sys.stdout.flush()

        time.sleep(delay)

    print(" Done ✓")


print("Loading Dataset...")
progress_bar("Reading Data")

df = pd.read_csv(
    "car_prediction_data.csv"
)

# Feature Engineering
current_year = 2025

df["Car_Age"] = (
    current_year -
    df["Year"]
)

# Target = depreciation
df["Depreciation"] = (
    df["Present_Price"] -
    df["Selling_Price"]
)

df.drop(
    "Year",
    axis=1,
    inplace=True
)

print(
    "\nRecords:",
    len(df)
)

progress_bar(
    "Preprocessing Data"
)

X = df.drop(
    ["Selling_Price", "Depreciation"],
    axis=1
)

y = df["Depreciation"]


cat_cols = [

    "Car_Name",
    "Fuel_Type",
    "Seller_Type",
    "Transmission"

]

num_cols = [

    "Present_Price",
    "Kms_Driven",
    "Owner",
    "Car_Age"

]


preprocessor = ColumnTransformer(
    transformers=[

        (
            "cat",

            OneHotEncoder(
                handle_unknown='ignore'
            ),

            cat_cols
        ),

        (
            "num",

            "passthrough",

            num_cols
        )

    ]
)


model = Pipeline([

    (
        "prep",
        preprocessor
    ),

    (
        "reg",
        LinearRegression()
    )

])


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)


progress_bar(
    "Training Model"
)

model.fit(
    X_train,
    y_train
)


progress_bar(
    "Evaluating Model"
)

y_pred = model.predict(
    X_test
)


print("\n===== MODEL LOGS =====")

print(
    "R2 Score:",
    round(
        r2_score(
            y_test,
            y_pred
        ),
        3
    )
)

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_test,
            y_pred
        ),
        3
    )
)


print(
    "\n===== FIND CARS ====="
)

low = float(
    input(
        "Minimum Budget: "
    )
)

high = float(
    input(
        "Maximum Budget: "
    )
)

progress_bar(
    "Searching Cars",
    20
)


# Filter by price range
cars = df[
    (df["Selling_Price"] >= low)
    &
    (df["Selling_Price"] <= high)
]


# Keep full row + sort by lower KM first
cars = cars.sort_values(
    by="Kms_Driven",
    ascending=True
).reset_index(drop=True)


if len(cars) == 0:

    print(
        "\nNo cars found"
    )

    exit()


print(
    "\nAvailable Cars (Lower KM first):\n"
)


for i, row in cars.iterrows():

    print(

        i + 1,

        ".",

        row["Car_Name"],

        "-",

        round(
            row["Selling_Price"],
            2
        ),

        "Lakhs",

        "| KM Driven:",

        int(
            row["Kms_Driven"]
        )

    )


choice = int(
    input(
        "\nSelect car number: "
    )
)


selected = cars.iloc[
    choice - 1
]


annual_run = int(
    input(
        "\nAnnual Running (km/year): "
    )
)


progress_bar(
    "Generating 5-Year Forecast",
    25
)


print(
    "\n===== NEXT 5 YEAR FORECAST ====="
)


base_age = selected[
    "Car_Age"
]

base_kms = selected[
    "Kms_Driven"
]

present = selected[
    "Present_Price"
]


for year in range(1,6):

    future_kms = (

        base_kms +

        annual_run * year

    )


    temp = pd.DataFrame([{

        "Car_Name":
        selected["Car_Name"],

        "Present_Price":
        selected["Present_Price"],

        "Kms_Driven":
        future_kms,

        "Fuel_Type":
        selected["Fuel_Type"],

        "Seller_Type":
        selected["Seller_Type"],

        "Transmission":
        selected["Transmission"],

        "Owner":
        selected["Owner"],

        "Car_Age":
        base_age + year

    }])


    # Uses TRAINED MODEL
    predicted_dep = model.predict(
        temp
    )[0]


    future_value = (

        present -

        predicted_dep

    )


    if future_value < 0:

        future_value = 0


    print(
        "\nYear:",
        year
    )

    print(
        "Car Age:",
        base_age + year
    )

    print(
        "KM Driven:",
        int(
            future_kms
        )
    )

    print(
        "Predicted Depreciation:",
        round(
            predicted_dep,
            2
        ),
        "Lakhs"
    )

    print(
        "Estimated Car Value:",
        round(
            future_value,
            2
        ),
        "Lakhs"
    )


print(
    "\nForecast Completed ✓"
)