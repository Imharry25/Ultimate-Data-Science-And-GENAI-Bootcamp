import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


train_path = "ML/boston-housing/train.csv"
test_path = "ML/boston-housing/test.csv"

train_df = pd.read_csv(train_path)

train_df.columns.to_list()

FEATURE_COLUMNS = [
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 
    'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat'
]

train_df.head()

train_df.shape

def basic_summary(df):
    summary_info = {
        "missing_values": df.isnull().sum(),
        "dtypes": df.dtypes,
        "stats": df.describe().T
    }
    print(summary_info)

basic_summary(train_df)


plt.figure(figsize=(7,5))
sns.histplot(train_df["medv"], kde=True, bins=30)
plt.title("distribution of Target (medv)")
plt.xlabel("Median Value of Homes ($1000s)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(12,8))
corr = train_df.drop(columns=["ID"]).corr()
sns.heatmap(corr, annot = True, cmap= "coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()


train_df.drop(columns=["ID"]).hist(figsize=(15,12), bins=30)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()


numeric_cols = train_df.drop(columns= ["ID","chas","medv"]).columns
plt.figure(figsize=(15,10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4 , 3, i)
    sns.boxplot(x=train_df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Features and target
X_train_full = train_df.drop(columns=["ID","medv"])
y_train_full = train_df['medv']


# Perform a Train-Validation Split for model evaluation (80/20 split)
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)


numeric_features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'black', 'lstat']
categorical_features = ['chas', 'rad']


numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')


# Create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder ='passthrough'
)

X_train_processed = preprocessor.fit_transform(X_train_split)
X_test_processed = preprocessor.transform(X_test_split)

print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_test_processed shape: {X_test_processed.shape}")


# Train Model using SGDREGRESSOR
sgd_reg = SGDRegressor(
    max_iter = 1000,
    eta0=0.01, # Learning rate
    random_state=42
)

# IN ML ALGO -  we only have fit & predict


sgd_reg.fit(X_train_processed, y_train_split)

# 20 features --> y = MX + C 
# M --> 20m
# C --> 1

sgd_reg.coef_, sgd_reg.intercept_

len(sgd_reg.coef_)

# OLS
linear_reg_ols = LinearRegression()
linear_reg_ols.fit(X_train_processed, y_train_split)


#Test Model

# With SGD
y_test_pred = sgd_reg.predict(X_test_processed)
y_train_pred = sgd_reg.predict(X_train_processed)

# with OLS
y_test_pred_ols = linear_reg_ols.predict(X_test_processed)
y_train_pred_ols = linear_reg_ols.predict(X_train_processed)


def plot_actual_vs_predicted_lines(y_actual, y_predicted):
    plt.figure(figsize=(14,7))

    if isinstance(y_actual, pd.Series):
        y_actual = y_actual.values

    plt.plot(y_actual, label='Actual Values', color='#1f77b4', linewidth=2)

    plt.plot(y_predicted, label='Predicted Values', color='#ff7f0e', linewidth=2, linestyle='--')

    plt.title('Actual  vs Predicted Values Over Index (Trend Comparison)')
    plt.xlabel('Data Index')
    plt.ylabel('Median Home Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


plot_actual_vs_predicted_lines(y_test_split, y_test_pred)

plot_actual_vs_predicted_lines(y_test_split, y_test_pred_ols)

# Evaluation
#Evaluation
from sklearn.metrics import mean_squared_error

mse_train = mean_squared_error(y_train_split, y_train_pred)
mse_test = mean_squared_error(y_test_split, y_test_pred)

print(f"MSE (TRAIN) (SGD) : {mse_train}")
print(f"MSE (TEST) (SGD): {mse_test}")

mse_train_ols = mean_squared_error(y_train_split, y_train_pred_ols)
mse_test_ols = mean_squared_error(y_test_split, y_test_pred_ols)

print(f"MSE (TRAIN) (OLS) : {mse_train_ols}")
print(f"MSE (TEST) (OLS) : {mse_test_ols}")

# Save Artefacts

import pickle

model_filename = 'final_sgd_reg_model.pkl'
preprocessor_filename = 'data_preprocessor.pkl'

with open(model_filename, "wb") as f:
    pickle.dump(sgd_reg, f)
with open(preprocessor_filename, "wb") as f:
    pickle.dump(preprocessor, f)


print(f"\nModel saved to: {model_filename}")
print(f"Preprocessor saved to: {preprocessor_filename}")

def make_prediction_from_list(input_list, feature_names, loaded_model, loaded_preprocessor):

    X_input = pd.DataFrame([input_list], columns=feature_names)
    print(f"\nInput DataFrame:\n{X_input}")
    
    X_input_processed = loaded_preprocessor.transform(X_input)

    predictions = loaded_model.predict(X_input_processed)

    return predictions[0]

model_filename = 'final_sgd_reg_model.pkl'
preprocessor_filename = 'data_preprocessor.pkl'


with open(model_filename, "rb") as f:
    loaded_model = pickle.load(f)
with open(preprocessor_filename, "rb") as f:
    loaded_preprocessor = pickle.load(f)


print("Artifacts loaded successfully.")


test_features_list = [
    50,    # crim
    0,     # zn
    18.1,  # indus
    0,     # chas
    0.7,   # nox
    4.0,   # rm 
    100,   # age
    1.0,   # dis 
    24,    # rad
    666,   # tax
    20.2,  # ptratio
    300,   # black
    30     # lstat 
]

# 13 columns


FEATURE_COLUMNS = [
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 
    'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat'
]


predicted_value_extreme = make_prediction_from_list(
    input_list=test_features_list,
    feature_names=FEATURE_COLUMNS,
    loaded_model=loaded_model,
    loaded_preprocessor=loaded_preprocessor
)

print(f"\n\nPredicted medv: {predicted_value_extreme:.2f}")

import pickle
import numpy as np

class PredictHousingMEDV:
    def __init__(self, model_path, preprocessor_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)
        self.feature_names = [
            "crim", "zn", "indus", "chas", "nox", "rm",
            "age", "dis", "rad", "tax", "ptratio", "black", "lstat"
        ]
        self.ranges = {
            "crim": (0, 100),
            "zn": (0, 100),
            "indus": (0, 30),
            "chas": (0, 1),
            "nox": (0.3, 1.0),
            "rm": (3.0, 9.0),
            "age": (0, 100),
            "dis": (1, 15),
            "rad": (1, 24),
            "tax": (100, 800),
            "ptratio": (10, 25),
            "black": (0, 400),
            "lstat": (1, 40)
        }

    def predict(self):
        input_values = []
        for feature in self.feature_names:
            frange = self.ranges[feature]
            prompt = f"Please enter value for {feature} ({frange[0]} - {frange[1]}): "
            while True:
                try:
                    value = float(input(prompt))
                    if frange[0] <= value <= frange[1]:
                        input_values.append(value)
                        break
                    else:
                        print(f"Value out of range for {feature}, try again.")
                except ValueError:
                    print("Invalid input, enter a number.")
        x_df = pd.DataFrame([input_values], columns=self.feature_names)
        x_processed = self.preprocessor.transform(x_df)
        pred = self.model.predict(x_processed)
        print(f"Predicted MEDV (housing value): {pred[0]:.2f}")



if __name__ == "__main__":
    model_path = "final_sgd_reg_model.pkl"         # Change to your pickle path if needed
    preprocessor_path = "data_preprocessor.pkl"    # Change to your pickle path if needed

    phm = PredictHousingMEDV(model_path, preprocessor_path)
    phm.predict()