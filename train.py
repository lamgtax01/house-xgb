# We will start with the data pre-loaded into train_X, test_X, train_y, test_y.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import os
import boto3

train_file = 'train.csv'
s3_bucket = 'houses-xgb-01'
s3 = boto3.client('s3')
with open(train_file, 'wb') as f:
    s3.download_fileobj(s3_bucket, train_file, f)

for f in os.listdir():
    print(f)

data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


# We build and fit a model just as we would in scikit-learn.
from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)
model_save_dir = f"{os.environ.get('SM_MODEL_DIR')}"

# make predictions
predictions1 = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Preditions after the training ...................................:")
print("Mean Absolute Error : " + str(mean_absolute_error(predictions1, test_y)))
print("Mean Squared Eerror : " + str(mean_squared_error(predictions1, test_y)))
print("R2 Score : " + str(r2_score(predictions1, test_y)))


# Save the model in JSON Format
print("Save the model in Universal Binary JSON format (.ubj) ...")
print("model_save_dir:", model_save_dir)
my_model.save_model(os.path.join(model_save_dir, "model.ubj"))


# Load the Model from JSON and print the metric
print("Load the model from Universal Binary JSON  format (.ubj) ...")
from xgboost import XGBRegressor
model_xgb_json = XGBRegressor()
model_xgb_json.load_model(os.path.join(model_save_dir, "model.ubj"))

predictions2 = my_model.predict(test_X)

print("Preditions after Save and Reload the model in JSON ...............:")
print("Mean Absolute Error : " + str(mean_absolute_error(predictions2, test_y)))
print("Mean Squared Eerror : " + str(mean_squared_error(predictions2, test_y)))
print("R2 Score : " + str(r2_score(predictions2, test_y)))

