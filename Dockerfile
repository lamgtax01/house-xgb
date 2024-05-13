#Download XGBoost Docker image
FROM python:3.10

# Install scikit-learn
RUN pip3 install -U scikit-learn

# Install xgboost
RUN pip3 install xgboost

# Install xgboost
RUN pip3 install pandas

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py