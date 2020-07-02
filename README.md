# transaction_fraud_detection
# Model Training 
1. Modify the `TRAIN_DATA_DIR` in `server/test_fraud.py` as your training data directory
2. Run `server/test_fraud.py`  for training and persisting the model 

# Fraud API
1. run `fraud_api.py`
2. send `GET` request to the `localhost:5000/fraud` with json request body as the example in `server/testconfig.py`