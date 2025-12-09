Titanic Survival Prediction — End-to-End ML Pipeline

This project builds a complete machine learning workflow to predict passenger survival on the Titanic dataset. It includes modular preprocessing, feature engineering, model training, evaluation, artifact saving, and a Streamlit web application for interactive inference.

titanic_set/
├── data/               # raw data
├── models/             # saved pipeline + model artifacts
├── notebooks/          # EDA and experiments
├── reports/            # metrics, plots, predictions
├── scripts/            # training + batch inference scripts
├── src/                # modular ML code (features, preprocessing, training)
└── streamlit_app/      # web app for interactive predictions

Train the Model
python3 -m src.train_model

This runs the preprocessing pipeline, trains Logistic Regression and Random Forest, evaluates them, and saves:
	•	pipeline.joblib
	•	rf.joblib
	•	Metrics, confusion matrices, ROC curves

Batch Predictions
python3 scripts/predict.py

Outputs:
reports/predictions.csv

Run the Streamlit App
streamlit run streamlit_app/app.py
Features:
	•	Single passenger input and prediction
	•	Preview of processed input
	•	Bulk prediction for test.csv

Key Features
	•	Modular preprocessing and feature engineering (src/)
	•	Title extraction, grouped Age imputation, OHE encoding, scaling
	•	Pipeline + model artifact saving with joblib
	•	Evaluation reports and visualizations
	•	Real-time inference UI using Streamlit
Next Enhancements
	•	FastAPI inference service
	•	Docker containerization
	•	CI/CD workflow
	•	Model versioning
	•	Hyperparameter optimization

Author

Vishal Singh


