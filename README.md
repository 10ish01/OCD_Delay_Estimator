
Please install Python(3.8+) and any code editor like Visual studio Code in your machine to use this app and access or modify the source code.

To install the dependencies for this app, navigate to the OCD_Delay_Estimator folder in your terminal and run:
pip install -r requirements.txt

To access the app, in your terminal, run:  streamlit run app.py

Your browser will open the app typically at a link like:  http://localhost:8501

To make any changes to the UI of the app, you can modify 'app.py'

Similarly to make any changes to the model pipeline, you can modify 'train_model.py'

To access the simple delay estimator tool independently, in your terminal run:  python simple_delay.py

You can pin the link to your browser to access the app on the go.

In case you wish to convert this into a standalone Executable (.exe) (Windows only), you can package the app into a single .exe file by following these steps-

Install PyInstaller: pip install pyinstaller

Create a script launch_app.py, copy the following code in it:
import os
os.system("streamlit run app.py")

Package it:
pyinstaller --noconsole --onefile launch_app.py

It will generate dist/launch_app.exe. Distribute this .exe with your app.py, models, and assets.

Limitations: Streamlit still opens in a browser. You can't embed it into a traditional desktop GUI window without advanced workarounds, like Electron, which can handle the UI window.

How to Run (Executable)

Double-click launch_app.exe.

Your browser will open:
http://localhost:8501

(Keep all files in the same folder. Do not rename or move them)


Structure of the main directory:

OCD_Delay_Predictor/
├── app.py                      # Main UI
├── train_model.py              # Training pipeline script
├── requirements.txt            # List of Python packages
├── dataset/                    # Folder containing data CSVs uploaded by the user
│   └── dataset1.csv, ...
├── simple_delay.py             # A standalone script for simple delay estimation tool
├── data/
│   ├── breakdowns.csv          # Breakdown logs (timestamped)
│   ├── train_data.csv          # Preprocessed training data 
│   ├── test_data.csv           # Test-wise aggregate features
│   ├── data_info.txt               # Summary of last uploaded file   
├── model/
│   ├── classifier.pkl          # Trained XGBoost classifier
│   ├── regressor.pkl           # Trained CatBoost regressor
│   ├── classifier_prev.pkl     # Previously trained XGBoost classifier
│   ├── regressor_prev.pkl      # Previously trained CatBoost regressor
│   └── model_metrics.txt       # Performance summary (classification & regression)
├── assets/
│   ├── confusion_matrix.png    # Classifier confusion matrix
│   ├── shap_plot.png           # SHAP feature importance
│   └── actual_vs_predicted.png # Regression scatter plot
    └── image.png               # Expected format for uploaded files
    └── logo.png                # Company logo


If you need any further help, you can contact me at: iamtanishhora@gmail.com