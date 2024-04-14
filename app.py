from dash import Dash, html, dcc
import dash
from dash.dependencies import  Input, Output
import data_processing

external_css = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" ]

# Initialize the dash app
app = dash.Dash(__name__)

# Fetching Titanic dataset
titanic_data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
X_train, X_test, y_train, y_test = data_processing.preprocess_data(titanic_data_url)
model = data_processing.train_model(X_train, y_train)

