from dash import Dash, html, dcc
import dash
from dash.dependencies import Input, Output
import data_processing

external_css = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"]

# Initialize the dash app
app = dash.Dash(__name__, external_stylesheets=external_css)

# Fetching Titanic dataset
titanic_data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
X_train, X_test, y_train, y_test = data_processing.preprocess_data(titanic_data_url)
model = data_processing.train_model(X_train, y_train)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Titanic Survival Prediction"),

    html.Div([
        html.Label("Select Age:"),
        dcc.Input(id='age-input', type='number')
    ]),

    html.Div([
        html.Label("Select Gender:"),
        dcc.Dropdown(id='gender-dropdown', options=[
            {'label': 'Male', 'value': 0},
            {'label': 'Female', 'value': 1}
        ])
    ]),

    html.Div([
        html.Label("Select Class:"),
        dcc.Dropdown(id='class-dropdown', options=[
            {'label': 'First', 'value': 1},
            {'label': 'Second', 'value': 2},
            {'label': 'Third', 'value': 3}
        ])
    ]),

    html.Div([
        html.Label("Select Number of Siblings/Spouses Aboard:"),
        dcc.Input(id='sibsp-input', type='number')
    ]),

    html.Div([
        html.Label("Select Number of Parents/Children Aboard:"),
        dcc.Input(id='parch-input', type='number')
    ]),

    html.Button('Predict', id='predict-button'),
    html.Div(id='prediction-output')
])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=5002)
