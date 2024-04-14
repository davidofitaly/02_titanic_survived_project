from dash import Dash, html, dcc
import dash
from dash.dependencies import Input, Output
import data_processing


# Initialize the dash app
app = dash.Dash(__name__, external_stylesheets=['styles.css'])

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

# Define the callback to predict survival
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('age-input', 'value'),
     Input('gender-dropdown', 'value'),
     Input('class-dropdown', 'value'),
     Input('sibsp-input', 'value'),
     Input('parch-input', 'value')]
)
def predict_survival(n_clicks, age, sex, pclass, sibsp, parch):
    if n_clicks is not None:
        data = [[pclass, sex, age, sibsp, parch, 0, 0]]  # Assuming Embarked and Fare are not used for prediction
        survival_prob = model.predict_proba(data)[0][1]  # Probability of belonging to class 'Survived'
        return f"Survival Probability: {survival_prob:.2%}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=5002)
