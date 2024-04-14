from dash import Dash, html, dcc
import dash
from dash.dependencies import Input, Output
import data_processing


# Initialize the dash app
app = dash.Dash(__name__, external_stylesheets=['./assets/styles.css'])

# Fetching Titanic dataset
titanic_data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
X_train, X_test, y_train, y_test = data_processing.preprocess_data(titanic_data_url)
model = data_processing.train_model(X_train, y_train)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Titanic Survival Prediction", className='header'),

    html.Div([
        html.Label("Select Age:"),
        dcc.Input(id='age-input', type='number')
    ], className='component-div'),

    html.Div([
        html.Label("Select Gender:"),
        dcc.Dropdown(id='gender-dropdown', options=[
            {'label': 'Male', 'value': 0},
            {'label': 'Female', 'value': 1}
        ])
    ],className='component-div'),

    html.Div([
        html.Label("Select Class:"),
        dcc.Dropdown(id='class-dropdown', options=[
            {'label': 'First', 'value': 1},
            {'label': 'Second', 'value': 2},
            {'label': 'Third', 'value': 3}
        ])
    ],className='component-div'),

    html.Div([
        html.Label("Select Number of Siblings/Spouses Aboard:"),
        dcc.Slider(
            id='sibsp-slider',
            min=0,
            max=10,
            step=1,
            value=0,
            marks={i: str(i) for i in range(11)}  # Tworzenie etykiet dla suwaka
        )
    ],className='component-div re-slider-mark-text'),

    html.Div([
        html.Label("Select Number of Parents/Children Aboard:"),
        dcc.Slider(
            id='parch-slider',
            min=0,
            max=10,
            step=1,
            value=0,
            marks={i: str(i) for i in range(11)}  # Tworzenie etykiet dla suwaka
        )
    ],className='component-div re-slider-mark-text'),

    html.Button('Predict', id='predict-button', className='button-center'),
    html.Br(),
    html.Div(id='prediction-output', className='result-div',)
], className='main-div')

# Define the callback to predict survival
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('age-input', 'value'),
     Input('gender-dropdown', 'value'),
     Input('class-dropdown', 'value'),
     Input('sibsp-slider', 'value'),  # Poprawiono identyfikator na 'sibsp-slider'
     Input('parch-slider', 'value')]  # Poprawiono identyfikator na 'parch-slider'
)
def predict_survival(n_clicks, age, sex, pclass, sibsp, parch):
    if n_clicks is not None:
        data = [[pclass, sex, age, sibsp, parch, 0, 0]]  # Assuming Embarked and Fare are not used for prediction
        survival_prob = model.predict_proba(data)[0][1]  # Probability of belonging to class 'Survived'
        # Set text color based on survival probability
        color = 'green' if survival_prob > 0.7 else 'red'
        # Return styled text with color
        return html.Span(f"Survival Probability: {survival_prob:.2%}", style={'color': color, 'font-weight': 'bold', 'font-size': '32px'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=5004)
