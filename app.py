# Import necessary modules
from dash import Dash, html, dcc
import dash
from dash.dependencies import Input, Output, State
import data_processing

# Initialize the Dash application
app = dash.Dash(__name__, external_stylesheets=['./assets/styles.css'])

# Fetch Titanic dataset
titanic_data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
X_train, X_test, y_train, y_test = data_processing.preprocess_data(titanic_data_url)
model = data_processing.train_model(X_train, y_train)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Titanic Survival Prediction", className='header'),  # Header of the application

    # Age input field
    html.Div([
        html.Label("Write Age:"),  # Label for the age input field
        dcc.Input(id='age-input', type='number', style={'width': '10%', 'height': '25px', 'font-size': '16px', 'font-weight': 'bold', 'border-radius': '5px', 'border':'0.5px solid grey'}),  # Age input field
    ], className='component-div'),

    # Gender dropdown selection
    html.Div([
        html.Label("Select Gender:"),  # Label for gender dropdown selection
        dcc.Dropdown(id='gender-dropdown', options=[  # Gender dropdown selection
            {'label': 'Male', 'value': 0},
            {'label': 'Female', 'value': 1}
        ],
        placeholder='Select Gender...',
        style={'font-size': '14px'})
    ], className='component-div'),

    # Class dropdown selection
    html.Div([
        html.Label("Select Class:"),  # Label for class dropdown selection
        dcc.Dropdown(id='class-dropdown', options=[  # Class dropdown selection
            {'label': 'First', 'value': 1},
            {'label': 'Second', 'value': 2},
            {'label': 'Third', 'value': 3}
        ],
        placeholder='Select Class...',
        style={'font-size': '14px'})
    ], className='component-div'),

    # Sibsp slider selection
    html.Div([
        html.Label("Select Number of Siblings/Spouses Aboard:"),  # Label for Sibsp slider selection
        dcc.Slider(
            id='sibsp-slider',
            min=0,
            max=10,
            step=1,
            value=0,
            marks={i: str(i) for i in range(11)},
            tooltip={'placement': 'bottom', 'always_visible': True, 'style':{'fontSize': '15px'}},
            className='custom-slider'
        )
    ], className='component-div re-slider-mark-text'),

    # Parch slider selection
    html.Div([
        html.Label("Select Number of Parents/Children Aboard:"),  # Label for Parch slider selection
        dcc.Slider(
            id='parch-slider',
            min=0,
            max=10,
            step=1,
            value=0,
            marks={i: str(i) for i in range(11)},
            tooltip={'placement': 'bottom', 'always_visible': True, 'style':{'fontSize': '15px'}},
            className='custom-slider'
        )
    ], className='component-div re-slider-mark-text'),

    # Predict button
    html.Button('Predict', id='predict-button', className='button-center'),
    html.Br(),
    html.Div(id='prediction-output', className='result-div',)
], className='main-div')

# Define the callback to predict survival
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('age-input', 'value'),
     State('gender-dropdown', 'value'),
     State('class-dropdown', 'value'),
     State('sibsp-slider', 'value'),
     State('parch-slider', 'value')]
)
def predict_survival(n_clicks, age, sex, pclass, sibsp, parch):
    if n_clicks is not None:  # Check if the "Predict" button is clicked
        data = [[pclass, sex, age, sibsp, parch, 0, 0]]  # Assuming Embarked and Fare are not used for prediction
        survival_prob = model.predict_proba(data)[0][1]  # Probability of belonging to class 'Survived'
        # Set text color based on survival probability
        color = '#35764B' if survival_prob > 0.7 else 'red'
        # Return styled text with color
        return html.Span(f"Survival Probability: {survival_prob:.2%}", style={'color': color, 'font-weight': 'bold', 'font-size': '32px'})
    else:
        raise dash.exceptions.PreventUpdate  # Stop callback if button is not clicked


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=5004)
