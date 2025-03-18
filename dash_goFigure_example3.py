import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Create a Dash app
app = dash.Dash(__name__)

# Initial dataset
datasets = {
    "dataset1": {'categories': ['A', 'B', 'C', 'D'], 'values': [10, 15, 7, 12]},
    "dataset2": {'categories': ['A', 'B', 'C', 'D'], 'values': [5, 10, 15, 20]}
}

# Define the app layout
app.layout = html.Div([
    html.H1("Interactive Line & Pie Charts"),
    
    # Dropdown for dataset selection
    dcc.Dropdown(
        id='data-selector',
        options=[
            {'label': 'Dataset 1', 'value': 'dataset1'},
            {'label': 'Dataset 2', 'value': 'dataset2'}
        ],
        value='dataset1',
        clearable=False
    ),
    
    # Sliders for dynamic value adjustments
    html.Div([
        html.Label("Adjust A"),
        dcc.Slider(id='slider-A', min=0, max=30, step=1, value=10, marks={i: str(i) for i in range(0, 31, 5)}),
        
        html.Label("Adjust B"),
        dcc.Slider(id='slider-B', min=0, max=30, step=1, value=15, marks={i: str(i) for i in range(0, 31, 5)}),

        html.Label("Adjust C"),
        dcc.Slider(id='slider-C', min=0, max=30, step=1, value=7, marks={i: str(i) for i in range(0, 31, 5)}),

        html.Label("Adjust D"),
        dcc.Slider(id='slider-D', min=0, max=30, step=1, value=12, marks={i: str(i) for i in range(0, 31, 5)}),
    ]),

    # Graphs for line and pie charts
    html.Div([
        dcc.Graph(id='line-chart', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='pie-chart', style={'width': '48%', 'display': 'inline-block'})
    ])
])

# Callback to update both charts based on dropdown and slider values
@app.callback(
    [Output('line-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('data-selector', 'value'),
     Input('slider-A', 'value'),
     Input('slider-B', 'value'),
     Input('slider-C', 'value'),
     Input('slider-D', 'value')]
)
def update_graphs(selected_dataset, a, b, c, d):
    """Updates the charts based on selected dataset and slider values."""
    
    # Get categories from the selected dataset
    categories = datasets[selected_dataset]['categories']
    
    # Use slider values as updated dataset values
    values = [a, b, c, d]
    
    # Create Line Chart
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=categories, y=values, mode='lines+markers', name=selected_dataset))
    line_fig.update_layout(title="Line Chart", xaxis_title="Category", yaxis_title="Values")

    # Create Pie Chart
    pie_fig = go.Figure()
    pie_fig.add_trace(go.Pie(labels=categories, values=values, name=selected_dataset))
    pie_fig.update_layout(title="Pie Chart")
    
    return line_fig, pie_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
