import pandas as pd
import numpy as np

# --- Sklearn Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score, classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# --- Dash and Plotly Imports ---
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Create a Dash app
app = dash.Dash(__name__)
server = app.server #required for render deployment

# --------------------------
# Load and Preprocess Data
# --------------------------
df = pd.read_csv("synthetic_cancer_data.csv")

# Define numeric columns and treat any negative values as missing
check_columns = ['Age', 'Total_Mutations', 'CEA_Level', 'AFP_Level',
                 'WBC_Count', 'CRP_Level', 'Tumor_Size', 'Tumor_Density']
df[check_columns] = df[check_columns].where(df[check_columns] >= 0, pd.NA)

# Imputation (numeric: median, categorical: most frequent)
median_cols = check_columns
mode_cols = ['Sex', 'Smoking_Status', 'Family_History', 'TP53_Mutation',
             'BRCA1_Mutation', 'KRAS_Mutation', 'Tumor_Location']

median_imputer = SimpleImputer(strategy='median')
mode_imputer = SimpleImputer(strategy='most_frequent')

df[median_cols] = median_imputer.fit_transform(df[median_cols])
df[mode_cols] = mode_imputer.fit_transform(df[mode_cols])

# Convert categorical columns to category type.
for col in mode_cols:
    df[col] = df[col].astype('category')

# Rename categories for display:
df['Gender_Label'] = df['Sex'].map({0: 'Male', 1: 'Female'})
df['Tumor_Location'] = df['Tumor_Location'].cat.rename_categories({0: "Lung", 1: "Breast", 2: "Colon"})

# --------------------------
# Prepare Data for Machine Learning (for Prediction)
# --------------------------
X = df.drop(columns=['Cancer_Status', 'Patient_ID'], errors='ignore')
y = df['Cancer_Status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# One-hot encode categoricals for ML
X_train_ml = pd.get_dummies(X_train, drop_first=True)
X_test_ml  = pd.get_dummies(X_test, drop_first=True)
X_test_ml = X_test_ml.reindex(columns=X_train_ml.columns, fill_value=0)

# --------------------------
# Main Prediction Model: Random Forest
# --------------------------
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train_ml, y_train)
cm_rf = confusion_matrix(y_test, random_forest_model.predict(X_test_ml))

# --------------------------
# Prepare Data for Clustering
# --------------------------
# Use only numeric columns (check_columns) for clustering
X_cluster = df[check_columns].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# --------------------------
# Hard-Coded Final Results for Display
# --------------------------
# Replace these example numbers with your final ones.
model_results = {
    "Logistic Regression": {
        "conf_matrix": np.array([[91, 28],
                                 [ 59,22]]),
        "cv_accuracy": 0.51875,
        "test_accuracy": 0.56,
        "auc": 0.4987031849776948
    },
    "Decision Tree": {
        "conf_matrix": np.array([[74, 45],
                                 [51, 30]]),
        "cv_accuracy": 0.49125,
        "test_accuracy": 0.52,
        "auc": 0.0  # update if available
    },
    "Random Forest": {
        "conf_matrix": np.array([[83, 36],
                                 [51,30]]),
        "cv_accuracy": 0.5149999999999999,
        "test_accuracy": 0.56,
        "auc": 0.5410312273057372
    },
    "KNN": {
        "conf_matrix": np.array([[69, 50],
                                 [46,35]]),
        "cv_accuracy": 0.4749999999999999,
        "test_accuracy": 0.52,
        "auc": 0.526351281253242
    },
    "AdaBoost": {
        "conf_matrix": np.array([[85, 34],
                                 [61,20]]),
        "cv_accuracy": 0.48375,
        "test_accuracy": 0.53,
        "auc": 0.46410416018259165
    },
    "XGBoost": {
        "conf_matrix": np.array([[74, 45],
                                 [48,33]]),
        "cv_accuracy": 0.49875,
        "test_accuracy": 0.54,
        "auc": 0.514679946052495
    }
}
# Use the Random Forest metrics for the main prediction page.
rf_cv_accuracy   = model_results["Random Forest"]["cv_accuracy"]
rf_test_accuracy = model_results["Random Forest"]["test_accuracy"]
rf_auc           = model_results["Random Forest"]["auc"]

# --------------------------
# Helper: Prepare Input Features for Prediction
# --------------------------
def prepare_features(df_input):
    df_out = pd.get_dummies(df_input, drop_first=True)
    df_out = df_out.reindex(columns=X_train_ml.columns, fill_value=0)
    return df_out

# --------------------------
# Dash App Setup
# --------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Cancer Prediction Dashboard"
app = dash.Dash(__name__, suppress_callback_exceptions=True) 
app.title = "Cancer Prediction Dashboard"
df['Sex'] = df['Sex'].cat.rename_categories({0: "Male", 1: "Female"}).astype(str)

biomarker_cols = ['CEA_Level', 'AFP_Level', 'WBC_Count', 'CRP_Level']
df_biomarkers = df[['Cancer_Status'] + biomarker_cols].copy()
df_long = df_biomarkers.melt(id_vars='Cancer_Status', var_name='Biomarker', value_name='Value')
df_long['Cancer_Status'] = df_long['Cancer_Status'].map({0: 'No Cancer', 1: 'Cancer'})

# Setting up the dashboard layout and title
app.layout = html.Div(style={'fontFamily': 'Helvetica Neue, sans-serif'}, children=[
    dcc.Tabs([
        dcc.Tab(label='📊 Statistics Overview', children=[
            html.Div(style={'backgroundColor': '#fff4e6', 'padding': '2rem'}, children=[
                html.H1("Canadian Cancer Statistics Dashboard", style={
    'textAlign': 'center',
    'color': '#fa7268',
    'fontSize': '40px',  
    'fontWeight': 'bold'
}),

# Setting up a dashboard message
                html.P("This interactive dashboard summarizes synthetic data from 1000 cancer patient records. This current tab provides an overview of this data set with exploratory and key descriptive plots. The following tab provides insight into the clustering patterns present. Finally, the last tab allows users to predict cancer risk by inputting values for each of the trained features.", style={'textAlign': 'center'}),
               html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '2rem'}, 

# Formatting the variable boxes                        
children=[
    html.Div(style={
        'backgroundColor': 'white',
        'padding': '1rem',
        'borderRadius': '12px',
        'boxShadow': '0px 2px 6px rgba(0, 0, 0, 0.1)',
        'width': '22%'
    }, children=[
        html.H3("🧑‍⚕️ Demographic Variables", style={'fontWeight': 'bold', 'fontSize': '18px'}),
        html.Ul([
            html.Li("Patient ID"),
            html.Li("Age"),
            html.Li("Sex"),
            html.Li("Smoking Status"),
            html.Li("Family Cancer History")
        ])
    ]),
    html.Div(style={
        'backgroundColor': 'white',
        'padding': '1rem',
        'borderRadius': '12px',
        'boxShadow': '0px 2px 6px rgba(0, 0, 0, 0.1)',
        'width': '22%'
    }, children=[
        html.H3("🧬 Genetic Mutations", style={'fontWeight': 'bold', 'fontSize': '18px'}),
        html.Ul([
            html.Li("TP53 Mutation"),
            html.Li("BRCA1 Mutation"),
            html.Li("KRAS Mutation")
        ])
    ]),
    html.Div(style={
        'backgroundColor': 'white',
        'padding': '1rem',
        'borderRadius': '12px',
        'boxShadow': '0px 2px 6px rgba(0, 0, 0, 0.1)',
        'width': '22%'
    }, children=[
        html.H3("🧪 Biomarkers", style={'fontWeight': 'bold', 'fontSize': '18px'}),
        html.Ul([
            html.Li("CEA Level"),
            html.Li("AFP Level"),
            html.Li("WBC Count"),
            html.Li("CRP Level")
        ])
    ]),
    html.Div(style={
        'backgroundColor': 'white',
        'padding': '1rem',
        'borderRadius': '12px',
        'boxShadow': '0px 2px 6px rgba(0, 0, 0, 0.1)',
        'width': '22%'
    }, children=[
        html.H3("🖼️ Imaging Features", style={'fontWeight': 'bold', 'fontSize': '18px'}),
        html.Ul([
            html.Li("Tumor Size"),
            html.Li("Tumor Location"),
            html.Li("Tumor Density")
        ])
    ])
]),

# Plotting: Tumor Location Pie Chart & Biomarker Violin Plot
html.Div(style={'display': 'flex', 'gap': '2rem', 'marginBottom': '2rem'}, children=[
    
    # Left: Pie Chart
    html.Div(style={
        'backgroundColor': 'white',
        'padding': '1.5rem',
        'borderRadius': '12px',
        'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)',
        'width': '50%'
    }, children=[
        dcc.Graph(figure=px.pie(df, names='Tumor_Location', title='Tumor Location Distribution',
                                color_discrete_sequence=px.colors.sequential.Peach)),
        html.P("This pie chart shows the distribution of tumor locations across the dataset.",
               style={'marginTop': '1rem', 'fontSize': '14px', 'color': '#555'})
    ]),

    # Right: Violin Plot
    html.Div(style={
        'backgroundColor': 'white',
        'padding': '1.5rem',
        'borderRadius': '12px',
        'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)',
        'width': '50%'
    }, children=[
        dcc.Graph(figure=px.violin(
            df_long, x='Biomarker', y='Value', color='Cancer_Status', box=True, points='all',
            title='Biomarker Distribution by Cancer Status',
            labels={'Value': 'Measured Level', 'Cancer_Status': 'Cancer Diagnosis'},
            color_discrete_map={'No Cancer': '#ffcc99', 'Cancer': '#fa7268'}
        )),
        html.P("This violin plot compares the distribution of biomarker levels between cancer and non-cancer patients.",
               style={'marginTop': '1rem', 'fontSize': '14px', 'color': '#555'})
    ])
]),
                
# Plotting: Cancer Rate by Gender Bar Plot
html.Div(style={'display': 'flex', 'gap': '2rem', 'marginBottom': '2rem'}, children=[

    html.Div(style={
        'backgroundColor': 'white',
        'padding': '1.5rem',
        'borderRadius': '12px',
        'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)',
        'width': '50%'
    }, children=[
        dcc.Graph(figure=px.bar(df.groupby('Sex')['Cancer_Status'].mean().reset_index(),
                                x='Sex', y='Cancer_Status',
                                title='Cancer Rate by Gender',
                                labels={'Sex': 'Gender', 'Cancer_Status': 'Cancer Rate'},
                                category_orders={'Sex': [0, 1]},
                                color_discrete_sequence=['#ffa07a'])
                 .update_xaxes(tickvals=[0, 1], ticktext=['Male', 'Female'])),
        html.P("This bar chart compares cancer rates between male and female patients.",
               style={'marginTop': '1rem', 'fontSize': '14px', 'color': '#555'})
    ]),

# Plotting: Cancer Rate by Smoking Status Bar Chart
    html.Div(style={
        'backgroundColor': 'white',
        'padding': '1.5rem',
        'borderRadius': '12px',
        'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)',
        'width': '50%'
    }, children=[
        dcc.Graph(figure=px.bar(df.groupby('Smoking_Status')['Cancer_Status'].mean().reset_index(),
                                x='Smoking_Status', y='Cancer_Status',
                                title='Cancer Rate by Smoking Status',
                                labels={'Smoking_Status': 'Smoking Status', 'Cancer_Status': 'Cancer Rate'},
                                category_orders={'Smoking_Status': [0, 1, 2]},
                                color_discrete_sequence=['#ffcba4'])
                 .update_xaxes(tickvals=[0, 1, 2], ticktext=['Non-smoker', 'Former smoker', 'Current smoker'])),
        html.P("This chart shows the relationship between smoking habits and cancer diagnosis.",
               style={'marginTop': '1rem', 'fontSize': '14px', 'color': '#555'})
    ])
]),

# Plotting: Tumor Size Distribution Boxplot
html.Div(style={
    'backgroundColor': 'white',
    'padding': '1.5rem',
    'marginBottom': '2rem',
    'borderRadius': '12px',
    'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)'
}, children=[
    dcc.Graph(figure=px.box(df, x='Cancer_Status', y='Tumor_Size',
                            title='Tumor Size by Cancer Status',
                            labels={'Cancer_Status': 'Cancer Diagnosis', 'Tumor_Size': 'Tumor Size (cm)'},
                            color_discrete_sequence=['#ffb347'])
                 .update_xaxes(tickvals=[0, 1], ticktext=['No Cancer', 'Cancer'])),
    html.P("This boxplot compares tumor sizes in patients with and without cancer diagnoses.",
           style={'marginTop': '1rem', 'fontSize': '14px', 'color': '#555'})
]),

# Plotting: Total Mutations vs. Age Scatter Plot           
html.Div(style={
    'backgroundColor': 'white',
    'padding': '1.5rem',
    'marginBottom': '2rem',
    'borderRadius': '12px',
    'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)'
}, children=[
    dcc.Graph(figure=px.scatter(
        df, x='Age', y='Total_Mutations', color='Gender_Label',
        title='Total Mutations vs. Age by Gender',
        labels={'Age': 'Age (years)', 'Total_Mutations': 'Total Genetic Mutations', 'Gender_Label': 'Gender'},
        color_discrete_map={'Male': '#e65c00', 'Female': '#ffcc99'}
    ).update_traces(marker=dict(size=8))
     .update_layout(legend_title_text='Gender', legend=dict(itemsizing='constant'))),
    html.P("This scatter plot shows the correlation between age and total mutations, split by gender.",
           style={'marginTop': '1rem', 'fontSize': '14px', 'color': '#555'})
])
            ])
        ]),


        
        # ----- Clustering (PCA + KMeans) Tab -----
        dcc.Tab(label='🔬 Clustering (PCA + KMeans)', children=[
            html.Div(style={'padding': '2rem'}, children=[
                html.H3("Interactive PCA Clustering", style={'textAlign': 'center'}),
                html.P(
                    "This 3D plot shows clusters of cancer patients based on patterns in age, biomarkers, and genetic features. "
                    "Principal Component Analysis (PCA) reduces high-dimensional data into a few components that explain the most variation. "
                    "Use the slider to explore different groupings using KMeans clustering.",
                    style={'textAlign': 'center'}
                ),
                dcc.Slider(
                    id='k-slider', min=2, max=6, step=1, value=2,
                    marks={i: str(i) for i in range(2, 7)}, tooltip={'placement': 'bottom'}
                ),
                dcc.Graph(id='pca-cluster-graph', style={'height': '700px'}),
                html.Div(id='cluster-accuracy', style={'textAlign': 'center', 'marginTop': '1rem', 'fontSize': '18px'}),
                dcc.Graph(id='confusion-matrix', style={'height': '500px'})
            ])
        ]),
       
        
        
        # ----- Predict Cancer Risk Tab -----
        dcc.Tab(label='🧪 Predict Cancer Risk', children=[
            html.Div(style={'padding': '2rem'}, children=[
                html.H3("Enter Patient Data", style={'marginBottom': '1rem'}),
                html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '1rem'}, children=[
                    html.Div([html.Label("Age"), dcc.Input(id='input-age', type='number', value=50)]),
                    html.Div([
                        html.Label("Sex (M/F)"),
                        dcc.Dropdown(
                            id='input-sex',
                            options=[{'label': 'M', 'value': 0}, {'label': 'F', 'value': 1}],
                            value=1
                        )
                    ]),
                    html.Div([
                        html.Label("Tumor Location"),
                        dcc.Dropdown(
                            id='input-location',
                            options=[
                                {'label': 'Lung', 'value': 0},
                                {'label': 'Breast', 'value': 1},
                                {'label': 'Colon', 'value': 2}
                            ],
                            value=0
                        )
                    ]),
                    html.Div([
                        html.Label("Smoking Status"),
                        dcc.Dropdown(
                            id='input-smoking',
                            options=[
                                {'label': 'Non-smoker', 'value': 0},
                                {'label': 'Former smoker', 'value': 1},
                                {'label': 'Current smoker', 'value': 2}
                            ],
                            value=0
                        )
                    ]),
                    html.Div([
                        html.Label("Family History"),
                        dcc.Dropdown(
                            id='input-family',
                            options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                            value=0
                        )
                    ]),
                    html.Div([
                        html.Label("TP53 Mutation"),
                        dcc.Dropdown(
                            id='input-tp53',
                            options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                            value=0
                        )
                    ]),
                    html.Div([
                        html.Label("BRCA1 Mutation"),
                        dcc.Dropdown(
                            id='input-brca1',
                            options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                            value=0
                        )
                    ]),
                    html.Div([
                        html.Label("KRAS Mutation"),
                        dcc.Dropdown(
                            id='input-kras',
                            options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                            value=0
                        )
                    ]),
                    html.Div([html.Label("Total Mutations"), dcc.Input(id='input-mutations', type='number', value=10)]),
                    html.Div([html.Label("CEA Level"), dcc.Input(id='input-cea', type='number', value=2.5)]),
                    html.Div([html.Label("AFP Level"), dcc.Input(id='input-afp', type='number', value=1.0)]),
                    html.Div([html.Label("WBC Count"), dcc.Input(id='input-wbc', type='number', value=6.0)]),
                    html.Div([html.Label("CRP Level"), dcc.Input(id='input-crp', type='number', value=3.0)]),
                    html.Div([html.Label("Tumor Size (cm)"), dcc.Input(id='input-size', type='number', value=3.5)]),
                    html.Div([html.Label("Tumor Density (g/cm³)"), dcc.Input(id='input-density', type='number', value=1.0)])
                ]),
                html.Br(),
                html.Div(style={'textAlign': 'center'}, children=[
                    html.Button("Predict", id='predict-button', style={'fontSize': '18px', 'padding': '0.5rem 1rem'}),
                    html.Div(
                        id='prediction-output', 
                        style={'marginTop': '1rem', 'fontSize': '22px', 'color': '#d6336c', 'fontWeight': 'bold'}
                    )
                ]),
                html.Br(),
                html.Div([
                    html.H4("Model Info"),
                    html.P(
                        "This Random Forest model has a mean cross-validation accuracy of "
                        f"{rf_cv_accuracy*100:.1f}% and a test accuracy of {rf_test_accuracy*100:.1f}%. "
                        f"It achieves an AUC of {rf_auc*100:.1f}%. Predictions are based on features including age, tumor characteristics, genetic mutations, and biomarker levels."
                    ),
                    html.Br(),
                    html.H4("Confusion Matrix (Random Forest)"),
                    dash_table.DataTable(
                        data=pd.DataFrame(
                            model_results["Random Forest"]["conf_matrix"],
                            columns=['Pred: No', 'Pred: Yes'],
                            index=['Actual: No', 'Actual: Yes']
                        ).reset_index().to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in ['index', 'Pred: No', 'Pred: Yes']],
                        style_cell={'textAlign': 'center', 'padding': '10px'},
                        style_header={'backgroundColor': '#ffe5b4', 'fontWeight': 'bold'},
                        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fffaf0'}]
                    ),
                    html.Hr(),
                    html.H4("Other Models' Performance"),
                    html.Div([
                        html.Div([
                            html.H5(model_name),
                            html.P(f"Cross-Validation Accuracy: {result['cv_accuracy']*100:.1f}%"),
                            html.P(f"Test Accuracy: {result['test_accuracy']*100:.1f}%"),
                            html.P(f"AUC: {result.get('auc', 0)*100:.1f}%"),
                            dash_table.DataTable(
                                data=pd.DataFrame(
                                    result["conf_matrix"],
                                    columns=['Pred: No', 'Pred: Yes'],
                                    index=['Actual: No', 'Actual: Yes']
                                ).reset_index().to_dict('records'),
                                columns=[{'name': i, 'id': i} for i in ['index', 'Pred: No', 'Pred: Yes']],
                                style_cell={'textAlign': 'center', 'padding': '10px'},
                                style_header={'backgroundColor': '#ffe5b4', 'fontWeight': 'bold'},
                                style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fffaf0'}]
                            )
                        ], style={'marginBottom': '2rem', 'border': '1px solid #ffe5b4', 'padding': '0.8rem'})
                        for model_name, result in model_results.items()
                        if model_name != "Random Forest"  # Exclude the main model from the grid.
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(300px, 1fr))', 'gap': '20px'})
                ])
            ])
        ])
    ])
])

# --------------------------
# Dash Callbacks
# --------------------------
@app.callback(
    Output('pca-cluster-graph', 'figure'),
    Output('cluster-accuracy', 'children'),
    Output('confusion-matrix', 'figure'),
    Input('k-slider', 'value')
)
def update_pca_plot(k):
    # Recompute KMeans on PCA-reduced data
    km = KMeans(n_clusters=k, random_state=42)
    clusters = km.fit_predict(X_pca)

    # Align length of labels (precaution)
    y_true = df['Cancer_Status'].values
    y_pred = clusters[:len(y_true)]

    # Compute Adjusted Rand Index
    ari = adjusted_rand_score(y_true, y_pred)

    # PCA scatter plot
    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_plot['Cluster'] = clusters.astype(str)
    pca_fig = px.scatter_3d(
        df_plot, x='PC1', y='PC2', z='PC3', color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"3D PCA Clustering (k={k})"
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Remove all-zero rows (where sum == 0)
    nonzero_rows = np.where(cm.sum(axis=1) > 0)[0]
    cm = cm[nonzero_rows, :]

    # Build axis labels (only for non-zero rows)
    y_labels = [f"Actual {i}" for i in nonzero_rows]
    x_labels = [f"Cluster {i}" for i in range(cm.shape[1])]

    # Create DataFrame and plot
    cm_df = pd.DataFrame(cm, index=y_labels, columns=x_labels)

    cm_fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale='Peach',
    labels=dict(x="Predicted Cluster", y="True Cancer Status", color="Count"),
    title="Confusion Matrix: True Labels vs KMeans Clusters"
    )

    return pca_fig, f"Clustering Accuracy (Adjusted Rand Index): {ari:.2f}", cm_fig

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        Input('input-age', 'value'),
        Input('input-sex', 'value'),
        Input('input-smoking', 'value'),
        Input('input-family', 'value'),
        Input('input-tp53', 'value'),
        Input('input-brca1', 'value'),
        Input('input-kras', 'value'),
        Input('input-mutations', 'value'),
        Input('input-cea', 'value'),
        Input('input-afp', 'value'),
        Input('input-wbc', 'value'),
        Input('input-crp', 'value'),
        Input('input-size', 'value'),
        Input('input-location', 'value'),
        Input('input-density', 'value')
    ]
)
def predict_cancer(n_clicks, age, sex, smoking, family, tp53, brca1, kras,
                   mutations, cea, afp, wbc, crp, size, location, density):
    if n_clicks is None:
        return ''
    # Map numeric Sex back to "Male"/"Female"
    sex_str = "Male" if sex == 0 else "Female"
    # Create an input DataFrame for prediction with same feature names as the training data
    df_input = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_str],
        "Smoking_Status": [smoking],
        "Family_History": [family],
        "TP53_Mutation": [tp53],
        "BRCA1_Mutation": [brca1],
        "KRAS_Mutation": [kras],
        "Total_Mutations": [mutations],
        "CEA_Level": [cea],
        "AFP_Level": [afp],
        "WBC_Count": [wbc],
        "CRP_Level": [crp],
        "Tumor_Size": [size],
        "Tumor_Location": [location],
        "Tumor_Density": [density]
    })
    df_input_enc = prepare_features(df_input)
    # Use the Random Forest model to predict
    prob = random_forest_model.predict_proba(df_input_enc)[0][1]
    emoji = "😊" if prob < 0.5 else "😟" if prob < 0.7 else "😲"
    return f"{emoji} Predicted cancer probability: {prob * 100:.2f}%"

if __name__ == '__main__':
    app.run(debug=True)
