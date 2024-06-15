# Prompts for each section
section_prompts = [
    "Introduction"
    "Import Modules"
    "Load the Data"
    "Exploratory Data Analysis",
    "Data Preprocessing and Feature Engineering",
    "Logistic Regression",
    "Decision Trees",
    "Random Forests",
    "K Nearest Neighbours",
    "Support Vector Machines",
    "Evaluation Metrics for Classification Models",
    "Handling Missing Data",
    "Categorical Data Encoding",
    "Feature Scaling",
    "PCA",
    "Pipelines and Model Persistence",
    "Imbalanced Data Handling",
    "Outlier Detection and Removal",
    "Feature Engineering and Creation"
]
# Sidebar for section selection
section = st.sidebar.selectbox("Choose a section", section_prompts)

introduction = '''
# **Breast Cancer Wiscnosin Dataset Description**
 
Features were computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

Attribute Information:

1. ID number
 2. Diagnosis (M = malignant, B = benign)
 3. The remaining columns 3-32:

Ten real-valued features are computed for each cell nucleus:

* radius (mean of distances from center to points on the perimeter)
* texture (standard deviation of gray-scale values)
* perimeter
* area
* smoothness (local variation in radius lengths)
* compactness (perimeter^2 / area - 1.0)
* concavity (severity of concave portions of the contour)
* concave points (number of concave portions of the contour)
 * symmetry
 * fractal dimension ("coastline approximation" - 1)

### **The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.**

* For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant'''

if section == "Introduction":
    st.title("Introduction")
    st.markdown(introduction)

if section == "Import Modules":
    st.title("Import Modules")
    st.markdown('# ** Import Modules**')
    st.write(''' Imported modules are \n
    import streamlit as st\n
    import pandas as pd\n
    import numpy as np\n
    import matplotlib.pyplot as plt\n
    import seaborn as sns\n
    from sklearn.model_selection import train_test_split\n
    from sklearn.linear_model import LogisticRegression\n
    from sklearn.tree import DecisionTreeClassifier\n
    from sklearn.ensemble import RandomForestClassifier\n
    from sklearn.neighbors import KNeighborsClassifier\n
    from sklearn.svm import SVC\n
    from sklearn.preprocessing import StandardScaler, OneHotEncoder\n
    from sklearn.impute import SimpleImputer\n
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n
    from sklearn.pipeline import Pipeline\n
    from sklearn.decomposition import PCA\n
    from imblearn.over_sampling import SMOTE\n
    from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline ''')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

if section == "Load the Data":
    st.title("Load the Data")
    st.write('data loaded with pd.read_csv("data.csv")')


df = pd.read_csv("data.csv")

explore_data = '''
# **Step 4: Exploratory Data Analysis**
* #### **Understanding the dataset**

* Checking the overall structure of the dataset and what each column.

Here we define a class called DataExplorer, that takes our data frame and does summary statistics on the features, including checking for null values and duplicates.

we use the display method form ipython to make the print output more readable'''

class DataExplorer:
    def __init__(self, data):
        self.data = data

    # the function that styles the text output
    def print_styled(self, text, size="normal"):
        if size == "normal":
            st.markdown(text, unsafe_allow_html=True)
        elif size == "large":
            st.markdown(f"# {text}", unsafe_allow_html=True)
        elif size == "medium":
            st.markdown(f"## {text}", unsafe_allow_html=True)
        elif size == "small":
            st.markdown(f"### {text}", unsafe_allow_html=True)
        else:
            st.markdown(text, unsafe_allow_html=True)
        
    # the rest are methods that presents outputs as tables
    def print_statistical_summary(self):
        self.print_styled("Statistical summary per feature of the Breast Cancer Wisconsin Dataset:", size="medium")
        st.table(self.data.describe().transpose())

    def print_dataset_information(self):
        self.print_styled("Dataset Information:", size="medium")
        self.print_styled("There are 569 entries and 32 columns. Each column has no null value.", size="small")
        print(self.data.info())

    def print_dataframe_shape(self):
        self.print_styled("Shape of the Dataframe:", size="medium")
        print(self.data.shape)

    def print_head_of_data(self):
       self.print_styled("Head of the Data:", size="medium")
       st.table(self.data.head())

    def print_tail_of_data(self):
        self.print_styled("Tail of the Data:", size="medium")
        st.table(self.data.tail())

    def print_null_values_count(self):
        self.print_styled("Number of Null Values:", size="medium")
        print(self.data.isnull().sum())

    def print_duplicated_values_count(self):
        self.print_styled("Number of Duplicated Values:", size="h2")
        print(self.data.duplicated().sum())

    def print_unique_values_count(self):
        self.print_styled("Number of Unique Values:", size="medium")
        st.table(self.data.nunique())

    def print_value_counts(self, category):
        self.print_styled("Data Distribution:", size="medium")
        print(self.data[category].value_counts())

    def df_encoded(self):
        self.print_styled("Encoded Data:", size="medium")
        self.print_styled("Malignant = 1, Benign = 0", size ="small")
        self.data["target"]= self.data["diagnosis"].map(lambda row: 1 if row=='M' else 0)
        return self.data

    def correlation(self):
        df_cleaned = self.data.drop("diagnosis", axis=1)
        self.print_styled("Correlation Matrix:", size="medium")
        st.table(df_cleaned.corr())
     
if section == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown(explore_data)
    # make an instance of DataExplorer()
    data_explorer = DataExplorer(df)
    data_explorer.print_statistical_summary()
    data_explorer.print_dataset_information()
    data_explorer.print_dataframe_shape()
    data_explorer.print_head_of_data()
    data_explorer.print_tail_of_data()
    data_explorer.print_null_values_count()
    data_explorer.print_duplicated_values_count()
    data_explorer.print_unique_values_count()
    data_explorer.print_value_counts("diagnosis")




description = '''
# The next three cells create column names that will be useful later, after our df has been changed
'''


# list all the columns
columns = list(df.columns)
# columns


# create a list of all the columns of possible combinations
mean_columns = [col for col in columns if 'mean' in col]
se_columns = [col for col in columns if 'se' in col]
worst_columns = [col for col in columns if 'worst' in col]

# add the diagnosis column
mean_columns.append("diagnosis")
se_columns.append("diagnosis")
worst_columns.append("diagnosis")
print(mean_columns)
print(se_columns)
print(worst_columns)

# create the data frames
df_mean = df[mean_columns]
df_se = df[se_columns]
df_worst = df[worst_columns]

data_processing = '''
## **Data Preprocessing**
Our dataframe has no null values, no duplicates. so we proceed to encode the diagnosis column with numerical values.

We replace M with 1 and B with 0. (Label Ecoding/ Binary Encoding)'''

#Replace M with 1 and Begnin with 0 (else 0)
encoded_data = DataExplorer(df)
df_encoded = encoded_data.df_encoded().drop("diagnosis", axis=1)
df_encoded.head()



# Remove the id column
df_encoded = df_encoded.drop("id", axis=1)
df_encoded.head()

encoded_data.correlation()

visualizations = '''
# **Step 5: Some Visualisations**

We define a class DataVIsualizer that takes our dataframe and performs some basic visualisations. '''

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_line(self, x, y, **kwargs):
        sns.lineplot(data=self.data, x=x, y=y, **kwargs)
        plt.title('Line Plot')
        st.pyplot(plt)

    # Distribution plots
    def plot_histogram(self, column,figsize=(12,8), **kwargs):
        """kwargs for histplot
        sns.set(style='darkgrid')
        sns.set(style='white')

          figsize=(12,8), bins=10, hue='a categorical column', color='red',edgecolor='black',lw=4,ls='--'
          """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.histplot(data=self.data, x=column, **kwargs)
        plt.title('Histogram')
        st.pyplot(plt)

    def plot_distplot(self, column,figsize=(12,8), **kwargs):
      """kwargs for distplot
      sns.set(style='darkgrid')
      sns.set(style='white')

        figsize=(12,8), kde=True, color='red',bins=10, rug=True, edgecolor='black',lw=4,ls='--'
        sns.kdeplot(data=sample_ages,x='age', clip=[0,100], bw_adjust=0.1, shade=True, color='red', linewidth=6)
        """
      plt.figure(figsize=figsize)
      kwargs.pop('figsize', None)  # Remove figsize from kwargs
      sns.histplot(data=self.data, x=column, **kwargs)
      plt.title('Distplot')
      st.pyplot(plt)


    # Categorical plots

    def plot_countplot(self, column,figsize=(12,8), **kwargs):
      """kwargs for countplot
        figsize=(12,8), hue='a categorical column', palette='Set1', palette='Paired'

        """
      plt.figure(figsize=figsize)
      kwargs.pop('figsize', None)  # Remove figsize from kwargs
      sns.countplot(data=self.data, x=column, **kwargs)
      plt.title('Countplot')
      st.pyplot(plt)


    def plot_barplot(self, x, y, figsize=(12,8), **kwargs):
      """kwargs for barplot
        figsize=(12,8), hue='a categorical column', palette='Set1', palette='Paired'

        """
      plt.figure(figsize=figsize)
      kwargs.pop('figsize', None)  # Remove figsize from kwargs
      sns.barplot(data=self.data, x=x, y=y, estimator=np.mean,ci='sd' **kwargs)
      plt.title('Barplot')
      st.pyplot(plt)


    def plot_scatter(self, x, y, figsize=(12,8), **kwargs):

        """kwargs for scatterplot
        figsize=(12,8), hue='diagnosis', size='another categorical column', s=200, palette='viridis', linewidth=0,alpha=0.2, edgecolor='black', marker='o',
        style='another categorical column', legend=False
        """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.scatterplot(data=self.data, x=x, y=y, **kwargs)
        plt.title('Scatter Plot')
        st.pyplot(plt)

    def plot_box(self, figsize=(12,8), **kwargs):

        """kwargs for boxplot
        figsize=(12,8), x= "a categorical var", y = "a continuous var", orient = "h", width=0.3, hue='a categorical column', palette='Set1', palette='Paired'

        """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.boxplot(data=self.data, **kwargs)
        plt.title('Box Plot')
        st.pyplot(plt)

    def plot_heatmap(self, figsize=(12,12), **kwargs):
        plt.figure(figsize=figsize)

        kwargs.pop('figsize', None)  # Remove figsize from kwargs

        sns.set_theme(style="white")
        corr = self.data.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr, mask= mask, cmap=cmap, square=True, fmt=".2f", **kwargs)
        plt.title('Heatmap')
        st.pyplot(plt)

    def plot_violin(self, x, y, figsize = (12,8), **kwargs):

        """kwargs for violinplot
        figsize=(12,8), x= "a categorical var", y = "a continuous var", hue='a categorical column', split=True, inner=None, inner='quartile',
        inner='box', inner='stick', palette='Set1', palette='Paired', bw=0.1,

        """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.violinplot(data=self.data, x=x, y=y, **kwargs)
        plt.title('Violin Plot')
        st.pyplot(plt)

    # Comparison plots
    def plot_pairplot(self, figsize= (12,8), **kwargs):
        """ kwargs for pairplort
        figsize=(12,8), hue='a categorical column', palette='Set1', palette='Paired', palette='viridis', corner=True, diag_kind='hist',
        """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.pairplot(self.data,  hue="diagnosis", corner=True,palette='viridis' )
        plt.title('Pairplot')
        st.pyplot(plt)


data_visuals = '# **Create an instance of DataVisualizer()**'

data_visualizer = DataVisualizer(df)

# %% [markdown]
# ## Distribution of cases as Malignant vs Benign

# %%
data_visualizer.plot_countplot("diagnosis", figsize = (6,4), edgecolor='black',)

# %% [markdown]
# ## Checking for Correlation

# %%
encoded_data_visualizer = DataVisualizer(df_encoded)
encoded_data_visualizer.plot_heatmap(figsize=(15,15), linewidths=.5, cbar_kws={"shrink": .6})


# %% [markdown]
# In a correlation heat map, the higher the correlation value, the more correlated the two variables are:
# 
# Radius, Area and Perimeter are correlated (corr>0.9) which is obvious as area and perimeter is calculated using the radius values.
# 
# Texture_mean and texture_worst are higly correlated with corr_value = 0.98 (texture_worst is the largest value of all the textures).
# 
# Compactness_mean,concavity_mean,concave_points_mean are also highy correlated with values in range 0.7 to 0.9.
# 
# Symmetry_mean and symmetry_worst are correlated too by values 0.7.
# 
# Fractural_dimension_mean and fractural_dimension_worst are correlated by value 0.77

# %% [markdown]
# ## Pairplots

# %% [markdown]
# Remember this statement?
# 
# ### **The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.**
# 
# In order to display some visualisations, we use dimensionality reduction techniques to better understand our data.
# 
# Lets visualize pairplots of the different dataframes that will result from groupimg together all the means, all the standard errors and all the worst/largest dimensions.

# %% [markdown]
# Now recall the three sub dataframes? df_mean, df_se, df_worst? We use them for out pair plots

# %%
df_mean

# %% [markdown]
# 

# %%
# instantiate the mean visualizer
data_viz = DataVisualizer(df_mean)
data_viz.plot_pairplot(figsize=(15,15),)


# %%
# instantiate the standard error visualizer
data_viz = DataVisualizer(df_se)
data_viz.plot_pairplot(figsize=(15,15))

# %%
# instantiate the worst dimension visualizer
data_viz = DataVisualizer(df_worst)
data_viz.plot_pairplot(figsize=(15,15))

# %% [markdown]
# ## **Dimensionality Reduction with Principal Component analysis**
# 
# We define a class DimensionalityReducer that will perform pca, plot the scree plot and a sctter plot of the chosen 2 principal components

# %%
class DimensionalityReducer:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    def __init__(self, data, n_components, target=None):
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
        self.n_components = n_components
        self.target = target
        self.pca = PCA(n_components=self.n_components)
        self.pca_result = self.pca.fit_transform(self.data)

    def apply_pca(self):

        pca_df = pd.DataFrame(data=self.pca_result, columns=[f'PC{i+1}' for i in range(self.n_components)])
        print(f'Explained variance by components: {self.pca.explained_variance_ratio_}')
        return pca_df

    def plot_pca(self, categorical_value = None):
        pca_df = self.apply_pca()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, c= categorical_value)
        plt.title('PCA Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def plot_scree_plot(self):
      # Create a scree plot
      plt.figure(figsize=(10, 6))
      plt.bar(range(1, len(self.pca.explained_variance_ratio_) + 1), self.pca.explained_variance_ratio_, alpha=0.6, color='b')
      plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), self.pca.explained_variance_ratio_, 'o-', color='b')
      plt.title('Scree Plot')
      plt.xlabel('Principal Component')
      plt.ylabel('Explained Variance Ratio')
      plt.xticks(range(1, len(self.pca.explained_variance_ratio_) + 1))
      plt.grid(True)
      plt.show()


# %%
# We only need to call the class, and the plot new plot is shown
d_reducer = DimensionalityReducer(df_encoded, n_components=2)
d_reducer.plot_pca(categorical_value= df_encoded["target"])

# %%
d_reducer.plot_scree_plot()

# %% [markdown]
# # **Step 6: Model Training and Deployment**

# %% [markdown]
# 

# %%


# %% [markdown]
# We define a function train_and_test model that will train all model and evaluate performance metrics

# %%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Function to train and test the model
def train_and_test_model(model, X, y):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model on the training data
    model.fit(x_train, y_train)
    print(f'Accuracy on training {model}:', model.score(x_train, y_train) * 100)

    # Print the accuracy on the test data
    print(f'Accuracy on testing {model} :', model.score(x_test, y_test) * 100)

    # Generate predictions on the test data
    y_pred = model.predict(x_test)

    # Print the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    # print(report.split())

    return model, report



# %% [markdown]
# Populate all models in alist, train them iteretively

# %%
# Create a list of all models
models = {
        'svc' : SVC(C=3),
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=10000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

# Train and test each model, and store the structured classification report
trained_models = {}
model_scores = {}
for model_name, model in models.items():
    model, report = train_and_test_model(model, df_encoded.drop("target", axis = 1), df_encoded["target"])
    trained_models[model_name] = model
    model_scores[model_name] = report
    # print(f"{model_name} \n {report}\n\n")



# %% [markdown]
# Train models with PCA-reduced data

# %%
print(model_scores)

# %%
# Negative values in data cannot be passed to MultinomialNB (input X), so we drop it
trained_models_after_pca = {}
x = d_reducer.apply_pca()
y = df_encoded["target"]
model_scores_after_pca = {}
for model_name, model in models.items():
  if model_name != 'Naive Bayes':
    model, report = train_and_test_model(model, x, y)
    trained_models_after_pca[model_name] = model
    model_scores_after_pca[model_name] = report
    # print(f"{model_name} \n {report}\n\n")

# %%
print(model_scores_after_pca)

# %% [markdown]
# Visualisations to evaluate perfomance of models before and after principal componenet analysis

# %%
# We start by tabulating model scores into a dataframe

metrics_of_interest = ['precision', 'recall', 'f1-score']

# Function to extract the metrics
def extract_metrics(report_dict):
    extracted_values = {metric: {} for metric in metrics_of_interest}
    for model, report in report_dict.items():
        for metric in metrics_of_interest:
            if 'weighted avg' in report:
                extracted_values[metric][model] = report['weighted avg'][metric]
    return extracted_values



# Extract metrics for both conditions
before_pca_metrics = extract_metrics(model_scores)
after_pca_metrics = extract_metrics(model_scores_after_pca)

# Create DataFrames for each condition
df_before_pca = pd.DataFrame(before_pca_metrics).transpose()
df_after_pca = pd.DataFrame(after_pca_metrics).transpose()

# Rename columns to indicate the condition
df_before_pca.columns = [f'{col}_Before_PCA' for col in df_before_pca.columns]
df_after_pca.columns = [f'{col}_After_PCA' for col in df_after_pca.columns]

# Combine the DataFrames
df_combined = pd.concat([df_before_pca, df_after_pca], axis=1)

# Display the combined DataFrame
df_combined

# %%
# Prepare the DataFrame for plotting
df_plot = df_combined.reset_index().melt(id_vars='index', var_name='Condition', value_name='Score')
print(df_plot.head())
print("===================================================================")
df_plot[['Model', 'Condition']] = df_plot['Condition'].str.split('_', n=1, expand=True)
print(df_plot.head())
print(df_plot.columns)
# Set custom palette
# palette = {'Before_PCA': 'lightblue', 'After_PCA': 'lightgreen'}

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='index', y='Score', hue='Condition', data=df_plot, palette="viridis")
plt.title('Model Performance Metrics Before and After PCA')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.ylim(0, 1)  # Assuming scores are between 0 and 1
plt.legend(title='Condition', loc='upper right', bbox_to_anchor=(1.1, 1))
plt.show()

# %%
# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Score', hue='Condition', data=df_plot, palette="viridis")
plt.title('Model Performance Metrics Before and After PCA')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.ylim(0, 1)  # Assuming scores are between 0 and 1
plt.legend(title='Condition', loc='upper right', bbox_to_anchor=(1.1, 1))
plt.show()

# %%
# Determine the number of columns (models)
n_models = len(df_plot['Model'].unique())
n_cols = 3  # Number of columns you want

# Plotting
g = sns.catplot(
    data=df_plot,
    x='index', y='Score', hue='Condition',
    col='Model', kind='bar',
    palette="viridis", height=6, aspect=1.2,
    col_wrap=n_cols
)
g.fig.suptitle('Model Performance Metrics Before and After PCA', y=1.03)
g.set_axis_labels("Metric", "Score")
g.set_titles("{col_name}")
g.set(ylim=(0, 1))
g.add_legend(title='Condition', loc="lower right")

# Adjust the layout
plt.show()

# %% [markdown]
# Noted overall better model perfomance with dimensionality reduction
# 

# %% [markdown]
# # **Model Deployment**




