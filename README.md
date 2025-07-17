# **Importing Libraries**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
```

# **Loading Data**


```python
df = pd.read_csv("/content/amazon.csv")
```


```python
df.head()
```





  <div id="df-07a4c544-c19b-4d1d-96c8-4450823c08a3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>category</th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>about_product</th>
      <th>user_id</th>
      <th>user_name</th>
      <th>review_id</th>
      <th>review_title</th>
      <th>review_content</th>
      <th>img_link</th>
      <th>product_link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B07JW9H4J1</td>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹399</td>
      <td>₹1,099</td>
      <td>64%</td>
      <td>4.2</td>
      <td>24,269</td>
      <td>High Compatibility : Compatible With iPhone 12...</td>
      <td>AG3D6O4STAQKAY2UVGEUV46KN35Q,AHMY5CWJMMK5BJRBB...</td>
      <td>Manav,Adarsh gupta,Sundeep,S.Sayeed Ahmed,jasp...</td>
      <td>R3HXWT0LRP0NMF,R2AJM3LFTLZHFO,R6AQJGUP6P86,R1K...</td>
      <td>Satisfied,Charging is really fast,Value for mo...</td>
      <td>Looks durable Charging is fine tooNo complains...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Wayona-Braided-WN3LG1-Sy...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B098NS6PVG</td>
      <td>Ambrane Unbreakable 60W / 3A Fast Charging 1.5...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹199</td>
      <td>₹349</td>
      <td>43%</td>
      <td>4.0</td>
      <td>43,994</td>
      <td>Compatible with all Type C enabled devices, be...</td>
      <td>AECPFYFQVRUWC3KGNLJIOREFP5LQ,AGYYVPDD7YG7FYNBX...</td>
      <td>ArdKn,Nirbhay kumar,Sagar Viswanathan,Asp,Plac...</td>
      <td>RGIQEG07R9HS2,R1SMWZQ86XIN8U,R2J3Y1WL29GWDE,RY...</td>
      <td>A Good Braided Cable for Your Type C Device,Go...</td>
      <td>I ordered this cable to connect my phone to An...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Ambrane-Unbreakable-Char...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B096MSW6CT</td>
      <td>Sounce Fast Phone Charging Cable &amp; Data Sync U...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹199</td>
      <td>₹1,899</td>
      <td>90%</td>
      <td>3.9</td>
      <td>7,928</td>
      <td>【 Fast Charger&amp; Data Sync】-With built-in safet...</td>
      <td>AGU3BBQ2V2DDAMOAKGFAWDDQ6QHA,AESFLDV2PT363T2AQ...</td>
      <td>Kunal,Himanshu,viswanath,sai niharka,saqib mal...</td>
      <td>R3J3EQQ9TZI5ZJ,R3E7WBGK7ID0KV,RWU79XKQ6I1QF,R2...</td>
      <td>Good speed for earlier versions,Good Product,W...</td>
      <td>Not quite durable and sturdy,https://m.media-a...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Sounce-iPhone-Charging-C...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B08HDJ86NZ</td>
      <td>boAt Deuce USB 300 2 in 1 Type-C &amp; Micro USB S...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹329</td>
      <td>₹699</td>
      <td>53%</td>
      <td>4.2</td>
      <td>94,363</td>
      <td>The boAt Deuce USB 300 2 in 1 cable is compati...</td>
      <td>AEWAZDZZJLQUYVOVGBEUKSLXHQ5A,AG5HTSFRRE6NL3M5S...</td>
      <td>Omkar dhale,JD,HEMALATHA,Ajwadh a.,amar singh ...</td>
      <td>R3EEUZKKK9J36I,R3HJVYCLYOY554,REDECAZ7AMPQC,R1...</td>
      <td>Good product,Good one,Nice,Really nice product...</td>
      <td>Good product,long wire,Charges good,Nice,I bou...</td>
      <td>https://m.media-amazon.com/images/I/41V5FtEWPk...</td>
      <td>https://www.amazon.in/Deuce-300-Resistant-Tang...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B08CF3B7N1</td>
      <td>Portronics Konnect L 1.2M Fast Charging 3A 8 P...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>₹154</td>
      <td>₹399</td>
      <td>61%</td>
      <td>4.2</td>
      <td>16,905</td>
      <td>[CHARGE &amp; SYNC FUNCTION]- This cable comes wit...</td>
      <td>AE3Q6KSUK5P75D5HFYHCRAOLODSA,AFUGIFH5ZAFXRDSZH...</td>
      <td>rahuls6099,Swasat Borah,Ajay Wadke,Pranali,RVK...</td>
      <td>R1BP4L2HH9TFUP,R16PVJEXKV6QZS,R2UPDB81N66T4P,R...</td>
      <td>As good as original,Decent,Good one for second...</td>
      <td>Bought this instead of original apple, does th...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Portronics-Konnect-POR-1...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-07a4c544-c19b-4d1d-96c8-4450823c08a3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-07a4c544-c19b-4d1d-96c8-4450823c08a3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-07a4c544-c19b-4d1d-96c8-4450823c08a3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4bb625b2-bd5d-45d0-8d91-622cf3f6c4d9">
  <button class="colab-df-quickchart" onclick="quickchart('df-4bb625b2-bd5d-45d0-8d91-622cf3f6c4d9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4bb625b2-bd5d-45d0-8d91-622cf3f6c4d9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1465 entries, 0 to 1464
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype 
    ---  ------               --------------  ----- 
     0   product_id           1465 non-null   object
     1   product_name         1465 non-null   object
     2   category             1465 non-null   object
     3   discounted_price     1465 non-null   object
     4   actual_price         1465 non-null   object
     5   discount_percentage  1465 non-null   object
     6   rating               1465 non-null   object
     7   rating_count         1463 non-null   object
     8   about_product        1465 non-null   object
     9   user_id              1465 non-null   object
     10  user_name            1465 non-null   object
     11  review_id            1465 non-null   object
     12  review_title         1465 non-null   object
     13  review_content       1465 non-null   object
     14  img_link             1465 non-null   object
     15  product_link         1465 non-null   object
    dtypes: object(16)
    memory usage: 183.2+ KB


# **Data Cleaning**


```python
df.isnull().sum()
```




    product_id             0
    product_name           0
    category               0
    discounted_price       0
    actual_price           0
    discount_percentage    0
    rating                 0
    rating_count           2
    about_product          0
    user_id                0
    user_name              0
    review_id              0
    review_title           0
    review_content         0
    img_link               0
    product_link           0
    dtype: int64




```python
df.dropna(subset=['rating_count'], inplace=True)

```


```python
df.drop(['product_link'], axis=1, inplace=True)
```


```python
df.drop(['img_link'], axis=1, inplace=True)
```


```python
df.drop_duplicates(inplace=True)
```


```python
df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%','').astype(float)/100
df = df[df['rating'].apply(lambda x: '|' not in str(x))]
df['rating'] = df['rating'].astype(str).str.replace(',', '').astype(float)
df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)
```

    <ipython-input-10-c67033094a80>:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['rating'] = df['rating'].astype(str).str.replace(',', '').astype(float)
    <ipython-input-10-c67033094a80>:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1397 entries, 0 to 1464
    Data columns (total 14 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   product_id           1397 non-null   object 
     1   product_name         1397 non-null   object 
     2   category             1397 non-null   object 
     3   discounted_price     1397 non-null   float64
     4   actual_price         1397 non-null   float64
     5   discount_percentage  1397 non-null   float64
     6   rating               1397 non-null   float64
     7   rating_count         1397 non-null   float64
     8   about_product        1397 non-null   object 
     9   user_id              1397 non-null   object 
     10  user_name            1397 non-null   object 
     11  review_id            1397 non-null   object 
     12  review_title         1397 non-null   object 
     13  review_content       1397 non-null   object 
    dtypes: float64(5), object(9)
    memory usage: 163.7+ KB



```python
df['weighted_rating'] = df['rating_count'] * df['rating']
```

The below two steps are done in order to segrgate the single category column into sub and main caegories so that it will be easy for us to analyse the data and represent them visually


```python
df['sub_category'] = df['category'].astype(str).str.split('|').str[-1]
```


```python
df['main_category'] = df['category'].astype(str).str.split('|').str[0]

```

# **Data Visualisation**


```python
mc_counts = df['main_category'].value_counts()[:25]

```


```python
plt.figure(figsize=(12, 6))
sns.barplot(x=mc_counts.values, y=mc_counts.index, palette="viridis")

plt.xlabel('Number of Products')
plt.ylabel('Main Category')
plt.title('Distribution of Products by Main Category (Top 25)')
plt.show()
```


    
![png](output_20_0.png)
    



```python
sc_counts = df['sub_category'].value_counts()[:25]
plt.figure(figsize=(10, 8))
sns.barplot(x=sc_counts.values, y=sc_counts.index, palette="RdBu_r")
plt.xlabel('Number of Products')
plt.ylabel('Sub Category')
plt.title('Distribution of Products by Sub Category (Top 25)')
plt.show()
```


    
![png](output_21_0.png)
    



```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

reviews_text = ' '.join(df['review_content'].dropna().values)

word_frequencies = {}
for word in reviews_text.split():
    word_frequencies[word] = word_frequencies.get(word, 0) + 1

wordcloud = WordCloud(width=600, height=860, background_color='orange', min_font_size=12).generate_from_frequencies(word_frequencies)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Reviews')
plt.show()
```


    
![png](output_22_0.png)
    



```python
plt.hist(df['rating'])
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.title('Customer Rating Distribution')
plt.show()
```


    
![png](output_23_0.png)
    



```python

top = df.groupby('main_category')['rating'].mean().sort_values(ascending=False).head(10).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='main_category', y='rating', data=top, palette='viridis')
plt.xlabel('Main Category')
plt.ylabel('Average Rating')
plt.title('Top Main Categories by Rating')
plt.xticks(rotation=45, ha='right')
plt.show()


```


    
![png](output_24_0.png)
    



```python
mean_discount_by_category = df.groupby('main_category')['discount_percentage'].mean()
mean_discount_by_category = mean_discount_by_category.sort_values(ascending=True)

plt.barh(mean_discount_by_category.index, mean_discount_by_category.values)
plt.title('Discount Percentage by Main Category')
plt.xlabel('Discount Percentage')
plt.ylabel('Main Category')
plt.show()
```


    
![png](output_25_0.png)
    



```python
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

    <ipython-input-22-dd73e8ae7eaa>:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      correlation_matrix = df.corr()



    
![png](output_26_1.png)
    





```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1397 entries, 0 to 1464
    Data columns (total 17 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   product_id           1397 non-null   object 
     1   product_name         1397 non-null   object 
     2   category             1397 non-null   object 
     3   discounted_price     1397 non-null   float64
     4   actual_price         1397 non-null   float64
     5   discount_percentage  1397 non-null   float64
     6   rating               1397 non-null   float64
     7   rating_count         1397 non-null   float64
     8   about_product        1397 non-null   object 
     9   user_id              1397 non-null   object 
     10  user_name            1397 non-null   object 
     11  review_id            1397 non-null   object 
     12  review_title         1397 non-null   object 
     13  review_content       1397 non-null   object 
     14  weighted_rating      1397 non-null   float64
     15  sub_category         1397 non-null   object 
     16  main_category        1397 non-null   object 
    dtypes: float64(6), object(11)
    memory usage: 196.5+ KB



```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


selected_columns = ['discounted_price', 'actual_price','discount_percentage', 'rating_count', 'rating']
X = df[selected_columns]
y = df['product_name']


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier()

tree_model.fit(X_train, y_train)

predictions = tree_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

```

    Accuracy: 0.07142857142857142

