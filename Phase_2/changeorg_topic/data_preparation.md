```python
"""
Use the csv files created by the code, and put the final label for each text on each
lockdown, mask, and vaccination file for each dataset.
"""
import pandas as pd
import sklearn
from sklearn import metrics
```

# Primary Sets

# Dataset 0


```python
df_0 = pd.read_csv('twitter_topic_0.csv')
```


```python
df_0.fillna('missing', inplace=True)
```


```python
df_0_lockdown = pd.read_csv('twitter_topic_0_lockdowns.csv')
```


```python
df_0_mask = pd.read_csv('twitter_topic_0_masking_and_distancing.csv')
```


```python
df_0_vaccination = pd.read_csv('twitter_topic_0_vaccination.csv')
```


```python
#df_0.head()
```

# Dataset 1


```python
df_1 = pd.read_csv('twitter_topic_1.csv')
```


```python
df_1.fillna('missing', inplace=True)
```


```python
df_1_lockdown = pd.read_csv('twitter_topic_1_lockdowns.csv')
```


```python
df_1_mask = pd.read_csv('twitter_topic_1_masking_and_distancing.csv')
```


```python
df_1_vaccination = pd.read_csv('twitter_topic_1_vaccination.csv')
```


```python
#df_1.head()
```

# Dataset 2


```python
df_2 = pd.read_csv('twitter_topic_2.csv')
```


```python
df_2.fillna('missing', inplace=True)
```


```python
df_2_lockdown = pd.read_csv('twitter_topic_2_lockdowns.csv')
```


```python
df_2_mask = pd.read_csv('twitter_topic_2_masking_and_distancing.csv')
```


```python
df_2_vaccination = pd.read_csv('twitter_topic_2_vaccination.csv')
```


```python
#df_2.head()
```

# Dataset 3


```python
df_3 = pd.read_csv('twitter_topic_3.csv')
```


```python
df_3.fillna('missing', inplace=True)
```


```python
df_3_lockdown = pd.read_csv('twitter_topic_3_lockdowns.csv')
```


```python
df_3_mask = pd.read_csv('twitter_topic_3_masking_and_distancing.csv')
```


```python
df_3_vaccination = pd.read_csv('twitter_topic_3_vaccination.csv')
```


```python
#df_3.head()
```

# Dataset 4


```python
df_4 = pd.read_csv('twitter_topic_4.csv')
```


```python
df_4.fillna('missing', inplace=True)
```


```python
df_4_lockdown = pd.read_csv('twitter_topic_4_lockdowns.csv')
```


```python
df_4_mask = pd.read_csv('twitter_topic_4_masking_and_distancing.csv')
```


```python
df_4_vaccination = pd.read_csv('twitter_topic_4_vaccination.csv')
```


```python
#df_4.head()
```

## Finding Kappa


```python
df_cur = df_2

kappa_1 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/3

kappa_2 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/3

kappa_3 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/3

kappa_4 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_0.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/3
```


```python
print('kappa 1:', kappa_1)
print('kappa 2:', kappa_2)
print('kappa 3:', kappa_3)
print('kappa 4:', kappa_4)
```

    kappa 1: 0.4125172568390408
    kappa 2: 0.4479413860850894
    kappa 3: 0.454406657189879
    kappa 4: 0.2803217338357632


# Annotators, kappa's

## df_0: 
### 4 Annotators, keep all.

kappa 56: 0.37449258477361463

kappa 57: 0.4628485822445733

kappa 58: 0.36724433436200576

kappa 59: 0.48572536773342634


## df_1
### 4 Annotators, keep all.

kappa 91: 0.6210232245439785

kappa 85: 0.5957329504706256

kappa 86: 0.5975649954411598

kappa 87: 0.38390911860685667 


## df_2
### 4 Annotators, keep all.

kappa 51: 0.4724964451243631

kappa 52: 0.34995673694855817

kappa 53: 0.4956326545021735

kappa 54: 0.37251496342922535

## df_3
### 4 Annotators, keep all.

kappa 104: 0.44315735995110944

kappa 101: 0.65511568783928

kappa 102: 0.569458591203231

kappa 103: 0.3508589039050169

## df_4
### 4 Annotators, keep all.

kappa 72: 0.44315735995110944

kappa 74: 0.65511568783928

kappa 76: 0.569458591203231

kappa 71: 0.3508589039050169


# Labeling The Datasets
### Do it for each dataset


```python
df_cur = df_4_lockdown
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 5] = True
    elif label_true < label_false:
        df_cur.iloc[i, 5] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 5] = True if kappa_scores>=0 else False
        
df_4_lockdown = df_cur
```


```python
df_cur = df_4_mask
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 5] = True
    elif label_true < label_false:
        df_cur.iloc[i, 5] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 5] = True if kappa_scores>=0 else False
        
df_4_mask = df_cur
```


```python
df_cur = df_4_vaccination
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 5] = True
    elif label_true < label_false:
        df_cur.iloc[i, 5] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 5] = True if kappa_scores>=0 else False
        
df_4_vaccination = df_cur
```

# Combine Datasets
### Combine each label type to itself

### Lockdown


```python
df_lockdown = df_0_lockdown[['text', 'label']]
df_lockdown = df_lockdown.append(df_1_lockdown[['text','label']], ignore_index = True)
df_lockdown = df_lockdown.append(df_2_lockdown[['text','label']], ignore_index = True)
df_lockdown = df_lockdown.append(df_3_lockdown[['text','label']], ignore_index = True)
df_lockdown = df_lockdown.append(df_4_lockdown[['text','label']], ignore_index = True)
```


```python
df_lockdown.head()
```




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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Federal Judge Rules Against CDC, Throws Out Cr...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Indeed. Even in the dysfunctional US health sy...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a vaccine seems to be ESSENTIAL if we are to s...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nurses union calls on CDC to reinstate univers...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sirf  #MukeshAmbani  Company &amp;amp; it's Worker...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>Dr Reddy's Laboratories and Russian Direct Inv...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>"#Covishield will be commercialised once the t...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>I am not, last lockdown here on RI was a compl...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>https://t.co/vuf52eSRgq  Why this warning or ...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>Opioid Prescribing Practices After the 2016 Re...</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1500 rows × 2 columns</p>
</div>




```python
df_lockdown.to_csv("twitter_topic_lockdown.csv")
```

### Masking and Distancing


```python
df_mask = df_0_mask[['text', 'label']]
df_mask = df_mask.append(df_1_mask[['text','label']], ignore_index = True)
df_mask = df_mask.append(df_2_mask[['text','label']], ignore_index = True)
df_mask = df_mask.append(df_3_mask[['text','label']], ignore_index = True)
df_mask = df_mask.append(df_4_mask[['text','label']], ignore_index = True)

```


```python
df_mask.head()
```




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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Federal Judge Rules Against CDC, Throws Out Cr...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Indeed. Even in the dysfunctional US health sy...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a vaccine seems to be ESSENTIAL if we are to s...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nurses union calls on CDC to reinstate univers...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sirf  #MukeshAmbani  Company &amp;amp; it's Worker...</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_mask.to_csv("twitter_topic_masking_and_distancing.csv")
```

### Vaccination


```python
df_vaccination = df_0_vaccination[['text', 'label']]
df_vaccination = df_vaccination.append(df_1_vaccination[['text','label']], ignore_index = True)
df_vaccination = df_vaccination.append(df_2_vaccination[['text','label']], ignore_index = True)
df_vaccination = df_vaccination.append(df_3_vaccination[['text','label']], ignore_index = True)
df_vaccination = df_vaccination.append(df_4_vaccination[['text','label']], ignore_index = True)
```


```python
df_vaccination.head()
```




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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Federal Judge Rules Against CDC, Throws Out Cr...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Indeed. Even in the dysfunctional US health sy...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a vaccine seems to be ESSENTIAL if we are to s...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nurses union calls on CDC to reinstate univers...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sirf  #MukeshAmbani  Company &amp;amp; it's Worker...</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_vaccination.to_csv("twitter_topic_vaccination.csv")
```

# Secondary Sets
### Do the same procedure for kappa's and labeling

# Dataset 0


```python
df_0 = pd.read_csv('changeorg_topic_0.csv')
```


```python
df_0.fillna('missing', inplace=True)
```


```python
df_0_lockdown = pd.read_csv('changeorg_topic_0_lockdowns.csv')
```


```python
df_0_mask = pd.read_csv('changeorg_topic_0_masking_and_distancing.csv')
```


```python
df_0_vaccination = pd.read_csv('changeorg_topic_0_vaccination.csv')
```


```python
#df_0.head()
```




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
      <th>text</th>
      <th>annotation_43</th>
      <th>annotation_44</th>
      <th>annotation_45</th>
      <th>annotation_46</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Save Western Ghats from annihilation</td>
      <td>lockdowns</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Save Beacon Hill Park</td>
      <td>lockdowns</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Government of India: Don't Tax Medical Bills</td>
      <td>lockdowns</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NO FAILING @ UCR</td>
      <td>lockdowns</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WIAA - Let them play football!</td>
      <td>lockdowns</td>
      <td>masking and distancing</td>
      <td>lockdowns</td>
      <td>missing</td>
    </tr>
  </tbody>
</table>
</div>



# Dataset 1


```python
df_1 = pd.read_csv('changeorg_topic_1.csv')
```


```python
df_1.fillna('missing', inplace=True)
```


```python
df_1_lockdown = pd.read_csv('changeorg_topic_1_lockdowns.csv')
```


```python
df_1_mask = pd.read_csv('changeorg_topic_1_masking_and_distancing.csv')
```


```python
df_1_vaccination = pd.read_csv('changeorg_topic_1_vaccination.csv')
```


```python
#df_1.head()
```




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
      <th>text</th>
      <th>annotation_73</th>
      <th>annotation_74</th>
      <th>annotation_75</th>
      <th>annotation_76</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COVID-19 Awards</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cancel the 2020-2021 school year SOLs</td>
      <td>lockdowns</td>
      <td>lockdowns</td>
      <td>missing</td>
      <td>lockdowns</td>
    </tr>
    <tr>
      <th>2</th>
      <td>..</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Protect our nurses!</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SENSELESS IMPRISONMENT FOR MY MOM.. FIRST TIME...</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>lockdowns</td>
    </tr>
  </tbody>
</table>
</div>



# Dataset 2


```python
df_2 = pd.read_csv('changeorg_topic_2.csv')
```


```python
df_2.fillna('missing', inplace=True)
```


```python
df_2_lockdown = pd.read_csv('changeorg_topic_2_lockdowns.csv')
```


```python
df_2_mask = pd.read_csv('changeorg_topic_2_masking_and_distancing.csv')
```


```python
df_2_vaccination = pd.read_csv('changeorg_topic_2_vaccination.csv')
```


```python
#df_2.head()
```




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
      <th>text</th>
      <th>annotation_48</th>
      <th>annotation_49</th>
      <th>annotation_50</th>
      <th>annotation_47</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Postpone CBSE Board Exams</td>
      <td>lockdowns</td>
      <td>missing</td>
      <td>missing</td>
      <td>lockdowns</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pay rise for paramedics and nurses</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Too Soon To Open Georgia!</td>
      <td>lockdowns</td>
      <td>lockdowns</td>
      <td>lockdowns</td>
      <td>lockdowns</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Request to reconsider DPS Ruby Park school tui...</td>
      <td>lockdowns</td>
      <td>missing</td>
      <td>missing</td>
      <td>lockdowns</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Covid-19 Aesthetics / Salon / Massage Industry...</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>lockdowns</td>
    </tr>
  </tbody>
</table>
</div>



# Dataset 3


```python
df_3 = pd.read_csv('changeorg_topic_3.csv')
```


```python
df_3.fillna('missing', inplace=True)
```


```python
df_3_lockdown = pd.read_csv('changeorg_topic_3_lockdowns.csv')
```


```python
df_3_mask = pd.read_csv('changeorg_topic_3_masking_and_distancing.csv')
```


```python
df_3_vaccination = pd.read_csv('changeorg_topic_3_vaccination.csv')
```


```python
#df_3.head()
```




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
      <th>text</th>
      <th>annotation_112</th>
      <th>annotation_109</th>
      <th>annotation_110</th>
      <th>annotation_111</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Get CoVID Vaccines to Nepal ASAP and prevent a...</td>
      <td>vaccination</td>
      <td>vaccination</td>
      <td>vaccination</td>
      <td>vaccination</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Compensate Essential Workers</td>
      <td>missing</td>
      <td>missing</td>
      <td>vaccination</td>
      <td>masking and distancing,lockdowns</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amend Recent COVID-19 Restrictions for Persona...</td>
      <td>missing</td>
      <td>lockdowns</td>
      <td>lockdowns</td>
      <td>masking and distancing</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Petition against COVID-19 &amp; Quarantine Bill: S...</td>
      <td>lockdowns</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KCL Has Coronavirus Case: Petition to Have Onl...</td>
      <td>lockdowns</td>
      <td>lockdowns</td>
      <td>lockdowns</td>
      <td>missing</td>
    </tr>
  </tbody>
</table>
</div>



# Dataset 4


```python
df_4 = pd.read_csv('changeorg_topic_4.csv')
```


```python
df_4.fillna('missing', inplace=True)
```


```python
df_4_lockdown = pd.read_csv('changeorg_topic_4_lockdowns.csv')
```


```python
df_4_mask = pd.read_csv('changeorg_topic_4_masking_and_distancing.csv')
```


```python
df_4_vaccination = pd.read_csv('changeorg_topic_4_vaccination.csv')
```


```python
#df_4.head()
```




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
      <th>text</th>
      <th>annotation_96</th>
      <th>annotation_53</th>
      <th>annotation_93</th>
      <th>annotation_94</th>
      <th>annotation_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Language Education in the Time of COVID-19</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COVID-19 Test Kits</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COVID 19 IN PRISON</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Get Waled Home</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Make pass/fail available for Mississippi State...</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
      <td>missing</td>
    </tr>
  </tbody>
</table>
</div>



### Dataset 4 has five annotators, change some of the code while looking at the kappa

## Finding Kappa


```python
df_cur = df_2

kappa_1 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/3

kappa_2 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/3

kappa_3 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/3

kappa_4 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_0.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/3
```


```python
#optimized for dataset 4 (has 5 annotators)

df_cur = df_4

kappa_1 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 1], df_cur.iloc[:, 5], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/4

kappa_2 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 2], df_cur.iloc[:, 5], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/4

kappa_3 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 3], df_cur.iloc[:, 5], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/4

kappa_4 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_0.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 4], df_cur.iloc[:, 5], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/4

kappa_5 = (sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 5], df_cur.iloc[:, 1], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          + sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 5], df_0.iloc[:, 2], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 5], df_cur.iloc[:, 3], 
                                             labels=None, weights=None, 
                                             sample_weight=None)
          +sklearn.metrics.cohen_kappa_score(df_cur.iloc[:, 5], df_cur.iloc[:, 4], 
                                             labels=None, weights=None, 
                                             sample_weight=None))/4

```


```python
print('kappa 1:', kappa_1)
print('kappa 2:', kappa_2)
print('kappa 3:', kappa_3)
print('kappa 4:', kappa_4)
print('kappa 4:', kappa_5)
```

    kappa 1: 0.3056438218179325
    kappa 2: 0.0479864857235143
    kappa 3: 0.230724059332207
    kappa 4: 0.315059519844437
    kappa 4: 0.3070102812116629


# Annotators, kappa's

## df_0: 
### 3 Annotators. Annotator 43 is unreliable.

kappa 43: 0.0729400722183449

kappa 44: 0.3653958972415916

kappa 45: 0.3950997775171102

kappa 46: 0.38086750540434416


## df_1
### 4 Annotators, keep all.

kappa 73: 0.5998101568552087

kappa 74: 0.6064004363143914

kappa 75: 0.4588812351405374

kappa 76: 0.4014548691920357


## df_2
### 4 Annotators, keep all.

kappa 48: 0.4125172568390408

kappa 49: 0.4479413860850894

kappa 50: 0.454406657189879

kappa 47: 0.2803217338357632

## df_3
### 3 Annotators Annotator 111 is unreliable.

kappa 112: 0.3744886183380893

kappa 109: 0.41148335806861763

kappa 110: 0.32150079175224594

kappa 111: 0.17381735266047096

## df_4
### 4 Annotators. Annotator 53 is unreliable.

kappa 96: 0.3056438218179325

kappa 53: 0.0479864857235143

kappa 93: 0.230724059332207

kappa 94: 0.315059519844437

kappa 95: 0.3070102812116629


## Drop unreliable annotators


```python
df_0 = df_0.drop(columns=['annotation_43'])
df_0_lockdown = df_0_lockdown.drop(columns=['annotation_43'])
df_0_mask = df_0_mask.drop(columns=['annotation_43'])
df_0_vaccination = df_0_vaccination.drop(columns=['annotation_43'])
```


```python
#df_0_vaccination.head()
```




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
      <th>text</th>
      <th>annotation_44</th>
      <th>annotation_45</th>
      <th>annotation_46</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Save Western Ghats from annihilation</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Save Beacon Hill Park</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Government of India: Don't Tax Medical Bills</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NO FAILING @ UCR</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WIAA - Let them play football!</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_3 = df_3.drop(columns=['annotation_111'])
df_3_lockdown = df_3_lockdown.drop(columns=['annotation_111'])
df_3_mask = df_3_mask.drop(columns=['annotation_111'])
df_3_vaccination = df_3_vaccination.drop(columns=['annotation_111'])
```


```python
#df_3.head()
```




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
      <th>text</th>
      <th>annotation_112</th>
      <th>annotation_109</th>
      <th>annotation_110</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Get CoVID Vaccines to Nepal ASAP and prevent a...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Compensate Essential Workers</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amend Recent COVID-19 Restrictions for Persona...</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Petition against COVID-19 &amp; Quarantine Bill: S...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KCL Has Coronavirus Case: Petition to Have Onl...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_4 = df_4.drop(columns=['annotation_53'])
df_4_lockdown = df_4_lockdown.drop(columns=['annotation_53'])
df_4_mask = df_4_mask.drop(columns=['annotation_53'])
df_4_vaccination = df_4_vaccination.drop(columns=['annotation_53'])
```


```python
#df_4_lockdown
```




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
      <th>text</th>
      <th>annotation_96</th>
      <th>annotation_93</th>
      <th>annotation_94</th>
      <th>annotation_95</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Language Education in the Time of COVID-19</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COVID-19 Test Kits</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COVID 19 IN PRISON</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Get Waled Home</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Make pass/fail available for Mississippi State...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>Cancellation of Lebanese Official Exams for 20...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Hazard pay for essential employees working wit...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>297</th>
      <td>Stop children being weighed in school without ...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Allow Visitors for Covid-19 Patients</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Exigimos acciones de control y salud en la Res...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 6 columns</p>
</div>



# Labeling The Datasets
### Do it for each dataset

### For 4 annotations


```python
df_cur = df_4_lockdown
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 5] = True
    elif label_true < label_false:
        df_cur.iloc[i, 5] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 5] = True if kappa_scores>=0 else False
        
df_4_lockdown = df_cur
```


```python
#df_4_lockdown
```




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
      <th>text</th>
      <th>annotation_96</th>
      <th>annotation_93</th>
      <th>annotation_94</th>
      <th>annotation_95</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Language Education in the Time of COVID-19</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COVID-19 Test Kits</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COVID 19 IN PRISON</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Get Waled Home</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Make pass/fail available for Mississippi State...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>Cancellation of Lebanese Official Exams for 20...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Hazard pay for essential employees working wit...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>297</th>
      <td>Stop children being weighed in school without ...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Allow Visitors for Covid-19 Patients</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Exigimos acciones de control y salud en la Res...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 6 columns</p>
</div>




```python
df_cur = df_4_mask
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 5] = True
    elif label_true < label_false:
        df_cur.iloc[i, 5] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 5] = True if kappa_scores>=0 else False
        
df_4_mask = df_cur
```


```python
df_cur = df_4_vaccination
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 5] = True
    elif label_true < label_false:
        df_cur.iloc[i, 5] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 5] = True if kappa_scores>=0 else False
        
df_4_vaccination = df_cur
```

### For 3 annotations


```python
df_cur = df_3_lockdown
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 4] = True
    elif label_true < label_false:
        df_cur.iloc[i, 4] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 4] = True if kappa_scores>=0 else False
        
df_3_lockdown = df_cur
```


```python
df_cur = df_3_mask
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 4] = True
    elif label_true < label_false:
        df_cur.iloc[i, 4] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 4] = True if kappa_scores>=0 else False
        
df_3_mask = df_cur
```


```python
df_cur = df_3_vaccination
df_cur['label'] = None

for i in range(len(df_cur)):
    label_true = 0
    label_false = 0
    
    for col in df_cur.loc[i][1:]:
        if col:
            label_true+=1
        else:
            label_false+=1
    
    if label_true > label_false:
        df_cur.iloc[i, 4] = True
    elif label_true < label_false:
        df_cur.iloc[i, 4] = False
    else: #labels are same
        kappa_scores = 0
        kappa_scores += kappa_1 if df_cur.loc[i][1] else -kappa_1
        kappa_scores += kappa_2 if df_cur.loc[i][1] else -kappa_2
        kappa_scores += kappa_3 if df_cur.loc[i][1] else -kappa_3
        kappa_scores += kappa_4 if df_cur.loc[i][1] else -kappa_4

        df_cur.iloc[i, 4] = True if kappa_scores>=0 else False
        
df_3_vaccination = df_cur
```

# Combine Datasets
### Combine each label type to itself

### Lockdown


```python
df_lockdown_secondary = df_0_lockdown[['text', 'label']]
df_lockdown_secondary = df_lockdown_secondary.append(df_1_lockdown[['text','label']], ignore_index = True)
df_lockdown_secondary = df_lockdown_secondary.append(df_2_lockdown[['text','label']], ignore_index = True)
df_lockdown_secondary = df_lockdown_secondary.append(df_3_lockdown[['text','label']], ignore_index = True)
df_lockdown_secondary = df_lockdown_secondary.append(df_4_lockdown[['text','label']], ignore_index = True)
```


```python
df_lockdown_secondary.head(-20)
```




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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Save Western Ghats from annihilation</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Save Beacon Hill Park</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Government of India: Don't Tax Medical Bills</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NO FAILING @ UCR</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WIAA - Let them play football!</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1475</th>
      <td>Allow parents to temporarily home educate with...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1476</th>
      <td>There will be a day AFTER COVID-19.  Help us c...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1477</th>
      <td>Close Madison City Schools due to COVID-19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1478</th>
      <td>Mthuli Ncube to remove 2% charge from all elec...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>Partially Refund Loughborough University Tuiti...</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1480 rows × 2 columns</p>
</div>




```python
df_lockdown_secondary.to_csv("changeorg_topic_lockdown.csv")
```

### Masking and Distancing


```python
df_mask_secondary = df_0_mask[['text', 'label']]
df_mask_secondary = df_mask_secondary.append(df_1_mask[['text','label']], ignore_index = True)
df_mask_secondary = df_mask_secondary.append(df_2_mask[['text','label']], ignore_index = True)
df_mask_secondary = df_mask_secondary.append(df_3_mask[['text','label']], ignore_index = True)
df_mask_secondary = df_mask_secondary.append(df_4_mask[['text','label']], ignore_index = True)
```


```python
df_mask_secondary.head()
```




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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Save Western Ghats from annihilation</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Save Beacon Hill Park</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Government of India: Don't Tax Medical Bills</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NO FAILING @ UCR</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WIAA - Let them play football!</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_mask_secondary.to_csv("changeorg_topic_masking_and_distancing.csv")
```

### Vaccination


```python
df_vaccination_secondary = df_0_vaccination[['text', 'label']]
df_vaccination_secondary = df_vaccination_secondary.append(df_1_vaccination[['text','label']], ignore_index = True)
df_vaccination_secondary = df_vaccination_secondary.append(df_2_vaccination[['text','label']], ignore_index = True)
df_vaccination_secondary = df_vaccination_secondary.append(df_3_vaccination[['text','label']], ignore_index = True)
df_vaccination_secondary = df_vaccination_secondary.append(df_4_vaccination[['text','label']], ignore_index = True)
```


```python
df_vaccination_secondary.head()
```




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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Save Western Ghats from annihilation</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Save Beacon Hill Park</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Government of India: Don't Tax Medical Bills</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NO FAILING @ UCR</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WIAA - Let them play football!</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_vaccination_secondary.to_csv("changeorg_topic_vaccination.csv")
```


```python

```
