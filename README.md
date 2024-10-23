## Negative Sampling in Next-POI Recommendations: Observation, Approach, and Evaluation

## References

```
 1) Yu, F., Cui, L., Guo, W., Lu, X., Li, Q., & Lu, H. (2020, April). A category-aware deep model for successive POI recommendation on sparse check-in data. In Proceedings of the web conference 2020 (pp. 1264-1274).
 
 2) Luo, Y., Liu, Q., & Liu, Z. (2021, April). Stan: Spatio-temporal attention network for next location recommendation. In Proceedings of the Web Conference 2021 (pp. 2177-2185).
 
 3) Wu, Y., Li, K., Zhao, G., & Qian, X. (2020). Personalized long-and short-term preference learning for next POI recommendation. IEEE Transactions on Knowledge and Data Engineering, 34(4), 1944-1957.
 
 4) Lian, D., Wu, Y., Ge, Y., Xie, X., & Chen, E. (2020, August). Geography-aware sequential location recommendation. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 2009-2019).
 
 5) Chen, W., Wan, H., Guo, S., Huang, H., Zheng, S., Li, J., ... & Lin, Y. (2022). Building and exploiting spatial–temporal knowledge graph for next POI recommendation. Knowledge-Based Systems, 258, 109951.
 
```

## Installation

- python 3.7
- pandas 1.1.5
- tensorflow 1.15.0
- torch 1.13.1
- scikit-learn 1.0.2
- torchtext 0.6.0
- numpy 1.23

## Data
The pre-processed data will be released together when the paper is accepted.

### 1) Original Data

   If you need the complete NYC and TKY data set, please click the link below to download.

  - [Downloads NYC,TKY Data](http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip): the Foursquare check-ins for NYC, TKY
  - [Downloads BrightKite Data](https://snap.stanford.edu/data/loc-brightkite.html): Brightkite check-ins

### 2) Preprocessing code

1) ***About Original Data***
   - The original dataset includes long-term (about 10 months) check-in data in New York city and Tokyo collected from Foursquare from 12 April 2012 to 16 February 2013.
      It contains two files in csv format. Each file contains 8 columns, which are:
      ```
     1. User ID (anonymized)
     2. Venue ID (Foursquare)
     3. Venue category ID (Foursquare)
     4. Venue category name (Fousquare)
     5. Latitude
     6. Longitude
     7. Timezone offset in minutes (The offset in minutes between when this check-in occurred and the same time in UTC)
     8. UTC time
     ``` 
     
2) ***For all methods with NYC, TKY data***
   - This is the code for preprocessing data according to the preprocessing method mentioned in our paper. 
      >    Go to file: [preprocessing/for_all_preprocessing.py](preprocessing/for_all_preprocessing.py)
     
  
3) ***For CatDM***
   - To better predict the successful check-ins within 24 hours, which is one of the main ideas of catdm, we created a preprocessing code that sets a ground truth of 24 hours for catdm.
      > Go to file: [preprocessing/for_catdm.py](preprocessing/for_catdm.py)
  
4) ***For PLSPL***
   - This is a code that creates a file storing the positive POIs visited by each user within the user-specific training dataset, ensuring that they are not selected as negative POIs.
      (Other methods generate this data within each code for experimentation.)
        > Go to file: [preprocessing/for_plspl_make_history_uid_loc.py](preprocessing/for_plspl_make_history_uid_loc.py)
     
5) ***For GeoSAN***
   - This is the code for converting data to the format used by Geosan.
        > Go to file: [preprocessing/for_geosan.py](preprocessing/for_geosan.py)
     
6) ***For STAN***
   - This is the code for converting data to the format used by STAN.
        > Go to file: [preprocessing/for_stan.py](preprocessing/for_stan.py)
     
7) ***For all methods with brightkite data***
   - This is the code for preprocessing data according to the preprocessing method mentioned in our paper. 
      >    Go to file: [preprocessing/for_all_brightkite.py](preprocessing/for_all_brightkite.py)

8) ***For catdm, plspl with brightkite data***
   - Since POI categories are not available in  Brightkite, we collect the POI categories from public Foursquare API, and perform POI category matching as in (Yu, Wang, and Li 2018). 
   - reference: Qin, G., Song, L., Yu, Y., Huang, C., Jia, W., Cao, Y., & Dong, J. (2023, June). Graph Structure Learning on User Mobility Data for Social Relationship Inference. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 4, pp. 4578-4586).
      >    Go to file: [preprocessing/for_catdm_brightkite.py](preprocessing/for_catdm_brightkite.py)

      >    Go to file: [preprocessing/for_catdm_plspl_brightkite.py](preprocessing/for_catdm_plspl_brightkite.py)


     
## How to RUN the file

## [1] CatDM 

### 1. Generate Candidates

To filter POIs and reduce the search space.
```bash
$ python CatDM/"NYC or TKY or Bri"/model/CatDM/train.py
```
To train and evaluate Encoder 1 and Filter, we split each dataset into a training set, a validation set and a test set, here. Encoder 1 and filtering layers form a reasonable filter capable of reducing search space, i.e., reducing the number of candidates from which recommended POIs are selected finally.
Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test.

### 2-1) Train with RNS
Train with RNS for ranking model.
```bash
$ python CatDM/"NYC or TKY or Bri"/model/RNS/train_rankpoi.py
```
Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test.

### 2-2) Train with DHNS
Train with DHNS for ranking model.
```bash
$ python CatDM/"NYC or TKY or Bri"/model/DHNS/train_rankpoi.py
```

### 2-3) Train with DHNSDist
Train with DHNS/Distance for ranking model.
```bash
$ python CatDM/"NYC or TKY or Bri"/model/CatDM/train_RNS.py
```

### 3) Test
Change 'save_dir' and test the saved model.
```bash
$ python CatDM/"NYC or TKY or Bri"/model/"RNS or DHNS or DHNSDist"/train_rankpoi.py
```

## [2] PLSPL


### 1-1) Train with RNS
Train with RNS.
```bash
$ python PLSPL/NYC or TKY or Bri/train_RNS.py
```

### 1-2) Train with DHNS
Train with DHNS.

```bash
$ python PLSPL/NYC or TKY or Bri/train_DHNS.py
```

### 1-3) Train with DHNSDist
Train with DHNS/Distance.
```bash
$ python PLSPL/NYC or TKY or Bri/train_DHNSDist.py
```

### 2) Test
Change the checkpoint file path to the epoch with the highest performance in the validation data and test it.
```bash
$ python PLSPL/NYC or TKY or Bri/test.py
```

## [3] GeoSAN

***This method reduces the experimental accuracy when the batch size is small. Be sure to set the batch size to 32 or more for both rns and dns methods and experiment.***

### 1. quadkey_dataset

To make quadkey dataset.
```bash
$ python GeoSAN/quadkey_dataset.py
```

### 2. near_loc_query
In the Geosan method, a tree of nearby locations is constructed using KNN (K-Nearest Neighbors) to facilitate the recommendation process.

```bash
$ python GeoSAN/near_loc_query.py
```

### 3-1) Train with RNS
Train with RNS.
```bash
$ python GeoSAN/train_RNS.py
```
Note that the value of variable 'tf.flags.DEFINE_string' can be selected by train or test.

### 3-2) Train with DHNS
Train with DHNS.
```bash
$ python GeoSAN/train_DHNS.py
```

### 3-3) Train with DHNSDist
Train with DHNS/Distance.
```bash
$ python GeoSAN/train_DHNSDist.py
```

### 3) Test
If you input the path of the saved model with the highest accuracy into "load_path", it will be evaluated.

## [4] STAN

### 1. load.py

To pkl file.
```bash
$ python STAN/load.py
```

### 2-1) Train with RNS
Divide the dataset into different proportions of users to test the performance and then average. 
And you set "part=100" in train_rns(or DHNS or DHNSDist).py, which means you use first 100 users to test performance.
 In this way, all the users is divided into different proportions ([:100], [100,200], [200,300],...[900,1000]) and avarage the results.
see detail in https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation/issues/11#issuecomment-955652797
For example, if the total number of users is 250, the parts would be 100, 200, -1, and when the part is -1, you should set st to 200.
Train with RNS .
```bash
$ python STAN/train_RNS.py
```


### 2-2) Train with DHNS
Train with DHNS.
```bash
$ python STAN/train_DHNS.py
```

### 2-3) Train with DHNSDist
Train with DHNS/Distance.
```bash
python STAN/train_DHNSDist.py
```

## [5] STKGRec

### 1. data_pre.py

To pkl file.
```bash
$ python STKGRec/data_pre.py
```

### 2-1) Train with RNS
```bash
$ python STKGRec/train_RNS.py
```

### 2-2) Train with DHNS
Train with DHNS.
```bash
$ python STKGRec/train_DHNS.py
```

### 2-3) Train with DHNSDist
Train with DHNS/Distance.
```bash
python STKGRec/train_DHNSDist.py
```

