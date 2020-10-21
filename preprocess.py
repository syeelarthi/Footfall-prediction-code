public class pre_processing:
  def __init__(self,data_frame):
        self.data_frame = data_frame # which data_frame to pre-process
        self.location_name_list = []
        self.location_name_col_list=[]
        self.scaler_target = MinMaxScaler(feature_range = (0, 1))
        self.scaler_input = MinMaxScaler(feature_range = (0,1))
        self.scaler_target_each_location = MinMaxScaler(feature_range = (0,1))
  def cleanup_data(self, col_name='Date'):
     # there is a white space in one of the date values in Easter holidays list. We need to remove that
     if( col_name in self.data_frame.columns):
        self.data_frame[col_name] = self.data_frame[col_name].str.replace(' ', '')
     return self.data_frame 
  def aggregate(self,group_by_cols, retain_list=None): 
      '''
      aggregate the column values based on conditions
      ''' 
      retain_list = {}
      # add all the location name columns to retain list
      for location_name in self.location_name_list:
        key = 'InCount'+location_name
        retain_list[key] = 'sum'
      # now aggregate based on Date
      self.data_frame = self.data_frame.groupby(group_by_cols).agg(retain_list).reset_index()
      return self.data_frame

  def reshape_location_name(self, group_by_cols, target_col='InCount'):# Review this
      
      '''data_temp_df = pd.DataFrame(data=self.data_frame[['Date','Weekday','Month']])
      data_temp_df.drop_duplicates('Date', inplace=True)
      
      self.location_name_list = self.data_frame["LocationName"].unique()

      self.data_frame = self.data_frame.pivot_table(index=group_by_cols, columns='LocationName',values=[target_col]).reset_index()
      self.data_frame.columns = [''.join(col) for col in self.data_frame.columns]'''

      self.data_frame['Date'] = pd.to_datetime(self.data_frame['Date']).dt.strftime('%d-%b-%Y')
      self.location_name_list = self.data_frame["LocationName"].unique()

      self.data_frame = self.data_frame.pivot_table(index=group_by_cols, columns='LocationName',values=[target_col]).reset_index()
      self.data_frame.columns = [''.join(col) for col in self.data_frame.columns]

      #return self.data_frame.merge(data_temp_df, on='Date', how='inner')

  def encode_variables(self,included_cols, target_variable = 'InCountTotal', encoding_type = 'target_encoding'):
      '''
      categorical/target encoding of the column values
      '''
      self.data_frame = MultiColumnEncoder(includedcolumns = included_cols).fit_transform(self.data_frame,target_variable,encoding_type = 'target_encoding')
      return self.data_frame
  def imputeData(self): 
      # fill missing values with mean column values as only missing column i see is InCountAlbionStSouth
      #print('impute data')
      self.data_frame.fillna(self.data_frame.mean(), inplace=True)
      return self.data_frame
  def pre_process(self,data_frame_type='Foot_Fall'):
      '''
      for different data frame types, do the different pre-processibg
      ''' 
      
      if(data_frame_type == 'EasterSunday'):
          # cleanup the date columns
          self.data_frame  = self.cleanup_data(col_name='Date')
      
      return self.data_frame
       
  
  def merge_data(self, data_to_merge, col_by ='Date'):
      data_to_merge['Date'] = pd.to_datetime(data_to_merge['Date']).dt.strftime('%d-%b-%Y')
      self.data_frame = self.data_frame.merge(data_to_merge, how='inner', left_on=col_by, right_on=col_by)
      return self.data_frame
  def add_new_column(self, data_to_add, new_col_name, source_date_format = '%d-%b-%Y', destination_date_format = '%m-%d-%Y' ):
      self.data_frame['Tempcolumn'] = pd.to_datetime(self.data_frame['Date'], format = source_date_format).dt.strftime('%d/%m/%Y')

      temp_data = data_to_add.copy()

      temp_data['Tempcolumn'] = pd.to_datetime(temp_data['Date'],format = destination_date_format).dt.strftime('%d/%m/%Y')

      self.data_frame[new_col_name] = (self.data_frame.set_index(['Tempcolumn']).index.isin(temp_data.set_index(['Tempcolumn']).index)).astype(int)
      temp_data.drop(['Tempcolumn'],axis=1)

      self.data_frame.drop(['Tempcolumn'],axis=1, inplace=True)
      
      return self.data_frame

  def createTimeSeries(self):
      self.data_frame["DateTime"] = pd.to_datetime(self.data_frame.Date.astype(str))
      self.data_frame = self.data_frame.set_index("DateTime")
      self.data_frame.drop(['Date'], inplace=True, axis=1)
      self.data_frame = self.data_frame.sort_index()

      self.data_frame['Month']=self.data_frame.index.month
      self.data_frame['Year']=self.data_frame.index.year
      self.data_frame['Day']= self.data_frame.index.day
      self.data_frame['Weekday']=self.data_frame.index.weekday
      self.data_frame['Week_no']=self.data_frame.index.week
      
      for i in range(len(self.data_frame)):
          self.data_frame['Weekday'].iloc[i]=calendar.day_name[self.data_frame['Weekday'].iloc[i]]

      return self.data_frame
      
  def createStationarySeries(self, col_name = 'InCountTotal', lag=1):
    # check for stationarity

    result = adfuller(self.data_frame[col_name].dropna())
    print("ADF test results for ",col_name )
    print('ADF Statistic: %f' % result[0])
    print('p-value: ', result[1])
    print('full result ', result)
    print("========================")

    for location in self.location_name_col_list:
      result = adfuller(self.data_frame[location].dropna())
      print("ADF test results for ",location )
      print('ADF Statistic: %f' % result[0])
      print('p-value: ', result[1])
      print('full result ', result)
      print("========================")

    #self.data_frame[col_name] = self.data_frame[col_name].diff(periods=lag)
    #self.data_frame.dropna(inplace = True)
    return self.data_frame

  def checkForMissingValues(self, location_name_list):
        for location in location_name_list:
            self.data_frame.loc[self.data_frame[self.data_frame['InCount'+location]==0].index,'InCount'+location] = np.NaN
        return self.data_frame
  
  def createLag_ForecastVariables(self,location_name_list=None, n_timesteps=7,n_forecast_steps = 1, location_wise_data = True):
      nLagClass = Lagging(location_name_list,n_timesteps, n_forecast_steps)
      #self.data_frame = nLagClass.createLagVariables(self.data_frame)
      # create lag data for total mall incount
      self.data_frame = nLagClass.createLagDiffVariables(self.data_frame, location_wise_data = location_wise_data)
      self.data_frame = nLagClass.createforecastTargetVariables(self.data_frame)
      
      return self.data_frame

  def addTotalIncountColumn(self):
    self.location_name_col_list=['InCount'+x for x in self.location_name_list]
    self.data_frame['InCountTotal'] = self.data_frame[self.location_name_col_list].sum(axis=1)
    self.data_frame['InCountTotal'] = self.data_frame['InCountTotal'].astype(int)
    return self.data_frame

  def normalizeTargetVariable(self):
    #print('normalization')
    self.data_frame[['InCountTotal']] = self.scaler_target.fit_transform(self.data_frame[['InCountTotal']])

    location_name_col_list=['InCount'+x for x in self.location_name_list]
    for location_col in location_name_col_list:
      self.data_frame[[location_col]] = self.scaler_target_each_location.fit_transform(self.data_frame[[location_col]])
    return self.data_frame

  
  def normalizeInputVariables(self, included_cols=None):
     #print('normalize input variables')
     self.data_frame[included_cols] = self.scaler_input.fit_transform(self.data_frame[included_cols])
     return self.data_frame

  def visualizeMissingValues(self):
       #print('missing values visualization')
       msno.bar(self.data_frame)
        
  #Imputation using KNN      
  def imputeDataWithKnn(self):
    X = self.data_frame[['InCountCommercialStBarratts', 'InCountCommercialStLush']]
    imputer = KNNImputer(n_neighbors=2)
    imputed_df = imputer.fit_transform(X)
    imputed_df = pd.DataFrame(imputed_df, columns=X.columns)
    #print(imputed_df)
    #print(self.data_frame['InCountCommercialStBarratts'])
    self.data_frame['InCountCommercialStBarratts'] = imputed_df.values
    return self.data_frame

  #imputation using IterativeImputer which works similar to MICE
  def imputeDataWithIterativeImputation(self):
    X = self.data_frame[['InCountCommercialStBarratts', 'InCountCommercialStLush']]
    imputer = IterativeImputer(max_iter=10, verbose=0)
    imputed_df = imputer.fit_transform(X)
    imputed_df = pd.DataFrame(imputed_df, columns=X.columns)
    #print(imputed_df)
    #print(self.data_frame['InCountCommercialStBarratts'])
    self.data_frame['InCountCommercialStBarratts'] = imputed_df.values
    return self.data_frame 


  def findoutliers_DBSCAN(self,columns,number_near):
      # This finction takes as input, the dataframe name, columns to be analysed for the outlier, nearesr neighbours to be considered as input
      # A new boolean column for outlier will be added with name Outlier_+column name and it will have values 1  If outlier and 0 otherwise
      # optimal eps is calculated based on distances
      nc=0
      fig, axs = plt.subplots(len(columns),figsize=(15,15))
      fig.tight_layout(pad=3.0)
      for col in columns:
          # finding the distances based on nearest neighbour to find optimal eps
          neigh = NearestNeighbors(n_neighbors=number_near)
          nbrs = neigh.fit(np.array(self.data_frame[col]).reshape(-1,1))
          distances, indices = nbrs.kneighbors(np.array(self.data_frame[col]).reshape(-1,1))
          distances = np.sort(distances, axis=0)
          distances = distances[:,1]
      
          # If plotted on a graph, a point with maximum curvature. , ie, a point with maximum slope gives the best eps
          maxslope=0
          neps=0
          diff=[] # A list to hold the differences in the distances
          for i in range (len(distances)-1):
              slope=( (distances[i]-distances[i+1])/(indices[i][0]-indices[i+1][0]))
              if (slope<1 and maxslope<slope):
                  maxslope=slope
                  neps=distances[i]
          db=DBSCAN(eps=neps, min_samples=number_near).fit(np.array(self.data_frame[col]).reshape(-1,1))
          labels=db.labels_
          self.data_frame['Outliers_'+col] = [1 if val ==-1 else 0 for val in labels]
          # Visualizing the Outliers marked   
          outlier=[]
          outlierindex=[]
          datapoint=[]
          dataindex=[]
          for j in range(len(labels)):
              if (labels[j]==-1):
                  outlier.append(self.data_frame[col][j])
                  outlierindex.append(indices[j][0])
              else:
                  datapoint.append(self.data_frame[col][j])
                  dataindex.append(indices[j][0])
          axs[nc].scatter(outlierindex,outlier,color='red')
          axs[nc].scatter(dataindex,datapoint,color='blue')
          axs[nc].set_title(col+" DBCSAN Outlers marked in Red")
          nc=nc+1
          n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
          n_noise_ = list(labels).count(-1)
          print(col+'Analysis')
          print('Optimal eps is ',neps)
          print('Estimated number of clusters: %d' % n_clusters_)
          print('Estimated number of noise points: %d' % n_noise_)
      plt.show
      return self.data_frame
  def findoutliers_lof(self,columns,number_near):
      # This finction takes as input, the dataframe name, columns to be analysed for the outlier, nearesr neighbours to be considered as input
      # A new boolean column for outlier will be added with name Outlier_+column name and it will have values 1  If outlier and 0 otherwise
  
      nc=0
      fig, axs = plt.subplots(len(columns),figsize=(15,15))
      fig.tight_layout(pad=3.0)
      for col in columns:
          clf = LocalOutlierFactor(n_neighbors=number_near)
          pred=clf.fit_predict(np.array(self.data_frame[col]).reshape(-1,1))
          #scores = clf.negative_outlier_factor_
          self.data_frame['Lof_Outlier'+col]=[1 if val ==-1 else 0 for val in pred]
    
          # Visualizing the Outliers marked   
          outlier=[]
          outlierindex=[]
          datapoint=[]
          dataindex=[]

          for j in range(len(self.data_frame['Lof_Outlier'+col])):
              if (self.data_frame['Lof_Outlier'+col][j]==1):
                  outlier.append(self.data_frame[col][j])
                  outlierindex.append(j+1)
              else:
                  datapoint.append(self.data_frame[col][j])
                  dataindex.append(j+1)

          axs[nc].scatter(dataindex,datapoint,color='blue')
          axs[nc].scatter(outlierindex,outlier,color='red')
          axs[nc].set_title(col+" LOF Outlers marked in Red")
          nc=nc+1
      return self.data_frame

  def detrend_time_series(self, target_col='InCountTotal'):# using signal.detrend()
    inCountData = self.data_frame[target_col]
    self.data_frame[target_col] = signal.detrend(self.data_frame[target_col])
    de_trended_Incount = self.data_frame[target_col]

    from matplotlib import pyplot as plt
    plt.figure(figsize=(16, 8))
    plt.plot(inCountData, label="normal InCount")
    plt.plot(de_trended_Incount, label="detrended InCount")
    plt.legend(loc='best')
    plt.show()
    return self.data_frame 

  def decomposeTimeSeries(self):

    # Multiplicative Decomposition 
    result_mul = seasonal_decompose(self.data_frame['InCountTotal'], model='multiplicative',  freq=30)
    # Additive Decomposition
    result_add = seasonal_decompose(self.data_frame['InCountTotal'], model='additive',  freq=30)

    detrended = self.data_frame['InCountTotal'].values - result_add.trend - result_add.seasonal

    # Plot
    plt.rcParams.update({'figure.figsize': (16,8)})
    result_add.plot().suptitle('Additive Decompose using seasonal_decompose', fontsize=22)
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.plot(detrended,label='detrended')
    plt.plot(self.data_frame['InCountTotal'], label="normal InCount")
    plt.show()

    plt.figure(figsize=(16, 8))
    result_add.resid.plot(kind='kde', title='residual Density')   
    plt.show()

    self.data_frame['InCountTotal'] = detrended
    df_decomposed = pd.concat([result_add.seasonal, result_add.trend, result_add.resid, result_add.observed], axis=1)
    df_decomposed.columns = ['seas', 'trend', 'resid', 'actual_incount_total_values']
    self.data_frame = pd.concat([self.data_frame,df_decomposed], axis=1)
       
    self.data_frame.dropna(inplace = True)    

  def decomposeTime_Series_for_locations(self):

    location_name_col_list=['InCount'+x for x in self.location_name_list]
    for location_col in location_name_col_list:
      # Multiplicative Decomposition 
      result_add = seasonal_decompose(self.data_frame[location_col], model='additive',  freq=30)
      result_mul = seasonal_decompose(self.data_frame[location_col], model='multiplicative', freq=30)
      detrended = self.data_frame[location_col].values - result_add.trend - result_add.seasonal
      # Plot
      plt.rcParams.update({'figure.figsize': (16,8)})
      result_add.plot().suptitle('Additive Decompose using seasonal_decompose ' + location_col, fontsize=22)
      plt.show()
      
      plt.figure(figsize=(16, 8))
      result_add.resid.plot(kind='kde', title='residual Density for '+ location_col )   
      plt.show()

      plt.figure(figsize=(16, 8))
      plt.plot(detrended,label='detrended '+ location_col)
      plt.plot(self.data_frame[location_col], label="normal InCount for "+ location_col)
      plt.show()
      self.data_frame[location_col] = detrended

  
  def decompose_STL(self):
    
    plt.rcParams.update({'figure.figsize': (16,8)})
    res = STL(self.data_frame['InCountTotal'], period=30).fit()

    detrended = self.data_frame['InCountTotal'].values - res.trend - res.seasonal

    res.plot().suptitle('Decompose InCountTotal using STL ', fontsize=22)
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.plot(detrended, label='detrended data')
    plt.plot(self.data_frame['InCountTotal'], label='original data')
    plt.title('detrended and original data using STL')
    plt.show()

  def applypca(self,pcollist,ncomp=0,varpercent=0):
    # This function takes the dataframe on which Principal Component Analysis needs to be done and the % of variance expected to be retained
    # the function then returns the number of optimal principal components and the dataframe with those principal components.
    var_cum_variance = []
    pcadf=pd.DataFrame(self.data_frame[pcollist])
    if((ncomp>0) & (ncomp<=len(pcadf.columns))):
        pca = PCA(n_components=ncomp)
        principalComponents = pca.fit_transform(pcadf)
        var_cum_variance = np.cumsum(pca.explained_variance_ratio_)
        bestncomp=ncomp
        principalDf = (pd.DataFrame(data = principalComponents)).iloc[:,:bestncomp]
        contributionDf=pd.DataFrame(data=pca.components_,columns=pcadf.columns).loc[:bestncomp]
    elif((varpercent>0) & (varpercent<=100)):
        pca = PCA(n_components=len(pcadf.columns))
        principalComponents = pca.fit_transform(pcadf)
        varratio=pca.explained_variance_ratio_
        var_cum_variance = np.cumsum(pca.explained_variance_ratio_)
        for i in range(len(varratio)+1):
            if(varpercent>=np.sum(varratio[:i])*100):
                bestncomp=i
        principalDf = (pd.DataFrame(data = principalComponents)).iloc[:,:bestncomp]
        contributionDf=pd.DataFrame(data=pca.components_,columns=pcadf.columns).loc[:bestncomp]
    else:
        print('Incorrect parameters passed. Either numer of components- 0>ncomp>=number of columns OR the percent of variance to be retained -0 >varpercent >=100')
        bestncomp=-1
        principalDf = pd.DataFrame(data = [])    

    return bestncomp,principalDf,contributionDf, var_cum_variance

  def findoutliersfulldata(self,number_near,metrics='euclidean'):
        odf=pd.DataFrame(self.data_frame)
        ncol=len(odf.columns)
        # finding the distances based on nearest neighbour to find optimal eps
        neigh = NearestNeighbors(n_neighbors=number_near,metric=metrics)
        nbrs = neigh.fit(np.array(odf).reshape(-1,ncol))
        distances, indices = nbrs.kneighbors(np.array(odf).reshape(-1,ncol))
        distance=[]
        for i in range (len(distances)):
          distance.append(np.sum(distances[i]))
        distance = np.sort(distance, axis=0)
        #distance = distance[:,1]
        # If plotted on a graph, a point with maximum curvature. , ie, a point with maximum slope gives the best eps
        maxslope=0
        neps=0
        diff=[] # A list to hold the differences in the distances
        for i in range (len(distance)-1):
                slope=( (distance[i]-distance[i+1])/(indices[i][0]-indices[i+1][0]))
                if (slope<1 and maxslope<slope):
                    maxslope=slope
                    neps=distance[i]
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(odf)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['D1','D2'])
        
        db=DBSCAN(eps=neps, min_samples=number_near,metric=metrics).fit(np.array(odf).reshape(-1,ncol))
        labels=db.labels_
       
        clf = LocalOutlierFactor(n_neighbors=number_near)
        pred=clf.fit_predict(np.array(odf).reshape(-1,ncol))
        odf['Outliers'] = [1 if val ==-1 else 0 for val in labels]
     
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('DBSCAN Analysis on Full Data')
        print('Optimal eps is ',neps)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

        n_clusters_ = len(set(pred)) - (1 if -1 in labels else 0)
        n_noise_ = list(pred).count(-1)
        print('LOF Analysis on Full Data')
        #print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

        #principalDf=data_frame
    
        # Visualizing the Outliers marked   
        outlier=[]
        outlierindex=[]
        datapoint=[]
        dataindex=[]
        for j in range(len(labels)):
          if (labels[j]==-1):
              #outlier.append(np.sum(np.array(principalDf.iloc[j])))
              outlier.append(((principalDf.iloc[j]['D1'])))
              outlierindex.append(((principalDf.iloc[j]['D2'])))
          else:
              #datapoint.append(np.sum(np.array(principalDf.iloc[j])))
              datapoint.append(((principalDf.iloc[j]['D1'])))
              dataindex.append(((principalDf.iloc[j]['D2'])))
      
        fig = plt.figure(figsize=(20,10)) 
        plt.scatter(datapoint,dataindex,color='blue')
        plt.scatter(outlier,outlierindex,color='red')
        plt.title("DBSCAN Outlers on Full Data marked in Red")
        plt.show

        #scores = clf.negative_outlier_factor_
        print(len(pred),len(principalDf))
        #for i in range(len(pred)):
        #      print(i, pred[i])
        odf['Lof_Outlier']=[1 if val ==-1 else 0 for val in pred]
    
        # Visualizing the Outliers marked   
        outlier=[]
        outlierindex=[]
        datapoint=[]
        dataindex=[]
        for j in range(len(odf['Lof_Outlier'])):
            if (odf['Lof_Outlier'][j]==1):
              #outlier.append(np.sum(np.array(principalDf.iloc[j])))
              outlier.append(((principalDf.iloc[j]['D1'])))
              outlierindex.append(((principalDf.iloc[j]['D2'])))
            else:
              #datapoint.append(np.sum(np.array(principalDf.iloc[j])))
              datapoint.append(((principalDf.iloc[j]['D1'])))
              dataindex.append(((principalDf.iloc[j]['D2'])))

        fig = plt.figure(figsize=(20,10)) 
        plt.scatter(datapoint,dataindex,color='blue')
        plt.scatter(outlier,outlierindex,color='red')
        plt.title("LOF Outlers on Full Data marked in Red")
        plt.show

        return odf

