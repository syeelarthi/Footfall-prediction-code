class Lagging:  
  def __init__(self, location_name_list=None, n_timesteps=7, n_forecast_steps=1):
      self.n_timesteps = n_timesteps
      self.location_name_list = location_name_list # array of locations
      self.n_forecast_steps = n_forecast_steps

  def createforecastTargetVariables(self, df):
    # copy the InCount to other forecast columns
    for forecast_step in range(1, self.n_forecast_steps+1): 
      df['InCountTotal'+'_forecast'+str(forecast_step)] = df['InCountTotal']
          
    #update multistep output variables
    for i in range(0, df.shape[0]-self.n_forecast_steps):
      for forecast_step in range(1, self.n_forecast_steps+1): 
        df['InCountTotal'+'_forecast'+str(forecast_step)][i] = df['InCountTotal'][i+forecast_step]
      
    #remove the last nforecast rows as these do not have proper values after creating the forecast variables
    df.drop(df.tail(self.n_forecast_steps).index,inplace=True)
    return df

  def createLagVariables(self,df): #Create lag variables of size=n_timesteps
      if location_wise_data == True:#location wise data
        for location in self.location_name_list:
            location = 'InCount'+location
            for lag in range(1, self.n_timesteps+1):
                df[location+'_lag'+str(lag)] = df[location].shift(periods=lag)
      else:# totalincount only
        for lag in range(1, self.n_timesteps):
            df['InCountTotal'+'_lag'+str(lag)] = df['InCountTotal'].shift(periods=lag)
        
      df.dropna(inplace = True)
      return df

  def createLagDiffVariables(self,df,location_wise_data = True): #Create lag variables of size=n_timesteps
    if location_wise_data == True:#location wise data
      for location in self.location_name_list:
        location = 'InCount'+location
        for lag in range(1, self.n_timesteps+1):
          df[location+'_lag'+str(lag)] = df[location].diff(periods=lag)
    else:# totalincount only
      for lag in range(1, self.n_timesteps+1):
          df['InCountTotal'+'_lag'+str(lag)] = df['InCountTotal'].diff(periods=lag)
    
    df.dropna(inplace = True)
    return df
