class Visualization:
    def __init__(self,data_frame = None, location_name_list=None):
        self.data_frame = data_frame # array of column names to encode
        self.location_name_list = location_name_list
        self.colors = ['b','r','g','p']
        self.distmapping = { 'W': 'Weekly', 'M': 'Monthly','Y':'Yearly'}
        self.cols_to_keep = ['mean_temp','rain','wind_speed','abnormal_rain','high_temp','low_temp','high_wind','EasterSundayHoliday',
                             'University_holidays','School_holidays','UKBankHoliday']
        
        self.modified_location_name_list=['InCount'+x for x in self.location_name_list]
        self.init_gg_plots()
        
    
    def init_gg_plots(self):
      theme_set(
            theme_538() +
            theme(
                figure_size = (8, 4),
                text = element_text(
                    size = 8,
                    color = 'black',
                    family = 'DejaVu Sans'
                ),
                plot_title = element_text(
                    color = 'black',
                    family = 'DejaVu Sans',
                    weight = 'bold',
                    size = 12
                ),
                axis_title = element_text(
                    color = 'black',
                    family = 'DejaVu Sans',
                    weight = 'bold',
                    size = 6
                ),
            )
        )
      
    def uni_variate_ggplot_histogram(self, x_list,x_limit_list, y_limit_list ):
      index = 0
      for col in x_list:
        fig = ggplot(self.data_frame) + geom_histogram(aes(x = col),fill = 'blue', color = 'red') + labs(title = col +' distribution - (median = dashed line; mean = solid line)', x = col, y = 'Frequency',
            ) + scale_x_continuous(
                limits = (0, x_limit_list[index]), 
                labels = self.labels(0, x_limit_list[index], int(x_limit_list[index]/10)),
                breaks = self.breaks(0, x_limit_list[index], int(x_limit_list[index]/10))
            )+  scale_y_continuous(
                limits = (0, y_limit_list[index]),
                labels = self.labels(0, y_limit_list[index], int(y_limit_list[index]/10)),
                breaks = self.breaks(0, y_limit_list[index], int(y_limit_list[index]/10))
            )+ geom_vline(aes(xintercept = self.data_frame[col].mean()), color = 'gray')+ geom_vline(aes(xintercept = self.data_frame[col].median()), linetype = 'dashed', color = 'gray')
        fig.draw()
        index += 1

    def facet_ggplot_histogram(self, x, facet_wrap_cols, x_limit, y_limit):
      fig= ggplot(self.data_frame) + geom_histogram(aes(x = x),fill = 'blue', color = 'red') + labs(title =x +' distribution', x = x, y = 'Frequency',
          ) + scale_x_continuous(
              limits = (0, x_limit), 
              labels = self.labels(0, x_limit, int(x_limit/10)),
              breaks = self.breaks(0, x_limit, int(x_limit/10))
          )+  scale_y_continuous(
              limits = (0, y_limit),
              labels = self.labels(0, y_limit, int(y_limit/10)),
              breaks = self.breaks(0, y_limit, int(y_limit/10))
          )+ theme(figure_size = (16, 24)) + facet_wrap(facet_wrap_cols, ncol = len(facet_wrap_cols), labeller='label_both')
      fig.draw()

    def scatter_plot_ggplot(self, x = 'InCountTotal',y='rain', hue='Weekday', color = 'red', x_limit=200000, y_limit=15):
      fig = ggplot(self.data_frame) + geom_point(aes(x = x,y=y, fill=hue),alpha = 0.5, color = color) + labs(title ='Scatter plot of '+ x + ' and '+y, x = x, y = y,
        ) +scale_x_continuous(
              limits = (0, x_limit), 
              labels = self.labels(0, x_limit, int(x_limit/10)),
              breaks = self.breaks(0, x_limit, int(x_limit/10))
          )+  scale_y_continuous(
              limits = (0, y_limit),
              labels = self.labels(0, y_limit, int(y_limit/10)),
              breaks = self.breaks(0, y_limit, int(y_limit/10))
        )+ theme(figure_size = (12, 8)) 
      fig.draw()

    def labels(self, from_, to_, step_):
      return pd.Series(np.arange(from_, to_ + step_, step_)).apply(lambda x: '{:,}'.format(x)).tolist()
    def breaks(self, from_, to_, step_):
      return pd.Series(np.arange(from_, to_ + step_, step_)).tolist()

    def plot_graphs(self, plot_type = 'boxplot', col_name = 'InCount', nplotrows=4,nplotcols=2):
        
        location_area_index = 0
        for i in range(nplotrows):# 4 rows of plots 
            fig = plt.figure(figsize=(16,8)) 
            for j in range(nplotcols):# 2 colums of sub plots
              ax1 = fig.add_subplot(1, 2, j+1)
              if(col_name == 'InCount'):
                target_variable = self.modified_location_name_list[location_area_index]
              else:
                target_variable = col_name                
            
              if(plot_type == 'boxplot'): 
                ax1.boxplot(self.data_frame[target_variable])
              elif(plot_type == 'histogram'):
                n, bins, patches = ax1.hist(self.data_frame[target_variable], bins=200, color=self.colors[j], density=True)
              elif(plot_type == 'heatmap'):
                cols = self.cols_to_keep.copy()
                cols.append(target_variable)
                sns.heatmap(self.data_frame[cols].corr(), annot=True, linewidths=.20)
                                        
              ax1.set_xlabel('Mall foot fall - ' + col_name + " for " + self.location_name_list[location_area_index] )
              ax1.set_ylabel('Frequency')
              ax1.set_title('Mall foot fall '+ plot_type +' - ' + col_name + " for " + self.location_name_list[location_area_index])
              location_area_index +=1
            
    def show_timedistribution(self, distribution_type='W', col_name='InCount'):
      for location_name in self.location_name_list:
        if(col_name == 'InCount'):
          target_variable = col_name + location_name
        else:
          target_variable = col_name
        self.data_frame[target_variable].resample(distribution_type).mean().plot(figsize = (20,8))
        
        xlabel = self.distmapping[distribution_type]
        plt.xlabel(xlabel)
        plt.ylabel('Mall Foot fall')
        plt.legend(self.location_name_list)
        plt.title("Mall Foot fall for different locations - " + xlabel)
    def show_pairplot(self,one_location_only = True, kind='scatter'):
      sns.set_style('whitegrid')
      cols = self.cols_to_keep
      cols.append('InCount'+self.location_name_list[0])
      sns.pairplot(self.data_frame[cols], kind=kind)

    def comparegates(self):
        gatecorrdata=data_final[self.modified_location_name_list]
        sns.pairplot(gatecorrdata)
        plt.show()
        sns.heatmap(gatecorrdata.corr(), annot = True)
        plt.show()
    
    def bi_variate_kernel_density_plot(self, x, y_list, hue_list=None):
     
      nplotrows = len(hue_list)
      nplotcols = len(y_list)
      for i in range(nplotrows):# 4 rows of plots 
            fig = plt.figure(figsize=(16,8)) 
            for j in range(nplotcols):# 2 colums of sub plots
              ax1 = fig.add_subplot(1, 3, j+1)
              
              sns.kdeplot(data = self.data_frame, x=x, y=y_list[j], hue=hue_list[i])
                                     
              ax1.set_xlabel('KDE plot ' + x  )
              ax1.set_ylabel(y_list[j])
              ax1.set_title('KDE plot '+ x +' and ' + y_list[j] )              

    
    def plot_graphs_for_list(self, plot_type = 'boxplot', col_list = ['rain'], target_col='InCountTotal'):
        
        col_index = 0
        nplotrows =1 
        nplotcols = 1
        if(len(col_list)>1):
          nplotrows = int(len(col_list)/2)
          nplotcols = 2
        for i in range(nplotrows):# 4 rows of plots 
            fig = plt.figure(figsize=(16,8)) 
            for j in range(nplotcols):# 2 colums of sub plots
              ax1 = fig.add_subplot(1, 2, j+1)

              col_name = col_list[col_index]           
              if(plot_type == 'boxplot'): 
                ax1.boxplot(self.data_frame[col_name])
              elif(plot_type == 'histogram'):
                n, bins, patches = ax1.hist(self.data_frame[col_name], bins=10, color=self.colors[j],density=True)
              elif(plot_type == 'kdeplot'):
                  #sns.kdeplot(self.data_frame, color=self.colors[j], shade=True, Label=col_name, hue='Weekday')
                  sns.kdeplot(data = self.data_frame, x= target_col, shade=True, hue=col_name)                             
              ax1.set_xlabel(col_name + " for " + self.location_name_list[0] )
              ax1.set_ylabel('Frequency')
              ax1.set_title(plot_type +' - ' + col_name )
              col_index +=1

    def show_categorical_plots(self, style="swarm", x='Weekday',y='InCountTotal', hue='Month', order=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]):
      if style == "swarm":
          sns.swarmplot(x=x, y=y,data=self.data_frame,order=order)
      elif style == "swarm_with_hue":
          sns.swarmplot(x=x, y=y,hue=hue, data=self.data_frame,order=order)
      elif style == "stripplot":
          sns.stripplot(x=x, y=y,data=self.data_frame,order=order)
      elif style == "stripplot_with_hue":
          sns.stripplot(x=x, y=y,hue=hue, data=self.data_frame,order=order)
      elif style == "box":
          sns.boxplot(x=x, y=y, data=self.data_frame, whis=np.inf,order=order)
      elif style == "box_with_hue":
          sns.boxplot(x=x, y=y, hue=hue, data=self.data_frame, whis=np.inf,order=order)
      elif style == "violen":
          sns.violinplot(x=x, y=y, data=self.data_frame,order=order)
      elif style == "boxen":
          sns.boxenplot(x=x, y=y, data=self.data_frame,order=order)
      else:
            print("no style is selected")  

    def ggplot_swarm(self, dataframe, x_column, y_column, hue=None):
      ax = sns.swarmplot(x=x_column, y=y_column, hue=hue, data=dataframe, size=6)
      ax.set_title(x_column + " and " + y_column + " analysis.")

    def ggplot_count(self, dataframe, x_column, y_column, hue=None):
      ax = sns.countplot(x=x_column, hue=hue, data=dataframe)
      ax.set_title(x_column + " and " + y_column + " analysis.")   

    def ggplot_cat(self, dataframe, x_column, y_column, col, hue=None):
      g = sns.catplot(x=x_column, y=y_column,
                hue=hue, col=col,
                data=dataframe, kind="swarm",
                height=6, aspect=.7); 
    
    def relplot_cat(self, dataframe, x_column, y_column, col, row, size, hue=None):
        g = sns.relplot(x=x_column,y=y_column,
                        hue=hue,col=col,
                        row=row,size=size,
                        data=dataframe,palette=["b", "r"]);
    
    
    def plotcategoricaldist(self,catcollist,featlist):
        dd=pd.DataFrame(self.data_frame)
        numb=len(dd)
        fig, axs = plt.subplots(len(catcollist),figsize=(15,40))
        fig.tight_layout(pad=2.0)
        nc=0

        for col in(catcollist):
            uniq=dd[col].unique()
            pcnt=[]
            for val in uniq:
                pcnt.append(100*(dd[col][dd[col]==val].count())/numb)

            hts=axs[nc].bar(uniq,pcnt)    

            for h in (hts):
                ht=h.get_height()
                axs[nc].text(h.get_x(), 1.05*ht, str(round(ht,2)))
            axs[nc].set_title(col+'Percentage Distribution in the Data')
    
            nc=nc+1
        plt.show
        
        
        fig, axs = plt.subplots(len(catcollist),figsize=(15,40))
        fig.tight_layout(pad=2.0)
        nc=0
        
        for catcol in catcollist:
            uniq=dd[catcol].unique()
            rn = np.arange(len(uniq))
            bwidth=.05
        
            for feature in featlist:
                temp=[]
                pcnt=[]
                for colval in uniq:
                    temp.append(np.sum(dd[feature][dd[catcol]==colval]))
                    pcnt.append(100*(dd[catcol][dd[catcol]==colval].count())/numb)
                axs[nc].bar(rn,temp,width=bwidth,label=feature)
        
                rn = [x + bwidth for x in rn]
            axs[nc].set_title(catcol+'Distribution in the Data')
            #plt.legend()
            plt.xticks([r + (bwidth*len(uniq)) for r in range(len(uniq))], uniq)
            nc=nc+1
        
        
        plt.show()
