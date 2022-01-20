from UserInfo import UserInfo
from libraries import *

class MakeModel(UserInfo):
    
    # def __init__(self):
    #     UserInfo.get_user_info(self)
    #     UserInfo.get_use_columns(self)
    def __init__(self):
        UserInfo.__init__(self)
    

    def model_info(self):
        model_dict={"displayModelName":"",
                   "modelName":"",
                   "accuracy":"",
                   "amountOfTrainData":"",
                   "amountOfTestData":"",
                   "detectedFraudData":"",
                   "actualFraudDataFromTest":"",
                   "modelPath":"",
                   "threshold":"",
                   "description":"",
                   "isActive":False}
        return model_dict
    
    def get_savepath(self, modelName):
        savePath = self.companyCode +'/'+self.machineNameValue+'/'+modelName+'/'
        if os.path.exists(savePath) is False:
            os.makedirs(savePath)
        return savePath
    
        
    def save_models(self, savePath,modelName=False, df=False, pipeline=False, model=False):
        if df is not False:
            df.to_pickle(savePath + 'df('+self.productNumberValue+').pkl')
        if pipeline is not False:
            joblib.dump(pipeline, savePath + 'pipeline('+self.productNumberValue+').pkl')
        if model is not False:
            model.save(savePath + modelName+'('+self.productNumberValue+').h5')
        return savePath
    
    
    
    def get_db_data_df(self):
        serverAddress = "server.interxlab.io:15115"
        clientUpdate = MongoClient("mongodb://interx:interx%40504@{0}/admin".format(serverAddress))
        db = clientUpdate[self.companyCode]
        col = db[self.collectionName]
        df = pd.DataFrame(list(col.find({self.machineName: self.machineNameValue,self.productNo:self.productNumberValue})))
        return df
    
    def get_db_data_df_error(self):
        serverAddress = "server.interxlab.io:15115"
        clientUpdate = MongoClient("mongodb://interx:interx%40504@{0}/admin".format(serverAddress))
        db = clientUpdate[self.companyCode]
        col = db[self.errorCountCollection]
        df = pd.DataFrame(list(col.find({self.machineName: self.machineNameValue,self.productNo:self.productNumberValue})))
        return df
    

    
    def make_merge_df(self,df_errorCount, df):
        order_clean = df_errorCount[df_errorCount[self.errorCountInErrorCountCollection] == 0 ].reset_index().drop('index',axis=1)
        order_fraud = df_errorCount[df_errorCount[self.errorCountInErrorCountCollection] != 0].reset_index().drop('index',axis=1)
        df_fraud = pd.merge(left = df , right = order_fraud, how = "inner", on = [self.timeStampInErrorCountCollection])
        df_clean = pd.merge(left = df , right = order_clean, how = "inner", on = [self.timeStampInErrorCountCollection])
        return df_clean, df_fraud
    
    def preprocessing(self, df):
        df = df.rename({self.prediction:'passorfail'},axis=1)
        columns_noncountable = [col for col in df.columns if df[col].nunique() == 1]

        if 'passorfail' in columns_noncountable:
            columns_noncountable.remove('passorfail')
        df = df.drop(columns_noncountable,axis=1)
        str_columns = []
        columns = df.columns
        for col in columns:
            try:
                df[col] = df[col].astype(float)
            except:
                pass
        str_col = df.dtypes[df.dtypes == 'object'].index
        df = df.drop(str_col, axis=1)
        df = df.dropna(axis=1)
        if df.passorfail.nunique() == 1:
            pass
        else:
            # Anova analysis
            anova_tables = []
            for column in df.columns:
                model = ols('passorfail ~'+column, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=1)
                anova_tables.append(anova_table)
            df_anova_tables = pd.concat(anova_tables)
            final_columns = df_anova_tables[(df_anova_tables['PR(>F)'].notnull()) & (
                df_anova_tables['PR(>F)'] < 0.05)].index
            final_columns = list(final_columns)
            if 'passorfail' not in final_columns:
                final_columns.append('passorfail')



            df = df[final_columns]
            lists = []
            x = df.corr()
            x = x.reset_index()
            col_list = x['index']
            x.drop(['index'], axis=1, inplace=True)
            columns = x.columns.drop('passorfail')


            #Correlation analysis
            for col in df.columns:
                indexes = x[col][(abs(x[col]) > 0.7) & (x[col] <= 1)].index
                if len(indexes) != 0:
                    max_index = x[x.passorfail == np.max(x.passorfail[indexes])].index
                    indexes = indexes.drop(max_index)
                for i in indexes:
                    lists.append(i)
            final_list = []
            for i in lists:
                if i not in final_list:
                    final_list.append(i)
            final_columns = x.drop(col_list[final_list], axis=1).columns
            df = df[final_columns].rename({'passorfail':self.prediction},axis=1)

        return df
    
    def create_stackingmodel(self, df):
    
        TEST_SAMPLE = round(len(df)/10)
        X_train, y_train = df.iloc[TEST_SAMPLE:].drop(self.prediction, axis=1),  df.iloc[TEST_SAMPLE:][self.prediction]
        X_test, y_test = df.iloc[:TEST_SAMPLE].drop(self.prediction, axis=1), df.iloc[:TEST_SAMPLE][self.prediction]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test =  scaler.transform(X_test)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, shuffle=True, stratify=y_train)
        
        savePath = self.get_savepath('machineLearning')
        self.save_models(savePath,df=df, pipeline = scaler)
        list_model_info = []
        
        model_name = {"RF" : 'RandomForestClassifier',
                # "XGB" : 'XGBClassifier',
                "LGBM" : 'LGBMClassifier', 
                "ERT" : 'ExtraTreesClassifier',
                "ADA" : 'AdaBoostClassifier',
                "DT" : 'DecisionTreeClassifier',
                "stacking" : "LogisticRegression"
            }
        
        CLF_MODELS = {
                "RF" : RandomForestClassifier(),
                # "XGB" : XGBClassifier(),
                "LGBM" : LGBMClassifier(), 
                "ERT" : ExtraTreesClassifier(),
                "ADA" : AdaBoostClassifier(learning_rate=0.01),
                "DT" : DecisionTreeClassifier(),
            }
        best_ML_accuracy = 0
        valid, X_valid_ensemble = [], []

        for name, model in CLF_MODELS.items():

                model.fit(X_train,y_train)
                real = y_valid
                pred = model.predict(X_valid)
                accuracy = accuracy_score(pred,y_valid)
                if accuracy > best_ML_accuracy:
                    best_ML_accuracy = accuracy
                    best_ML_model = model
                valid.append([pred,name])
                pred_test = model.predict(X_test)
                X_valid_ensemble.append([pred_test,name])
                joblib.dump(model,savePath+'/'+name+'('+self.productNumberValue+').pkl')
                model_info = self.model_info()
                model_info['modelName'] = name
                model_info['accuracy'] = accuracy
                model_info['amountOfTrainData'] = len(X_train)
                model_info['amountOfTestData'] = len(X_valid)
                model_info['modelPath'] = savePath
                model_info['displayModelName'] = model_name[name]
                list_model_info.append(model_info)

        df = pd.DataFrame()
        df['RF'] = valid[0][0]
        # df['XGB'] = valid[1][0]
        df['LGBM'] = valid[1][0]
        df['ERT'] = valid[2][0]
        df['ADA'] = valid[3][0]
        df['DT'] = valid[4][0]
        df_test = pd.DataFrame()
        df_test['RF'] = X_valid_ensemble[0][0]
        # df_test['XGB'] = X_valid_ensemble[1][0]
        df_test['LGBM'] = X_valid_ensemble[1][0]
        df_test['ERT'] = X_valid_ensemble[2][0]
        df_test['ADA'] = X_valid_ensemble[3][0]
        df_test['DT'] = X_valid_ensemble[4][0]
        lr = LogisticRegression()
        lr.fit(df,y_valid)
        stacking_model = lr.predict(df_test)
        stacking_accuracy = accuracy_score(stacking_model,y_test)
        joblib.dump(lr, savePath+'/stacking('+self.productNumberValue+').pkl')
        model_info = self.model_info()

        model_info['modelName'] = 'stacking'
        model_info['displayModelName'] = 'stacking'
        model_info['accuracy'] = stacking_accuracy
        model_info['amountOfTrainData'] = len(X_train)
        model_info['amountOfTestData'] = len(df_test)
        model_info['modelPath'] = savePath
        list_model_info.append(model_info)

        return list_model_info
    
    def create_autoencoder(self, X_train, X_valid,save_number=False):
        input_dim = X_train.shape[1]
        model = tf.keras.models.Sequential([
            # deconstruct / encode
            tf.keras.layers.Dense(input_dim, activation='elu',
                                  input_shape=(input_dim, )),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dense(8, activation='elu'),
            tf.keras.layers.Dense(4, activation='elu'),
            tf.keras.layers.Dense(2, activation='elu'),
            # reconstruction / decode
            tf.keras.layers.Dense(4, activation='elu'),
            tf.keras.layers.Dense(8, activation='elu'),
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(input_dim, activation='elu')
        ])
        
        BATCH_SIZE = 512
        EPOCHS = 1500
        # https://keras.io/layers/core/
        model.compile(optimizer="adam",
                            loss="mse",
                            metrics=["acc"])
        from datetime import datetime
        # current date and time
        yyyymmddHHMM = datetime.now().strftime('%Y%m%d%H%M')
        # new folder for a new run
        log_subdir = f'{yyyymmddHHMM}_batch{BATCH_SIZE}_layers{len(model.layers)}'
        # define our early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=200,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=6, verbose=1)
        # callbacks argument only takes a list
        
        savePath = self.get_savepath('autoEncoder')
        if save_number is not False:
            filename = os.path.join(savePath,f"autoEncoder{save_number}({self.productNumberValue}).h5")
        else:
            filename = os.path.join(savePath,f"autoEncoder({self.productNumberValue}).h5")
        sv = keras.callbacks.ModelCheckpoint(
                        filename, monitor='val_loss', verbose=1, save_best_only=True,
                        save_weights_only=False, mode='auto', save_freq='epoch')

    #     lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, verbose=1)
        cb = [early_stop,  lr,sv]
        history = model.fit(
            X_train, X_train,
            shuffle=True,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=cb,
            validation_data=(X_valid, X_valid)
        )
        return model, savePath
    
    def preprocessing_noneLabel(self,df):
        columns_noncountable = [col for col in df.columns if df[col].nunique() == 1]
        df = df.drop(columns_noncountable,axis=1)
        str_columns = []
        columns = df.columns
        for col in columns:
            try:
                df[col] = df[col].astype(float)
            except:
                pass
        str_col = df.dtypes[df.dtypes == 'object'].index
        df = df.drop(str_col, axis=1)
        df = df.dropna(axis=1)
        return df
        
    
    
    def create_autoencoder_onetoone(self, df):
        TEST_SAMPLE = round(len(df)/10)
        fraud_df = df[df[self.prediction] == 0]
        df = df[df[self.prediction] == 1]
        X_train = df.iloc[TEST_SAMPLE:].drop([self.prediction], axis=1)
        X_test = df.iloc[:TEST_SAMPLE].append(fraud_df).sample(frac=1)
        X_train, X_validate = train_test_split(X_train, test_size=0.1, random_state=156)
        X_test, y_test = X_test.drop(self.prediction, axis=1).values, X_test[self.prediction]
        pipeline = Pipeline([('normalizer', Normalizer()),
                             ('scaler', MinMaxScaler())])
        pipeline.fit(X_train)
        # transform the training and validation data with these parameters
        X_train_transformed = pipeline.transform(X_train)
        X_validate_transformed = pipeline.transform(X_validate)
        autoencoder, savePath = self.create_autoencoder(X_train_transformed,X_validate_transformed)
        self.save_models(savePath, df = df, pipeline = pipeline)
        
        # transform the test set with the pipeline fitted to the training set
        X_test_transformed = pipeline.transform(X_test)
        # pass the transformed test set through the autoencoder to get the reconstructed result

        reconstructions = autoencoder.predict(X_test_transformed)
        mse = np.mean(np.power(X_test_transformed - reconstructions, 2), axis=1)
        mse = mse * 1000000000
        y_test= pd.DataFrame(y_test)
        y_test['mse'] = mse
        if pd.DataFrame(y_test)[y_test[self.prediction] == 0].empty == False:
            for i in range(len(y_test[y_test[self.prediction] == 0])):
                THRESHOLD = y_test.mse[y_test[self.prediction] == 0].sort_values(
                ).reset_index().drop('index', axis=1).iloc[i][0]
                outliers = mse < THRESHOLD
                # get (mis)classification
                cm = confusion_matrix(y_test[self.prediction], outliers)
                if cm[1][0] > 10:
                    continue
                else:
                    break     
        else:
            THRESHOLD = math.ceil(max(y_test.mse[y_test[self.prediction] == 1]))
            outliers = mse <= THRESHOLD
          # get (mis)classification
            cm = confusion_matrix(y_test[self.prediction], outliers)
            y_test['outliers'] = outliers
            result_df = y_test.copy()
        
        model_info = self.model_info()
        model_info['displayModelName'] = 'autoEncoder'
        model_info['modelName'] = 'autoEncoder'
        model_info['accuracy'] = accuracy_score(y_test[self.prediction], outliers)
        model_info['threshold'] = THRESHOLD
        model_info['description'] = 'basic autoencoder'
        model_info['modelPath'] = savePath
        
        return model_info
    
    
    def create_autoencoder_errorCount(self, df,fraud_df):
        
        model_infos = []
        
        train_sample_list1 = [int(round(len(df)*0.1,0)), int(round(len(df)*0.3,0)), int(round(len(df)*0.2,0)),int(round(len(df)*0.05,0)) ]
        train_sample_list2 = [int(round(len(df)*0.9,0)), int(round(len(df)*0.8,0)), int(round(len(df)*0.2,0)),int(round(len(df)*0.85,0)) ]
        sample_lists = [train_sample_list1,train_sample_list2]
        save_number = 0
        for i, train_sample_list in enumerate(sample_lists):
            for TEST_SAMPLE in train_sample_list:
                save_number +=1
                if i == 0:
                    X_train = df.iloc[TEST_SAMPLE:]
                    X_test = df.iloc[:TEST_SAMPLE]
                else:
                    X_train = df.iloc[:TEST_SAMPLE]
                    X_test = df.iloc[TEST_SAMPLE:]
                X_train, X_validate = train_test_split(X_train, test_size =0.15,random_state = 121)
                pipeline = Pipeline([('normalizer', Normalizer()),
                                     ('scaler', MinMaxScaler())])
                pipeline.fit(X_train)
                # transform the training and validation data with these parameters
                X_train_transformed = pipeline.transform(X_train)
                X_validate_transformed = pipeline.transform(X_validate)

                autoencoder, savePath = self.create_autoencoder(X_train_transformed,X_validate_transformed,save_number)
                self.save_models(savePath, df = df, pipeline = pipeline)
                # transform the test set with the pipeline fitted to the training set
                X_test_transformed = pipeline.transform(X_test)
                # pass the transformed test set through the autoencoder to get the reconstructed result
                reconstructions = autoencoder.predict(X_test_transformed)
                mse = np.mean(np.power(X_test_transformed - reconstructions, 2), axis=1)
                mse = mse * 1000000000
                THRESHOLD = max(mse)
                outliers = mse > THRESHOLD
                fraud_df_list = fraud_df[df.columns]
                fraud_df_transformed = pipeline.transform(fraud_df_list)
                reconstructions = autoencoder.predict(fraud_df_transformed)
                mse = np.mean(np.power(fraud_df_transformed - reconstructions, 2), axis=1)
                mse = mse*1000000000

                error_count = fraud_df.groupby([self.errorCountInErrorCountCollection,self.timeStampInErrorCountCollection]).mean(self.errorCountInErrorCountCollection).reset_index()[self.errorCountInErrorCountCollection].sum()
                outliers = mse > THRESHOLD        
                detect_error = len([outlier for outlier in outliers if outlier == True])
                difference = abs(error_count - len([outlier for outlier in outliers if outlier == True]))
                amount_data = len(fraud_df)


                model_info = self.model_info()
                model_info['displayModelName'] = 'autoEncoder'
                model_info['modelName'] = f'autoEncoder{save_number}'
                model_info['accuracy'] = (amount_data-abs(error_count - detect_error))/(amount_data)
                model_info['threshold'] = THRESHOLD
                model_info['modelPath'] = savePath
                model_info['description'] = 'parameter tunning'
                model_info['amountOfTrainData'] = len(X_train)
                model_info['amountOfTestData'] = len(X_test)
                model_infos.append(model_info)


        return model_infos
    


        
    
    def kMeans_clustering(self, df, n=2):
        
        from sklearn.cluster import KMeans
        pipe = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters = n, random_state = 0))])
        df_clustered = pipe.fit_predict(df)
        df['cluster_label'] = df_clustered
        model_name = 'kMeans'
        savePath = self.get_savepath('clustering')

        self.save_models(savePath,modelName = model_name,df=df,pipeline=pipe)
        model_dict = self.model_info()
        model_dict['displayModelName'] = 'clustering_KMeans'
        model_dict["modelName"]="KMeans"
        model_dict["amountOfTrainData"]=len(df)
        model_dict['modelPath'] = savePath
        model_dict['detectedFraudData'] =df['cluster_label'].value_counts()[1]
        make_list = []
        make_list.append(model_dict)
        return make_list
    
    
    def make_finalModel(self):
        df = self.get_db_data_df()
        use_columns = self.get_use_columns()

        if self.predictionAvailable == 'no':
            df = df[use_columns]
            df = self.preprocessing_noneLabel(df)
            return self.kMeans_clustering(df)

        elif (self.predictionAvailable == 'yes')&(self.predictionOneToOne == 'yes'):
            df[self.prediction] = df[self.prediction].map(lambda x: 1 if str(x) == self.trueVal else (0 if str(x) == self.falseVal else -1))
            df = df[(df[self.prediction] == 1)|(df[self.prediction] == 0)]
            use_columns.append(self.prediction)
            df = df[use_columns]
            df_preprocessed = self.preprocessing(df)
            info_models= self.create_stackingmodel(df_preprocessed)
            info_models.append(self.create_autoencoder_onetoone(df_preprocessed))
            return info_models    

        elif (self.predictionAvailable == 'yes')&(self.predictionOneToOne == 'no'):
            df_errorCount = self.get_db_data_df_error()        
            df, df_fraud = self.make_merge_df(df_errorCount, df)
            use_columns = ['Average_Screw_RPM', 'Max_Injection_Pressure',
                'Max_Switch_Over_Pressure', 'Max_Back_Pressure',
                'Average_Back_Pressure', 'Barrel_Temperature_1', 'Barrel_Temperature_2',
                'Barrel_Temperature_3', 'Barrel_Temperature_4', 'Barrel_Temperature_5',
                'Barrel_Temperature_6', 'Barrel_Temperature_7', 'Hopper_Temperature']
            df, df_fruad = df[use_columns], df_fraud[use_columns]
            info_model = self.create_autoencoder_errorCount(df,df_fraud)
            return info_model