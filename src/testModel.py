from libraries import *
from createModel import MakeModel
from UserInfo import UserInfo


class TestModel(MakeModel):
    def __init__(self, model_info):
        for  i in range(len(model_info)):
            if model_info[i]['isActive'] == True:
                MakeModel.__init__(self)
                self.modelName = model_info[i]['modelName']
                self.modelPath = model_info[i]['modelPath']
                self.threshold = model_info[i]['threshold']                 
                    
    def callModels(self,df=False,pipeline=False,autoEncoder=False, machineLearning=False):       
        model_package = []
        if df is not False:
            df = pd.read_pickle(self.modelPath+'df('+self.productNumberValue+').pkl')
            model_package.append(df)
        if pipeline is not False:
            pipeline = joblib.load(self.modelPath + 'pipeline('+self.productNumberValue+').pkl')
            model_package.append(pipeline)
        if autoEncoder is not False:
            model = load_model(self.modelPath + self.modelName+'('+self.productNumberValue+').h5')
            model_package.append(model)
        if machineLearning is not False:
            model = joblib.load(self.modelPath+self.modelName+'('+self.productNumberValue+').pkl')
            model_package.append(model)

        return model_package

    def preprocessing_new_data(self,new_data, df, pipeline=False,model=False):
        if self.prediction in df.columns:
            df.drop(self.prediction, axis=1,inplace=True)
        if 'cluster_label' in df.columns:
            df.drop('cluster_label',axis=1,inplace=True)
        new_data = np.array(new_data[df.columns]).reshape(1,-1)
        if pipeline is not False:
            new_data = pipeline.transform(new_data)
        if model is not False:
            new_data = model.predict(new_data)
        return new_data        
        
        
    def testData(self,new_data):
        modelName = self.modelName[:11]
        if modelName == 'autoEncoder':
            model_package = self.callModels(df=True,pipeline=True,autoEncoder=True)
            df, pipeline, autoEncoder = model_package[0], model_package[1], model_package[2]
            new_data = self.preprocessing_new_data(new_data,df,pipeline=pipeline)
            reconstructions = autoEncoder.predict(new_data)
            mse = np.mean(np.power(new_data - reconstructions, 2), axis=1)
            mse = mse * 1000000000
            outlier = mse < self.threshold
            print(mse)
            if outlier[0]: result= 1
            else: result= 0
            return result
            
#             try:
#                 PART_NO_df, PART_NO_model, PART_NO_pipeline = self.callModels(companyCode,modelName)
#             except Exception as e:
#                 PART_NO_model = 0 
#                 print('No model for this PART_NO')
#             if PART_NO_model == 0:
#                 result = 0 
#             return model_package
            
        elif modelName =='KMeans':
            model_package = self.callModels(df=True,pipeline=True)
            df, pipeline = model_package[0], model_package[1]
            result = self.preprocessing_new_data(new_data,df,pipeline=pipeline)
            return np.argmax(result)

        elif modelName == 'stacking':
            model_package = self.callModels(df=True,pipeline=True,machineLearning=True)
            df, pipeline, stacking = model_package[0], model_package[1], model_package[2]
            new_data_scaled = self.preprocessing_new_data(new_data,df,pipeline=pipeline)
            
            RF = joblib.load(self.modelPath+'RF('+self.productNumberValue+').pkl')
            LGBM = joblib.load(self.modelPath+'LGBM('+self.productNumberValue+').pkl')
            DT = joblib.load(self.modelPath+'DT('+self.productNumberValue+').pkl')
            ERT = joblib.load(self.modelPath+'ERT('+self.productNumberValue+').pkl')
            ADA = joblib.load(self.modelPath+'ADA('+self.productNumberValue+').pkl')
            
            model_list = [RF,LGBM,DT,ERT,ADA]
            preds = []
            
            for model in model_list:
                preds.append(model.predict(new_data_scaled))
        
            result = stacking.predict(np.array(preds).reshape(1,-1))
            return result

        else:#ML
            model_package = self.callModels(df=True,pipeline=True,machineLearning=True)
            df,pipeline, model = model_package[0],model_package[1],model_package[2]
            result = self.preprocessing_new_data(new_data,df,pipeline=pipeline,model=model)
            return result