
from libraries import *

class UserInfo:

    def __init__(self):
        companyCode="ixSolutionsDB"
        collectionName="aiVoucherSettings"
        df = self.get_db_data(companyCode,collectionName)
        user_info = df.iloc[0]
        self.user_info = user_info
        self.companyCode = user_info.companyCode
        self.timeStamp = user_info.timeStamp
        self.machineCode = user_info.machineCode
        self.machineName = user_info.machineName
        self.productNo = user_info.productNo
        self.productName = user_info.productName
        self.prediction = user_info.prediction
        self.predictionAvailable = user_info.predictionAvailable
        self.predictionOneToOne = user_info.predictionOneToOne
        self.errorCountCollection = user_info.errorCountCollection
        self.timeStampInErrorCountCollection = user_info.timeStampInErrorCountCollection
        self.productNoInErrorCountCollection = user_info.productNoInErrorCountCollection
        self.errorCountInErrorCountCollection = user_info.errorCountInErrorCountCollection
        self.trueVal = user_info.trueVal
        self.falseVal = user_info.falseVal
        self.collectionName = user_info.collectionName
        self.productNumberValue = user_info.productNumberValue
        self.machineNameValue = user_info.machineNameValue



    def get_db_data(self,companyCode,collectionName):
        serverAddress = "server.interxlab.io:15115"
        clientUpdate = MongoClient("mongodb://interx:interx%40504@{0}/admin".format(serverAddress))
        db = clientUpdate[companyCode]
        col = db[collectionName]
        df = pd.DataFrame(list(col.find({})))
        return df


    
    
    def get_use_columns(self):
        column_lists = ['field1','field2','field3','field4','field5','field6','field7','field8','field9','field9','field10','field11',
                  'field12','field13','field14','field15']
        use_columns = [self.user_info[col] for col in column_lists if self.user_info[col] != '']
        if self.user_info.others != '':
            for i in self.user_info.others.split(','):
                use_columns.append(i)
        use_columns = list(dict.fromkeys(use_columns))
        return use_columns