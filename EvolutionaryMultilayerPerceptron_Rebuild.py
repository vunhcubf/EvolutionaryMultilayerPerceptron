from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
import math
import sys
import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Utils:
    @staticmethod
    def RemoveZeros(x):
        if not isinstance(x,np.ndarray):
            x=np.array(x)
        x=x[x!=0]
        return x
# 用于处理结果的部分
def lerp(x1,x2,t):
    return x1*(1-t)+x2*t
class EmlpResultHandler:
    def DrawPercentageErrorHistogram(self,XTrain,YTrain,key_of_model,Name=''):
        YPred=self.PredictModel(key_of_model,XTrain)
        RmsePercentage=100*np.abs((YPred-YTrain)/(YTrain+1e-3))
        plt.hist(RmsePercentage, bins=100,color='red')
        plt.xlabel('Percentage Absolute Error(%)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution Histogram'+Name)
        plt.xlim(0,max(40,np.max(RmsePercentage)))
        plt.show()
    def DrawErrorHistogram(self,XTrain,YTrain,key_of_model,Name=''):
        YPred=self.PredictModel(key_of_model,XTrain)
        RmsePercentage=np.abs(YPred-YTrain)
        plt.hist(RmsePercentage, bins=100,color='red')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution Histogram'+Name)
        plt.xlim(np.min(RmsePercentage),np.max(RmsePercentage))
        plt.show()
    def __init__(self,ResultDictionary) -> None:
        self.ResultDictionary=ResultDictionary
    def PrintModel(self):
        info=[]
        index=[]
        for key,value in self.ResultDictionary.items():
            index.append(sum(value['TopologyInInt']))
            info_sub=f"模型网络结构为:{value['TopologyInInt']},\tEmlp误差为{value['EmlpError']}\t总神经元数:{sum(value['TopologyInInt'])}"
            if 'TotalPercentageRmseError' in value:
                info_sub+=f"\t总数据集百分比Rmse误差:{value['TotalPercentageRmseError']}"
            if 'TotalRmseError' in value:
                info_sub+=f"\t总数据集Rmse误差:{value['TotalRmseError']}"
            info.append(info_sub)
        sorted_list = sorted(enumerate(index), key=lambda x: x[1])
        for index,value in sorted_list:
            print(info[index])
    def PredictModel(self,key_of_model,x):
        ModelData=self.ResultDictionary[tuple(key_of_model)]
        # 检查是否应用了PCA
        if ('ComponentMatrix' in ModelData) and ('StandardScaler' in ModelData):
            # 应用变换
            x=ModelData['StandardScaler'].transform(x)
            x=np.dot(x,ModelData['ComponentMatrix'])
        key_of_model=Utils.RemoveZeros(key_of_model)
        if 'ActivateFunction' in ModelData:
            activate_function=ModelData['ActivateFunction']
        else:
            activate_function=[torch.sigmoid for _ in range(key_of_model.size)]
        x=(x-ModelData['XMin'])/(ModelData['XMax']-ModelData['XMin'])
        x=x.astype(np.float32)
        x=torch.from_numpy(x)
        model=EmlpNet(ModelData['Dimension'],key_of_model,1,activate_function)
        model.load_state_dict(ModelData['ModelDictionary'])
        y=model(x).data.numpy()
        y=ModelData['YMin']+y*(ModelData['YMax']-ModelData['YMin'])
        return y
    def DrawParetoSet(self,Name=''):
        complexity=[]
        accuracy=[]
        for key,value in self.ResultDictionary.items():
            complexity.append(sum(value['TopologyInInt']))
            accuracy.append(value['EmlpError'])
        plt.scatter(complexity,accuracy,s=20,edgecolors='red',facecolor='none')
        plt.xlabel('Complexity')
        plt.ylabel('EmlpError')
        plt.title('Pareto Set'+Name)
        plt.show()
##################################################################
##################################################################
class EmlpNet(nn.Module):
    def __init__(self,NVarsInput,ModelTopology,NVarsOutput,ActivateFunctionList) :
            super().__init__()
            self.ModelTopology=ModelTopology
            self.ActivateFunctionList=ActivateFunctionList
            self.layers=nn.ModuleList()
            for i in range(ModelTopology.size):
                if i==0:
                    self.layers.append(nn.Linear(NVarsInput,ModelTopology[i]))
                else:
                    self.layers.append(nn.Linear(ModelTopology[i-1],ModelTopology[i]))
            self.layers.append(nn.Linear(ModelTopology[ModelTopology.size-1],NVarsOutput))
    def forward(self,x):
        for i in range(len(self.layers)):
            x=self.layers[i](x)
            if i!=len(self.layers)-1:
                x=self.ActivateFunctionList[i](x)
        return x
class ValidationCheck():
    def __init__(self,max_count) -> None:
        self.counter=-1
        self.error=1e100
        self.max_count=max_count
    def update(self,y_pred,y):
        this_error=torch.nn.functional.mse_loss(y_pred,y)
        if self.error<=this_error:
            self.counter+=1
        else:
            self.counter=0
        self.error=this_error
        return self.counter>=self.max_count
##################################################################
##################################################################
class EmlpTrainer:
    # 构建非支配解集的函数
    @staticmethod
    def ResolveActivateFunction(ActivateFunctionName):
        if ActivateFunctionName == 'Sigmoid':
            return torch.sigmoid
        elif ActivateFunctionName == 'Tanh':
            return torch.tanh
        elif ActivateFunctionName == 'Relu':
            return torch.relu
        else:
            raise RuntimeError("无效的激活函数")
    def __init__(self,ConfigDict) -> None:
        # 保存训练的网络
        self.ParetoSetNeuralNetwork={}
        self.ConfigDict=copy.deepcopy(ConfigDict)
        # 新增PCA相关内容
        if 'UsePCA' in self.ConfigDict:
            if self.ConfigDict['UsePCA']:
                Scaler=StandardScaler()
                Scaler.fit(self.ConfigDict['X'])
                X_Std=Scaler.transform(self.ConfigDict['X'])

                model0=PCA()
                model0.fit(X_Std)
                if not('ExplainedVarianceRatio' in self.ConfigDict):
                    ExplainedVarianceRatio=0.95
                else:
                    ExplainedVarianceRatio=self.ConfigDict['ExplainedVarianceRatio']
                Dim=np.argmax(np.cumsum(model0.explained_variance_ratio_)>ExplainedVarianceRatio)

                plt.figure()
                plt.plot(np.arange(model0.explained_variance_ratio_.size),model0.explained_variance_ratio_)
                plt.title('Variance Contribution')
                plt.show()
                print(f'保留维度:{Dim+1}')

                Model=PCA(n_components=Dim+1,whiten=False)
                Model.fit(X_Std)
                # 保存相关数据
                self.ConfigDict['ComponentMatrix']=Model.components_.T
                self.ConfigDict['StandardScaler']=Scaler
                # 应用变换
                self.ConfigDict['X']=self.ConfigDict['StandardScaler'].transform(self.ConfigDict['X'])
                self.ConfigDict['X']=np.dot(self.ConfigDict['X'],self.ConfigDict['ComponentMatrix'])
        # 配置求解器相关的参数
        self.ConfigDict['NumberOfVars']=len(self.ConfigDict["LowerBound"])
        self.ConfigDict['LowerBound']=np.array(self.ConfigDict['LowerBound'],dtype=np.int32)
        self.ConfigDict['UpperBound']=np.array(self.ConfigDict['UpperBound'],dtype=np.int32)
        self.ConfigDict['Dimension']=self.ConfigDict['X'].shape[1]
        self.ConfigDict['NumberOfSamples']=self.ConfigDict['X'].shape[0]
        if not('PenaltyCoefficient' in self.ConfigDict):
            self.ConfigDict['PenaltyCoefficient']=[0.33,1]
        # 数据转化为单精度数据
        self.ConfigDict['X']=self.ConfigDict['X'].astype(np.float32)
        self.ConfigDict['Y']=self.ConfigDict['Y'].astype(np.float32)
        # 获取数据的最大值和最小值并归一化
        self.ConfigDict['XMin']=np.amin(self.ConfigDict['X'],axis=0)
        self.ConfigDict['XMax']=np.amax(self.ConfigDict['X'],axis=0)
        self.ConfigDict['YMin']=np.min(self.ConfigDict['Y'])
        self.ConfigDict['YMax']=np.max(self.ConfigDict['Y'])
        # 对数据使用0-1归一化
        self.ConfigDict['X']=(self.ConfigDict['X']-self.ConfigDict['XMin'])/(self.ConfigDict['XMax']-self.ConfigDict['XMin'])
        self.ConfigDict['Y']=(self.ConfigDict['Y']-self.ConfigDict['YMin'])/(self.ConfigDict['YMax']-self.ConfigDict['YMin'])
        # 处理一下输入的激活函数，扩展成函数指针的列表
        ActivateFunctionListTemporary=[]
        if len(self.ConfigDict['ActivateFunction'])==1 or not isinstance(self.ConfigDict['ActivateFunction'],list):
                for i in range(self.ConfigDict['NumberOfVars']):
                    func_ptr=EmlpTrainer.ResolveActivateFunction(self.ConfigDict['ActivateFunction'][0] if isinstance(self.ConfigDict['ActivateFunction'],list) else self.ConfigDict['ActivateFunction'])
                    ActivateFunctionListTemporary.append(func_ptr)
        else:
            for i in range(self.ConfigDict['NumberOfVars']):
                func_ptr=EmlpTrainer.ResolveActivateFunction(self.ConfigDict['ActivateFunction'][i])
                ActivateFunctionListTemporary.append(func_ptr)
        self.ConfigDict['ActivateFunction']=ActivateFunctionListTemporary
        # 将训练数据上传到Gpu
        try:
            self.ConfigDict['XGpu']=torch.from_numpy(self.ConfigDict['X']).float().to('cuda')
            self.ConfigDict['YGpu']=torch.from_numpy(self.ConfigDict['Y']).float().to('cuda')
        except:
            raise "上传训练数据到Gpu失败"
        # 创建待求解的问题
        self.Problem=EmlpProblem(self.ConfigDict,self.ParetoSetNeuralNetwork)
        self.Algorithm=NSGA2(pop_size=self.ConfigDict['PopSize'])
    def Train(self):
        minimize(self.Problem,self.Algorithm,('n_gen', self.ConfigDict['MaxIteration']),verbose=False)
    def GetTrainResult(self):
        return self.ParetoSetNeuralNetwork
    def DrawParetoSet(self):
        complexity=[]
        accuracy=[]
        for key,value in self.ParetoSetNeuralNetwork.items():
            complexity.append(sum(value['TopologyInInt']))
            accuracy.append(value['EmlpError'])
        plt.figure()
        plt.scatter(complexity,accuracy,s=20,edgecolors='red',facecolor='none')
        plt.xlabel('Complexity')
        plt.ylabel('EmlpError')
        plt.title('Pareto Set')
        plt.show()
##################################################################
##################################################################
    # Nsga2求解器的问题类
class EmlpProblem(Problem):
    @staticmethod
    def is_a_dominate_b(a_cost, b_cost):
        if np.all(a_cost <= b_cost) and np.any(a_cost < b_cost):
            result = 1
        elif np.all(b_cost <= a_cost) and np.any(b_cost < a_cost):
            result = -1
        else:
            result = 0
        return result
    def __init__(self,ConfigDict,ParetoSetNeuralNetwork,**kwargs):
        self.ConfigDict=ConfigDict
        self.ParetoSetNeuralNetwork=ParetoSetNeuralNetwork
        # 似乎vtype=int没有什么用
        super().__init__(n_var=ConfigDict['NumberOfVars'], n_obj=2, xl=ConfigDict['LowerBound'], xu=ConfigDict['UpperBound'], vtype=int,elementwise=True, **kwargs)
        # 当前的评估次数，用于显示进度
        self.CurrentEvaluateTimes=0
    def TrainNeuralNetwork(self,ModelData,XTrainGpu,XValidGpu,YTrainGpu,YValidGpu,UseValidationCheck):
        ModelTopology=Utils.RemoveZeros(ModelData['TopologyInInt'])
        # 创建需要训练的模型
        Model=EmlpNet(self.ConfigDict['Dimension'],ModelTopology,1,self.ConfigDict['ActivateFunction']).to('cuda')
        # 创建评价标准
        Criterion=nn.MSELoss()
        # 创建Adam优化器
        Optimizer=torch.optim.RAdam(Model.parameters(),lr=self.ConfigDict['MaxLearningRate'])
        # 创建可变学习率的Scheduler
        # LearningRateSchedulerStepSize=round(self.ConfigDict['Epochs']*0.01)
        # Gamma=(self.ConfigDict['MinLearningRate']/self.ConfigDict['MaxLearningRate'])**(1/100)
        # Scheduler=torch.optim.lr_scheduler.StepLR(Optimizer,step_size=LearningRateSchedulerStepSize,gamma=Gamma)
        # 创建验证检查器
        if UseValidationCheck:
            ValidationChecker=ValidationCheck(self.ConfigDict['MaxValidationCheck'])
        # 控制台输出的变量
        LastMessage=''
        CurrentMessage=''
        # 训练的循环
        for epochs in range(self.ConfigDict['Epochs']):
            YTrainPred=Model(XTrainGpu)
            Loss=Criterion(YTrainPred,YTrainGpu)
            Loss.backward()
            # 打印当前轮次的信息
            if (epochs+1)%100==0:
                sys.stdout.write('\r')
                CurrentMessage=f"训练轮次:{epochs+1}     训练误差:{Loss.item():.7e}     验证检查失败次数:{ValidationChecker.counter if UseValidationCheck else -1}     神经网络结构:{ModelTopology}     学习率:{Optimizer.param_groups[0]['lr']:.5e}"
                sys.stdout.write(CurrentMessage.ljust(max(len(LastMessage),1)))
                sys.stdout.flush()
                LastMessage=CurrentMessage
            # 求解器进行迭代
            Optimizer.step()
            Optimizer.zero_grad()
            # 可变学习率规划器迭代
            for ParamGroup in Optimizer.param_groups:
                ParamGroup['lr']=lerp(self.ConfigDict['MaxLearningRate'],self.ConfigDict['MinLearningRate'],epochs/self.ConfigDict['Epochs'])
            # 设置判断是否结束训练的标志
            QuitFlag=False
            # 进行验证检查
            if (UseValidationCheck) and ((epochs+1) % self.ConfigDict['ValidationCheckInterval']==0):
                YValidPred=Model(XValidGpu)
                QuitFlag=ValidationChecker.update(YValidGpu,YValidPred)
            if Loss.item()<=self.ConfigDict['Precision']:
                QuitFlag=True
            if QuitFlag:
                break
        if UseValidationCheck:
            YValidPred=Model(XValidGpu)
            return Model.cpu(),YTrainPred.cpu().data.numpy(),YValidPred.cpu().data.numpy()
        else:
            return Model.cpu(),YTrainPred.cpu().data.numpy(),None
    def _evaluate(self, x, out, *args, **kwargs):
        # 使用一个字典来保存这个网络的数据
        ModelData={}
        # 处理输入的浮点x
        ModelTopology_Float=x.tolist()
        ModelTopology_Int=np.round(x).astype(np.int32).tolist()
        ModelData['TopologyInFloat']=ModelTopology_Float
        ModelData['TopologyInInt']=ModelTopology_Int
        ModelData['ActivateFunction']=self.ConfigDict['ActivateFunction']
        UseValidationCheck=self.ConfigDict['TrainDataPercentage'] < 1
        if UseValidationCheck:
            # 划分这次训练的验证集和训练集
            RandomIndex=np.arange(0,self.ConfigDict['NumberOfSamples'])
            np.random.shuffle(RandomIndex)
            TrainDataIndex=RandomIndex[0 : round(self.ConfigDict['TrainDataPercentage']*self.ConfigDict['NumberOfSamples'])]
            ValidDataIndex=RandomIndex[round(self.ConfigDict['TrainDataPercentage']*self.ConfigDict['NumberOfSamples']) :]
            XTrainGpu=self.ConfigDict['XGpu'][TrainDataIndex]
            YTrainGpu=self.ConfigDict['YGpu'][TrainDataIndex]
            XValidGpu=self.ConfigDict['XGpu'][ValidDataIndex]
            YValidGpu=self.ConfigDict['YGpu'][ValidDataIndex]
        else:
            XTrainGpu=self.ConfigDict['XGpu']
            YTrainGpu=self.ConfigDict['YGpu']
            XValidGpu=None
            YValidGpu=None
        # 开始训练
        Model,YTrainPred,YValidPred=self.TrainNeuralNetwork(ModelData,XTrainGpu,XValidGpu,YTrainGpu,YValidGpu,UseValidationCheck)
        # 根据论文计算Emlp误差
        if not UseValidationCheck:
            YTrain=self.ConfigDict['Y']
            ErrorTrain=np.abs((YTrainPred-YTrain)/(YTrain+self.ConfigDict['PercentageErrorBias']))
            NValidAvg=np.sum((ErrorTrain>0.15)&(ErrorTrain<0.25))
            NValidBad=np.sum(ErrorTrain>0.25)
            EmlpError=np.mean(ErrorTrain)*(1+ self.ConfigDict['PenaltyCoefficient'][0]*NValidAvg + self.ConfigDict['PenaltyCoefficient'][1]*NValidBad)
        else:
            YValid=self.ConfigDict['Y'][ValidDataIndex]
            ErrorValid=np.abs((YValidPred-YValid)/(YValid+self.ConfigDict['PercentageErrorBias']))
            NValidAvg=np.sum((ErrorValid>0.15)&(ErrorValid<0.25))
            NValidBad=np.sum(ErrorValid>0.25)
            EmlpError=(np.mean(ErrorTrain)+np.mean(ErrorValid))*(1+ self.ConfigDict['PenaltyCoefficient'][0]*NValidAvg + self.ConfigDict['PenaltyCoefficient'][1]*NValidBad)
        # 返回两个目标的适应度值
        out['F']=np.array([sum(ModelTopology_Int),EmlpError])
        # 记录各种误差
        ModelData['EmlpError']=EmlpError
        YPred=np.vstack((YTrainPred,YValidPred)) if UseValidationCheck else YTrainPred
        # 映射回原来的大小
        YPred=self.ConfigDict['YMin']+(self.ConfigDict['YMax']-self.ConfigDict['YMin'])*YPred
        Y=self.ConfigDict['YMin']+(self.ConfigDict['YMax']-self.ConfigDict['YMin'])*self.ConfigDict['Y']
        ModelData['TotalPercentageRmseError']=math.sqrt( np.mean( ((Y-YPred)/(Y+1e-7))**2 ) )
        ModelData['TotalRmseError']=math.sqrt( np.mean( (Y-YPred)**2 ) )
        ModelData['ModelDictionary']=Model.state_dict()
        ModelData['Dimension']=self.ConfigDict['Dimension']
        ModelData['YMin']=self.ConfigDict['YMin']
        ModelData['YMax']=self.ConfigDict['YMax']
        ModelData['XMin']=self.ConfigDict['XMin']
        ModelData['XMax']=self.ConfigDict['XMax']
        ModelData['ComponentMatrix']=self.ConfigDict['ComponentMatrix']
        ModelData['StandardScaler']=self.ConfigDict['StandardScaler']
        # 评估帕累托解集
        self.ParetoSetNeuralNetwork[tuple(ModelTopology_Int)]=ModelData
        if len(self.ParetoSetNeuralNetwork)>1:
            DominateCounter={}
            # 被支配计数器
            for key in self.ParetoSetNeuralNetwork:
                DominateCounter[key]=0
            for key,value in self.ParetoSetNeuralNetwork.items():
                a=np.array([ModelData['EmlpError'],sum (ModelData['TopologyInInt'])],dtype=np.float32)
                b=np.array([value['EmlpError'],sum (value['TopologyInInt'])],dtype=np.float32)
                A_Dominate_B=EmlpProblem.is_a_dominate_b(a,b)
                if A_Dominate_B==1:
                    DominateCounter[key]=DominateCounter[key]+1
                elif A_Dominate_B==-1:
                    DominateCounter[tuple(ModelTopology_Int)]=DominateCounter[tuple(ModelTopology_Int)]+1
            for key,value in DominateCounter.items():
                if value>0:
                    del self.ParetoSetNeuralNetwork[key]
        TotalEvaluateTimes=self.ConfigDict['PopSize']*self.ConfigDict['MaxIteration']
        self.CurrentEvaluateTimes+=1
        print(f"\n总评估次数:{TotalEvaluateTimes}     当前进度:{100*self.CurrentEvaluateTimes/TotalEvaluateTimes:.4f}%     第{1+math.floor((self.CurrentEvaluateTimes-1)/self.ConfigDict['PopSize'])}轮、第{1+((self.CurrentEvaluateTimes-1) % self.ConfigDict['PopSize'])}次评估     总Rmse误差:{ModelData['TotalRmseError']:.6e}     EmlpError误差为:{EmlpError:.6e}\n")
    