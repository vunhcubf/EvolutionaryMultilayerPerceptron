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

class EMLPHandler:
    def __init__(self,ParetoSetNeuralNetwork):
        self.ParetoSetNeuralNetwork=ParetoSetNeuralNetwork
    def DrawParetoSet(self):
        complexity=[]
        accuracy=[]
        for key,value in self.ParetoSetNeuralNetwork.items():
            complexity.append(sum(value['TopologyInInt']))
            accuracy.append(value['EmlpError'])
        plt.scatter(complexity,accuracy,s=20,edgecolors='red',facecolor='none')
        plt.xlabel('Complexity')
        plt.ylabel('EmlpError')
        plt.title('Pareto Set')
        plt.show()
    def PrintModel(self):
        for key,value in self.ParetoSetNeuralNetwork.items():
            print(f"模型网络结构为:{value['TopologyInInt']},\t训练误差为{value['EmlpError']}\t总神经元数:{sum(value['TopologyInInt'])}")
    def PredictModel(self,key_of_model,x):
        n_var=self.ParetoSetNeuralNetwork[tuple(key_of_model)]['Dimension']
        x_min=self.ParetoSetNeuralNetwork[tuple(key_of_model)]['XMin']
        x_max=self.ParetoSetNeuralNetwork[tuple(key_of_model)]['XMax']
        y_min=self.ParetoSetNeuralNetwork[tuple(key_of_model)]['YMin']
        y_max=self.ParetoSetNeuralNetwork[tuple(key_of_model)]['YMax']
        activate_function=self.ParetoSetNeuralNetwork[tuple(key_of_model)]['ActivateFunction']
        State_Dict=self.ParetoSetNeuralNetwork[tuple(key_of_model)]['ModelDictionary']
        key_of_model=np.array(key_of_model)
        key_of_model=key_of_model[key_of_model!=0]
        x=(x-x_min)/(x_max-x_min)
        x=x.astype(np.float32)
        x=torch.from_numpy(x)
        model=EMLP.Net(n_var,key_of_model,1,activate_function)
        model.load_state_dict(State_Dict)
        y=model(x).data.numpy()
        y=y_min+y*(y_max-y_min)
        return y
class EMLP:
    def is_a_dominate_b(a_cost, b_cost):
        if np.all(a_cost <= b_cost) and np.any(a_cost < b_cost):
            result = 1
        elif np.all(b_cost <= a_cost) and np.any(b_cost < a_cost):
            result = -1
        else:
            result = 0
        return result
    # 核心功能，序列化保存网络
    class Net(nn.Module):
        def __init__(self,n_in,params,n_out,activate_func) :
            self.params=params
            self.activate_func=activate_func
            super().__init__()
            if params.size==1:
                self.layer1=nn.Linear(n_in,params[0])
                self.layer2=nn.Linear(params[0],n_out)
            elif params.size==2:
                self.layer1=nn.Linear(n_in,params[0])
                self.layer2=nn.Linear(params[0],params[1])
                self.layer3=nn.Linear(params[1],n_out)
            elif params.size==3:
                self.layer1=nn.Linear(n_in,params[0])
                self.layer2=nn.Linear(params[0],params[1])
                self.layer3=nn.Linear(params[1],params[2])
                self.layer4=nn.Linear(params[2],n_out)
            elif params.size==4:
                self.layer1=nn.Linear(n_in,params[0])
                self.layer2=nn.Linear(params[0],params[1])
                self.layer3=nn.Linear(params[1],params[2])
                self.layer4=nn.Linear(params[2],params[3])
                self.layer5=nn.Linear(params[3],n_out)
            elif params.size==5:
                self.layer1=nn.Linear(n_in,params[0])
                self.layer2=nn.Linear(params[0],params[1])
                self.layer3=nn.Linear(params[1],params[2])
                self.layer4=nn.Linear(params[2],params[3])
                self.layer5=nn.Linear(params[3],params[4])
                self.layer6=nn.Linear(params[4],n_out)
            elif params.size==6:
                self.layer1=nn.Linear(n_in,params[0])
                self.layer2=nn.Linear(params[0],params[1])
                self.layer3=nn.Linear(params[1],params[2])
                self.layer4=nn.Linear(params[2],params[3])
                self.layer5=nn.Linear(params[3],params[4])
                self.layer6=nn.Linear(params[4],params[5])
                self.layer7=nn.Linear(params[5],n_out)
        def activate_function(self,x):
            if self.activate_func==0:
                x=torch.sigmoid(x)
            elif self.activate_func==1:
                x=torch.tanh(x)
            elif self.activate_func==2:
                x=torch.relu(x)
            elif self.activate_func==3:
                return x
            return x
        def forward(self,x):
            if self.params.size==1:
                x=self.layer1(x)
                x=self.activate_function(x)
                x=self.layer2(x)
            elif self.params.size==2:
                x=self.layer1(x)
                x=self.activate_function(x)
                x=self.layer2(x)
                x=self.activate_function(x)
                x=self.layer3(x)
            elif self.params.size==3:
                x=self.layer1(x)
                x=self.activate_function(x)
                x=self.layer2(x)
                x=self.activate_function(x)
                x=self.layer3(x)
                x=self.activate_function(x)
                x=self.layer4(x)
            elif self.params.size==4:
                x=self.layer1(x)
                x=self.activate_function(x)
                x=self.layer2(x)
                x=self.activate_function(x)
                x=self.layer3(x)
                x=self.activate_function(x)
                x=self.layer4(x)
                x=self.activate_function(x)
                x=self.layer5(x)
            elif self.params.size==5:
                x=self.layer1(x)
                x=self.activate_function(x)
                x=self.layer2(x)
                x=self.activate_function(x)
                x=self.layer3(x)
                x=self.activate_function(x)
                x=self.layer4(x)
                x=self.activate_function(x)
                x=self.layer5(x)
                x=self.activate_function(x)
                x=self.layer6(x)
            elif self.params.size==6:
                x=self.layer1(x)
                x=self.activate_function(x)
                x=self.layer2(x)
                x=self.activate_function(x)
                x=self.layer3(x)
                x=self.activate_function(x)
                x=self.layer4(x)
                x=self.activate_function(x)
                x=self.layer5(x)
                x=self.activate_function(x)
                x=self.layer6(x)
                x=self.activate_function(x)
                x=self.layer7(x)
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
    class opt_problem(Problem):
        def __init__(self,config,x_train,x_valid,y_train,y_valid,nn_buffer, **kwargs):
            lb=np.array(config["LowerBound"],dtype=np.int32)
            ub=np.array(config["UpperBound"],dtype=np.int32)
            super().__init__(n_var=ub.size, n_obj=2, xl=lb, xu=ub, vtype=int,elementwise=True, **kwargs)
            self.nn_buffer=nn_buffer
            self.x_train=x_train
            self.x_valid=x_valid
            self.y_train=y_train
            self.y_valid=y_valid
            self.config=config
            self.times=0
            if config['ActivateFunction'] == 'Sigmoid':
                self.activate_func=0
            elif config['ActivateFunction'] == 'Tanh':
                self.activate_func=1
            elif config['ActivateFunction'] == 'Relu':
                self.activate_func=2
            elif config['ActivateFunction'] == 'Linear':
                self.activate_func=3
            else:
                raise RuntimeError("invalid activate function")
        def _evaluate(self, x, out, *args, **kwargs):
            # 创建一个字典来保存这个网络的数据
            params_float=x.tolist()
            params_int=np.round(x).astype(np.int32).tolist()
            nn_data_dict={}
            nn_data_dict['TopologyInFloat']=params_float
            nn_data_dict['TopologyInInt']=params_int
            nn_data_dict['ActivateFunction']=self.activate_func
            
            model,y_valid_pred,y_train_pred,current_error=EMLP.train_nn(params_int,self.x_train,self.x_valid,self.y_train,self.y_valid,self.config,self.x_train.shape[1],self.activate_func)
            model=model.cpu()

            bias=self.config['PercentageErrorBias']
            Etst=np.mean(np.abs((y_valid_pred-self.y_valid)/(self.y_valid+bias)))
            Etrn=np.mean(np.abs((y_train_pred-self.y_train)/(self.y_train+bias)))
            f=np.array([sum(params_int),(Etst+Etrn)*(1+0.33*Etst[Etst>0.15 and Etst<0.25].size+Etst[Etst>0.25].size)])
            nn_data_dict['EmlpError']=f[1]
            nn_data_dict['ModelDictionary']=model.state_dict()
            nn_data_dict['Dimension']=self.x_train.shape[1]
            # 评估帕累托最优解集
            self.nn_buffer[tuple(params_int)]=nn_data_dict
            if len(self.nn_buffer)>1:
                dominate_counter={}
                # 被支配计数器
                for key in self.nn_buffer:
                    dominate_counter[key]=0
                for key,value in self.nn_buffer.items():
                    a=np.array([nn_data_dict['EmlpError'],sum (nn_data_dict['TopologyInInt'])],dtype=np.float32)
                    b=np.array([value['EmlpError'],sum (value['TopologyInInt'])],dtype=np.float32)
                    A_Dominate_B=EMLP.is_a_dominate_b(a,b)
                    if A_Dominate_B==1:
                        dominate_counter[key]=dominate_counter[key]+1
                    elif A_Dominate_B==-1:
                        dominate_counter[tuple(params_int)]=dominate_counter[tuple(params_int)]+1
                for key,value in dominate_counter.items():
                    if value>0:
                        del self.nn_buffer[key]
            total_evaluate_times=self.config['PopSize']*self.config['MaxIteration']
            self.times+=1
            print(f"\n总评估次数:{total_evaluate_times}     当前进度:{100*self.times/total_evaluate_times:.4f}%     第{1+math.floor((self.times-1)/self.config['PopSize'])}轮、第{1+((self.times-1) % self.config['PopSize'])}次评估     Mse误差为:{current_error:.6e}     EmlpError误差为:{f[1]:.6e}\n")
            out["F"] =f
    def train_nn(params,x_train,x_valid,y_train,y_valid,config,n_vars,activate_func):
        params=np.array(params)
        params=params[params!=0]
        x_train=torch.from_numpy(x_train).float().to('cuda')
        y_train=torch.from_numpy(y_train).float().to('cuda')
        x_valid=torch.from_numpy(x_valid).float().to('cuda')
        y_valid=torch.from_numpy(y_valid).float().to('cuda')
        max_iter=config['Epochs']
        model=EMLP.Net(n_vars,params,1,activate_func).to('cuda')
        criterion=nn.MSELoss()
        opt=torch.optim.Adam(model.parameters(),lr=config['MaxLearningRate'])
        lr_scheduler_stepsize=round(config['Epochs']*0.01)
        gamma=(config['MinLearningRate']/config['MaxLearningRate'])**(1/100)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=lr_scheduler_stepsize,gamma=gamma)
        validcheck=EMLP.ValidationCheck(config['MaxValidationCheck'])
        current_error=0
        # 控制台输出
        last_message="a"
        for epoch in range(max_iter):
            y_pred=model(x_train)
            loss=criterion(y_pred,y_train)
            # 进行验证检查
            y_pred_valid=model(x_valid)
            loss.backward()
            if (epoch+1)%100==0:
                sys.stdout.write('\r')
                current_message=f"训练轮次:{epoch+1}     训练误差:{loss.item():.7e}     验证检查失败次数:{validcheck.counter}     神经网络结构:{params}     学习率:{opt.param_groups[0]['lr']:.5e}"
                sys.stdout.write(current_message.ljust(len(last_message)))
                sys.stdout.flush()
                last_message=current_message
            current_error=loss.item()
            opt.step()
            if lr_scheduler_stepsize != 0:
                scheduler.step()
            opt.zero_grad()
            quit_flag=False
            if (epoch+1) % 100==0:
                quit_flag=validcheck.update(y_valid,y_pred_valid)
            if loss.item()<=config['Precision']:
                quit_flag=True
            if quit_flag:
                break
        y_train_pred=model(x_train).cpu().data.numpy()
        y_valid_pred=model(x_valid).cpu().data.numpy()
        return (model,y_valid_pred,y_train_pred,current_error)
    def __init__(self,dict):
        # 保存训练的网络
        self.ParetoSetNeuralNetwork={}
        # 输入字典中的数据集：输入x，输出y，网络结构的上下限，种群数量，迭代次数，目标误差，验证检查次数，输入数据一行为一个数据，训练集占比，训练最大迭代次数
        data_count=dict['X'].shape[0]
        train_data_percentage=dict['TrainDataPercentage']
        rnd_index=np.arange(0,data_count)
        np.random.shuffle(rnd_index)
        train_index=rnd_index[0:round(train_data_percentage*data_count)]
        valid_index=rnd_index[round(train_data_percentage*data_count):]
        
        # 归一化数据到01
        x=dict['X']
        x=x.astype(np.float32)
        x_min=np.amin(x,axis=0)
        x_max=np.amax(x,axis=0)
        self.XMin=x_min
        self.XMax=x_max
        x=(x-x_min)/(x_max-x_min)
        y=dict['Y']
        y=y.astype(np.float32)
        y_min=np.min(y)
        y_max=np.max(y)
        self.YMin=y_min
        self.YMax=y_max
        y=(y-y_min)/(y_max-y_min)
        x_train=x[train_index]
        y_train=y[train_index]
        x_valid=x[valid_index]
        y_valid=y[valid_index]
        
        # 创建问题
        self.Problem=self.opt_problem(dict,x_train,x_valid,y_train,y_valid,self.ParetoSetNeuralNetwork)
        self.MaxIteration=dict['MaxIteration']
        self.algorithm = NSGA2(pop_size=dict['PopSize'])
    def Train(self):
        minimize(self.Problem,self.algorithm,('n_gen', self.MaxIteration),verbose=False)
        # 为所有网络添加缩放数据
        for key in self.ParetoSetNeuralNetwork:
            self.ParetoSetNeuralNetwork[key]['XMin']=self.XMin
            self.ParetoSetNeuralNetwork[key]['XMax']=self.XMax
            self.ParetoSetNeuralNetwork[key]['YMin']=self.YMin
            self.ParetoSetNeuralNetwork[key]['YMax']=self.YMax
    def GetTrainResult(self):
        return self.ParetoSetNeuralNetwork
    def DrawParetoSet(self):
        complexity=[]
        accuracy=[]
        for key,value in self.ParetoSetNeuralNetwork.items():
            complexity.append(sum(value['TopologyInInt']))
            accuracy.append(value['EmlpError'])
        plt.scatter(complexity,accuracy,s=20,edgecolors='red',facecolor='none')
        plt.xlabel('Complexity')
        plt.ylabel('EmlpError')
        plt.title('Pareto Set')
        plt.show()
