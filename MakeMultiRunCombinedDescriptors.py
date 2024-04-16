#Combine the run data and descriptors from multiple runs
import pandas as pd
import os
import numpy as np

Runs = ["2pure_HCl_run1","3pure_1prot_Cl_run1","prot1_Cl_run3","run4_3mol_cubic","Piperidine_2unit_HCl"]#['135Triazine_4unit', '135Triazine_4unit-pt3', '135Triazine_4unit-pt4']#,'135Triazine_1prot_Br','123Triazine_1prot_Cl','124Triazine_1prot_Cl','Pyridine_1prot_Cl','Thiophene_1prot_Cl']
RedirectFolder = ""
TopFolder = RedirectFolder + "AllDescriptors/"

OutputFolder = TopFolder #+
CombinedName = '+'.join(Runs)
if not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder)

VASPData = pd.DataFrame()

for Run in Runs:
        NewData = pd.read_csv(TopFolder + "VASPData-"+ Run +".csv", index_col=False)
        VASPData = pd.concat([VASPData,NewData])


DescriptorData = pd.DataFrame()
for Run in Runs:
        NewData = pd.read_csv(TopFolder +"AllDescriptors-" + Run + ".csv", index_col=False)
        DescriptorData = DescriptorData.append(NewData, ignore_index=True)

DescriptorData.fillna(0,inplace=True)
#Do not want weights in the descriptor file - number of materials will change
#keep this section for how to do calculation.
'''#Add a weight column to descriptor file
DescriptorData['Run'] = DescriptorData['MaterialID'].str.split('-').str[0] #classifies each point by run name
RunCount = DescriptorData.Run.value_counts()
z1 = RunCount.to_dict()
print(DescriptorData.Run.value_counts())

DescriptorData['Weight']  = 1/(DescriptorData['Run'].map(z1))
print(DescriptorData.iloc[0])'''

VASPData.to_csv(OutputFolder+"VASPData-" + CombinedName + ".csv",index=False)
DescriptorData.to_csv(OutputFolder+"AllDescriptors-" + CombinedName + ".csv",index=False)











