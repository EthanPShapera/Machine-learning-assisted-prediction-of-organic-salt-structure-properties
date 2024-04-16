'''
Created on Aug 4, 2022

@author: shapera
'''
#Combines Coulomb matrix, Crystal graph and crystal structure descriptors into a single file
import pandas as pd
import shutil
import sys

RunName = sys.argv[1] # "prot1_Cl_run3" #"prot1_Cl_run3"
RedirectFolder = ""
DataFolder = RedirectFolder + 'CollectedData/' + RunName + "/"
AllDescriptorsFolder = RedirectFolder + 'AllDescriptors/'
CoulombFile = DataFolder + "CoulombDescriptors.csv"
CrystalGraphFile = DataFolder + "CrystalGraphDescriptors.csv"
CrystalStructFile = DataFolder + "CrystalStructureDescriptors.csv"
VASPFile = DataFolder + "VASPData.csv"
AllDescriptorsFile = DataFolder + "AllDescriptors.csv"

CoulombDescr = pd.read_csv(CoulombFile, index_col=False)
CrystalGDescr = pd.read_csv(CrystalGraphFile, index_col=False)
CrystalSDescr = pd.read_csv(CrystalStructFile, index_col=False)

FullData = pd.merge(CoulombDescr, CrystalGDescr, on='MaterialID')
FullData = pd.merge(FullData, CrystalSDescr, on='MaterialID')

FullData.to_csv(AllDescriptorsFile,index=False)
#FullData.to_csv(AllDescriptorsFolder+"AllDescriptors-"+RunName+".csv",index=False)
#copy descriptor and VASP data files to AllDescriptorsFolder
shutil.copyfile(AllDescriptorsFile,AllDescriptorsFolder+"AllDescriptors-"+RunName+".csv")
shutil.copyfile(VASPFile,AllDescriptorsFolder+"VASPData-"+RunName+".csv")














