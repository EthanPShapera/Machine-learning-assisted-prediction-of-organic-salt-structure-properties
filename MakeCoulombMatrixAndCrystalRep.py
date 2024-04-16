'''
Created on Aug 4, 2022

@author: shapera
'''
#constructs coulomb matrix descriptors and descriptors of crystal structure
#run this after making the crystal graph representation
import pandas as pd
from pymatgen.io.cif import CifParser
import numpy as np
from numpy.linalg import eigvalsh
from numpy.linalg import det
import sys

RunName = sys.argv[1] #"prot1_Cl_run3"
RedirectFolder = ""
DataFolder = RedirectFolder + 'CollectedData/' + RunName + "/"
CrystalGraphFile = DataFolder + "CrystalGraphDescriptors.csv"

NumCoulombMatrixelements = 40 #Number of Coulomb matrix elements, include padding, should be longer than number of atoms

CrystalgraphDescriptors = pd.read_csv(CrystalGraphFile, index_col=False)

#np.set_printoptions(precision=2)

f = open(DataFolder + "CoulombDescriptors.csv",'w')
f.write("MaterialID,")
for i in range(0,NumCoulombMatrixelements):
    f.write("CMDescriptor" + str(i) +",")
f.write("CMDescriptorTrace,CMDescriptorDeterminant,CMDescriptorNumPositive,CMDescriptorNumNegative" + "\n")

g = open(DataFolder + "CrystalStructureDescriptors.csv",'w')
g.write("MaterialID,ADescriptor,BDescriptor,CDescriptor,alphaDescriptor,betaDescriptor,gammaDescriptor" + "\n")

for i in range(0,len(CrystalgraphDescriptors)):
    CurrentMat = CrystalgraphDescriptors.iloc[i]["MaterialID"]
    print(i, CurrentMat)
    #read .cif
    structure = CifParser(DataFolder + CurrentMat + ".cif").get_structures()[0]
    NumAtoms = len(structure.species)
    #MAke sure there are enough coulomb matrix elements allowed
    if NumAtoms > NumCoulombMatrixelements:
        print("Too few Coulomb matrix elments requested.")
        print(CurrentMat + " has " + str(NumAtoms) + " atoms")
        break
    #print(structure)
    #print(structure.species)
    #print(structure.atomic_numbers)
    #print(structure.distance_matrix) #verified this works with periodic boundaries
    #Calculate Coulomb matrix
    #only need diagonal and upper/lower triangle
    CMatTriangle=np.zeros((NumCoulombMatrixelements,NumCoulombMatrixelements)) #this format puts 0's in the middle
    CMatDiag=np.zeros((NumCoulombMatrixelements,NumCoulombMatrixelements))
    #first do diagonal
    for j in range(0,NumAtoms):
        CMatDiag[j][j] = 0.5 * structure.atomic_numbers[j] **2.4
    
    #do a triangle
    for j in range(0,NumAtoms-1):
        for k in range(j+1,NumAtoms):
            CMatTriangle[j][k] = structure.atomic_numbers[j] * structure.atomic_numbers[k] / structure.distance_matrix[j][k]
    #print(CMatTriangle) #verified correct
    #print(CMatTriangle.T)
    #print(CMatTriangle + CMatTriangle.T)
    CMat = CMatTriangle + CMatDiag + CMatTriangle.T
    #print(CMat)
    #print(CMatDiag)
    CMatEvals = eigvalsh(CMat)
    #print(CMatEvals)
    
    #write data to Coulomb matrix file
    f.write(CurrentMat + ",")
    for value in CMatEvals:
        f.write(str(value) + ",")
    f.write(str(np.trace(CMat)) + ",")
    NonZeroDet = np.abs(np.prod([a for a in CMatEvals if a !=0])) #tested correct
    ScaledNonZeroDet = np.sign(NonZeroDet) * NonZeroDet ** (1/NumAtoms)
    NumPositive = len([a for a in CMatEvals if a > 0])
    NumNegative = len([a for a in CMatEvals if a < 0])
    f.write(str(ScaledNonZeroDet) + "," + str(NumPositive) + "," + str(NumNegative) +"\n")

    #write data to crystal structure file
    #print(structure.lattice.a)
    #print(structure.lattice.alpha)
    g.write(CurrentMat + ",")
    g.write(str(structure.lattice.a) + ",")
    g.write(str(structure.lattice.b) + ",")
    g.write(str(structure.lattice.c) + ",")
    g.write(str(structure.lattice.alpha) + ",")
    g.write(str(structure.lattice.beta) + ",")
    g.write(str(structure.lattice.gamma) + "\n")


f.close()
g.close()








