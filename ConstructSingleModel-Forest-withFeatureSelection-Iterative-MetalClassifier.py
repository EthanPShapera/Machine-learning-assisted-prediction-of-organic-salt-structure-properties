import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
from random import randint
import copy
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel

TreeDepths = [8, 10, 12, 15, 20]
MinSplits = [5,10,15,20]

for minsplit in MinSplits:
    for TreeDepth in TreeDepths:
        RunName = "2pure_HCl_run1+3pure_1prot_Cl_run1+prot1_Cl_run3+run4_3mol_cubic"  # "run4_3mol_cubic"
        TargetValue1 = "Metallicity"
        TargetValue2 = "Volume"
        TargetValue3 = "Enthalpy"
        TargetQuantity = "Metallicity"
        RegressionTargetValues = [TargetValue2, TargetValue3]
        ClassTargetValues = [TargetValue1]
        RedirectFolder = ""
        DataFolder = RedirectFolder + "AllDescriptors/"
        ModelFolder = RedirectFolder + 'Models/' + RunName + "/"
        if not os.path.exists(ModelFolder):  # check if DataFolder exists
            os.makedirs(ModelFolder)

        DescriptorFile = DataFolder + "AllDescriptors-" + RunName + ".csv"
        TargetFile = DataFolder + "VASPData-" + RunName + ".csv"
        # OutputFolder = RedirectFolder + "AlgorithmSweep/"

        StandardScaleData = True

        Descriptors = pd.read_csv(DescriptorFile, index_col=False)
        Targets = pd.read_csv(TargetFile, index_col=False)

        FullData = pd.merge(Descriptors, Targets, on='MaterialID')

        ColumnNames = list(FullData.columns)
        DescriptorColumnsAll = [item for item in ColumnNames if "Descriptor" in item]
        DescriptorColumns = []
        NumDescriptors = len(DescriptorColumnsAll)
        for i in range(0, NumDescriptors):
            DescriptorColumns.append(DescriptorColumnsAll[i])

        AdaIters = 0
        CrossValIters = 10
        ValidFrac = 0.1
        TestingFrac = 0.1
        # minsplit=10
        cutoff = 0.1
        NumTrees = 50
        FeatureSelectType = "FromModel"
        Direction = "forward"
        FeatureCutoff = 0.1
        OutputFolder = ModelFolder + TargetQuantity + "_FeatureSelect" + FeatureSelectType + "_AdaForest_" + str(
            AdaIters) + "AdaIters_Depth" + str(TreeDepth) + "_CrossvalIters" + str(CrossValIters) + "_MinSplit" + str(
            minsplit) + "_NumTrees" + str(NumTrees) + "_FeatCut" + str(FeatureCutoff) + "_largeSelectTree/"

        # OutputFolder = RedirectFolder + TargetQuantity + "_Forest_" + "0" + "AdaIters_Depth" + str(TreeDepth) +"_CrossvalIters"+str(CrossValIters)+"/"

        print(OutputFolder)
        if not os.path.exists(OutputFolder):
            os.makedirs(OutputFolder)

        f = open(OutputFolder + "Summary.txt", "w")
        f.write("AdaIters = " + str(AdaIters) + "\n")
        f.write("Tree Depth = " + str(TreeDepth) + "\n")
        f.write("CrossValidation Iters = " + str(CrossValIters) + "\n")
        f.write("Validation Fraction = " + str(ValidFrac) + "\n")
        f.write("Testing Frac = " + str(TestingFrac) + "\n")
        f.write("Number of Descriptors = " + str(NumDescriptors) + "\n")
        f.write("Training Data Size = " + str(FullData.shape[0]) + "\n")
        f.write("Min Split = " + str(minsplit) + "\n")

        # Split into 90-10 to make withheld testing set
        X_FV, X_test, y_FV, y_test = train_test_split(FullData[DescriptorColumns], FullData[TargetQuantity],
                                                      train_size=1.0 - TestingFrac, random_state=randint(1, 10 ** 8))

        # open files to store predictions of target quantity
        f3 = open(OutputFolder + "TestingResults.csv", "w")
        f3p = open(OutputFolder + "TestingResultsProb.csv", "w")
        f5 = open(OutputFolder + "FittingResults.csv", "w")
        f6 = open(OutputFolder + "ValidationResults.csv", "w")
        #f7 = open(OutputFolder + "ErrorSummary.csv", "w")

        f3.write("ID" + "," + "Calculated" + ",")
        f3p.write("ID" + "," + "CalculatedMetal" + ",")
        for ii in range(0, CrossValIters):
            f3.write("Iteration" + str(ii) + ",")
            f3p.write("Iteration" + str(ii) + ",")
        f3.write("\n")
        f3p.write("\n")
        #f3.write("AveragePrediction" + "," + "PredictionSTD" + "," + "PredictionError" + "\n")
        f5.write('ID' + ',' + 'Iteration' + ',' + 'Calculated' + ',' + 'ML' + ',ProbMetal,ProbNonMetal'+ '\n')
        f6.write('ID' + ',' + 'Iteration' + ',' + 'Calculated' + ',' + 'ML' + ',ProbMetal,ProbNonMetal'+ '\n')
        #f7.write("Iteration" + "," + "FittingMAE" + "," + "ValidMAE" + "," + "TestingMAE" + "\n")

        print(X_FV)
        # scale descriptor values
        sc = StandardScaler()
        X_FV = sc.fit_transform(X_FV)
        X_test = sc.transform(X_test)
        print(X_FV.shape)

        # feature selector
        # tree = DecisionTreeRegressor(max_depth=5,min_samples_split=minsplit)
        # sfs_forward = SequentialFeatureSelector(tree, n_features_to_select=20, direction=Direction,cv=2).fit(X_FV, y_FV)
        # for item in DescriptorColumns[sfs_forward.get_support()]:
        #    print(item)

        # Select from Model - will used a single decision tree
        if FeatureSelectType == "FromModel":
            f8 = open(OutputFolder + "ModelDescriptorSelections.csv", "w")
            # tree = DecisionTreeRegressor(max_depth=TreeDepth,min_samples_split=minsplit) #varying feature select tree
            tree = DecisionTreeClassifier(max_depth=20, min_samples_split=10,class_weight='balanced')  # large select tree
            # tree = AdaBoostRegressor(DecisionTreeRegressor(max_depth=TreeDepth,min_samples_split=minsplit),n_estimators=10) #adaboosting does not improve model
            selector = SelectFromModel(estimator=tree, threshold=str(FeatureCutoff) + "*mean").fit(X_FV, y_FV)
            importances = selector.estimator_.feature_importances_
            support = selector.get_support()
            for i in range(0, len(DescriptorColumns)):
                f8.write(DescriptorColumns[i] + "," + str(importances[i]) + "," + str(support[i]) + "\n")
            f8.write("Threshold," + str(selector.threshold_) + "\n")
            f8.write("Features," + str(list(support).count(True)))
        X_FV = selector.transform(X_FV)
        print(X_FV.shape)
        X_test = selector.transform(X_test)
        f.write("Reduced Number of Descriptors = " + str(X_FV.shape[1]) + "\n")

        # '''
        TestingPredicted = [[]] * CrossValIters  # will hold predicted values for each material excluded from fitting and validation
        TestingPredictedProb = [[]] * CrossValIters

        for i in range(0, CrossValIters):
            print(i)
            X_fit, X_valid, y_fit, y_valid = train_test_split(X_FV, y_FV, train_size=int(
                np.ceil(len(FullData) * (1 - ValidFrac - TestingFrac))), random_state=randint(1, 10 ** 8))
            # print(len(X_fit),len(X_valid),len(X_test))

            if AdaIters == 0:
                regr_1 = RandomForestClassifier(max_depth=TreeDepth, min_samples_split=minsplit, n_estimators=NumTrees,class_weight="balanced_subsample")
            else:
                regr_1 = AdaBoostRegressor(
                    RandomForestClassifier(max_depth=TreeDepth, min_samples_split=minsplit, n_estimators=NumTrees,class_weight="balanced_subsample"),
                    n_estimators=AdaIters)
            # regr_1 = RandomForestRegressor(max_depth=TreeDepth)
            regr_1.fit(X_fit, y_fit)
            FittingIndices = list(y_fit.index)
            ValidationIndices = list(y_valid.index)
            TestingIndices = list(y_test.index)

            # write fitting results to file
            FittingPredictions = regr_1.predict(X_fit)
            FittingPredictionsProb = regr_1.predict_proba(X_fit)
            MetalIndex = list(regr_1.classes_).index(True)
            NonMetalIndex = list(regr_1.classes_).index(False)
            # for SampleIndex in FittingIndices:
            SampleNames = FullData["MaterialID"].loc[FittingIndices]
            # print(SampleNames)
            # print(y_fit)
            # print(FittingPredictions)
            for j in range(0, len(FittingIndices)):
                f5.write(
                    SampleNames.iloc[[j]].values[0] + "," + str(i) + "," + str(y_fit.iloc[[j]].values[0]) + "," + str(
                        FittingPredictions[j]) + ",")
                f5.write(str(FittingPredictionsProb[j][MetalIndex]) + "," + str(FittingPredictionsProb[j][NonMetalIndex]))
                f5.write("\n")
            #FittingMAE = np.mean([abs(a - b) for a, b in zip(list(y_fit), FittingPredictions)])
            # write validation results to file
            ValidationPredictions = regr_1.predict(X_valid)
            ValidationPredictionsProb = regr_1.predict_proba(X_valid)
            SampleNames = FullData["MaterialID"].loc[ValidationIndices]
            for j in range(0, len(ValidationIndices)):
                f6.write(
                    SampleNames.iloc[[j]].values[0] + "," + str(i) + "," + str(y_valid.iloc[[j]].values[0]) + "," + str(
                        ValidationPredictions[j]) + ",")
                f6.write(str(ValidationPredictionsProb[j][MetalIndex]) + "," + str(ValidationPredictionsProb[j][NonMetalIndex]))
                f6.write("\n")
            #ValidMAE = np.mean([abs(a - b) for a, b in zip(list(y_valid), ValidationPredictions)])
            # Get testing results
            TestingPredictions = regr_1.predict(X_test)
            TestingPredictionsProb = regr_1.predict_proba(X_test)
            #Store only the probability of being metal
            TestingPredictionsProbList = [a[MetalIndex] for a in TestingPredictionsProb]
            TestingPredicted[i] = copy.copy(TestingPredictions)
            TestingPredictedProb[i] = copy.copy(TestingPredictionsProbList)
            #TestingMAE = np.mean([abs(a - b) for a, b in zip(list(y_test), TestingPredictions)])
            # write MAEs to file
            #f7.write(str(i) + "," + str(FittingMAE) + "," + str(ValidMAE) + "," + str(TestingMAE) + "\n")

        # write testing results to file
        TestNames = FullData["MaterialID"].loc[TestingIndices]
        TestingPredicted = list(map(list, zip(*TestingPredicted)))
        TestingPredictedProb = list(map(list, zip(*TestingPredictedProb)))
        #TestingMean = np.atleast_1d(np.mean(TestingPredicted, axis=1))
        #TestingSTD = np.atleast_1d(np.std(TestingPredicted, axis=1))
        for i in range(0, len(TestingIndices)):
            f3.write(TestNames.iloc[[i]].values[0] + ",")  # write name
            f3p.write(TestNames.iloc[[i]].values[0] + ",")  # write name
            f3.write(str(y_test.iloc[[i]].values[0]) + ",")
            f3p.write(str(int(y_test.iloc[[i]].values[0])) + ",")
            for value in TestingPredicted[i]:
                f3.write(str(value) + ",")
            for value in TestingPredictedProb[i]:
                f3p.write(str(value) + ",")
            f3.write("\n")
            f3p.write("\n")
            #f3.write(str(TestingMean[i]) + "," + str(TestingSTD[i]) + ",")
            #f3.write(str(y_test.iloc[[i]].values[0] - TestingMean[i]) + "\n")

        # '''
        f.close()
        f3.close()
        f3p.close()
        f5.close()
        f6.close()
        #f7.close()
        f8.close()
