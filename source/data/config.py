# -*- coding: utf-8 -*-
import os

UNIVARIATE_TS_LINK = 'https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip'

UNIVARIATE_TS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "Univariate2018_ts")
UNIVARIATE_HDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "Univariate2018_hdf")

DATASET_NAMES = [
                'SmoothSubspace',
                'Chinatown',
                'ItalyPowerDemand',
                'SyntheticControl',
                'SonyAIBORobotSurface1',
                'DistalPhalanxOutlineAgeGroup',
                'DistalPhalanxOutlineCorrect',
                'DistalPhalanxTW',
                'MiddlePhalanxOutlineAgeGroup',
                'MiddlePhalanxOutlineCorrect',
                'MiddlePhalanxTW',
                'PhalangesOutlinesCorrect',
                'ProximalPhalanxOutlineAgeGroup',
                'ProximalPhalanxOutlineCorrect',
                'ProximalPhalanxTW',
                'TwoLeadECG',
                'MoteStrain',
                'ECG200',
                'MedicalImages']

LARGER_DATASETS_NAMES = ['ACSF1', 'Adiac', 'ChlorineConcentration', 'Computers',
       'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DistalPhalanxOutlineAgeGroup',
       'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200',
       'ECG5000', 'EOGHorizontalSignal', 'EOGVerticalSignal',
       'Earthquakes', 'ElectricDevices', 'EthanolLevel', 'FaceAll',
       'FacesUCR', 'FiftyWords', 'Fish', 'FordA',
       'FreezerRegularTrain', 'GunPointAgeSpan',
       'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham',
       'Haptics', 'InlineSkate', 'InsectWingbeatSound',
       'LargeKitchenAppliances', 'MedicalImages',
       'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
       'MiddlePhalanxTW', 'MixedShapesRegularTrain',
       'MixedShapesSmallTrain', 'OSULeaf', 'PhalangesOutlinesCorrect',
       'Phoneme', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
       'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
       'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
       'RefrigerationDevices', 'ScreenType', 'SemgHandGenderCh2',
       'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapesAll',
       'SmallKitchenAppliances', 'SmoothSubspace', 'Strawberry',
       'SwedishLeaf', 'SyntheticControl', 'Trace', 'TwoPatterns',
       'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
       'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer',
       'WordSynonyms', 'Worms', 'WormsTwoClass','Yoga']