# -*- coding: utf-8 -*-
import os

UNIVARIATE_TS_LINK = 'https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip'

UNIVARIATE_TS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "Univariate_ts")
UNIVARIATE_HDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "Univariate2018_hdf/")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))


EXTENSION = '.h5'

LONGEST_DATASETS = ['Strawberry', 'MiddlePhalanxOutlineCorrect', 'Wafer', 'TwoPatterns', 'PhalangesOutlinesCorrect']
# Crop too large for tuning parameters...
WIDEST_DATASETS = ['ACSF1', 'CinCECGTorso', 'InlineSkate', 'HouseTwenty', 'Rock']
LARGEST_DATASETS = ['NonInvasiveFetalECGThorax2', 'FordB', 'HandOutlines']
# PigCVP, PigArtPressure and PigAirwayPressure has less than 5 objects per class


DATA_INFO = 'source/data/info.csv'

# Univariate equal length datasets
DATASET_NAMES = [
    "ACSF1", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME",
    "Car",     "CBF",     "Chinatown",     "ChlorineConcentration",     "CinCECGTorso",     "Coffee",     "Computers",
    "CricketX",     "CricketY",     "CricketZ",     "Crop",     "DiatomSizeReduction", "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",     "DistalPhalanxTW",     "Earthquakes",     "ECG200", "ECG5000", "ECGFiveDays",
    "ElectricDevices",     "EOGHorizontalSignal",     "EOGVerticalSignal",     "EthanolLevel", "FaceAll", "FaceFour",
    "FacesUCR",     "FiftyWords",     "Fish",     "FordA",     "FordB",     "FreezerRegularTrain", "FreezerSmallTrain",
    "Fungi",     "GunPoint",     "GunPointAgeSpan",     "GunPointMaleVersusFemale",    "GunPointOldVersusYoung", "Ham",
    "HandOutlines",     "Haptics",     "Herring",     "HouseTwenty", "InlineSkate",     "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",     "InsectWingbeatSound",     "ItalyPowerDemand",  "LargeKitchenAppliances", "Lightning2",
    "Lightning7",     "Mallat",     "Meat",     "MedicalImages",     "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",     "MiddlePhalanxTW",     "MixedShapesRegularTrain",     "MixedShapesSmallTrain",
    "MoteStrain",     "NonInvasiveFetalECGThorax1",     "NonInvasiveFetalECGThorax2",     "OliveOil", "OSULeaf",
    "PhalangesOutlinesCorrect", "Phoneme", "PigAirwayPressure", "PigArtPressure", "PigCVP", "Plane", "PowerCons",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW", "RefrigerationDevices",
    "Rock", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2", "ShapeletSim", "ShapesAll",
    "SmallKitchenAppliances", "SmoothSubspace", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
    "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace",
    "TwoLeadECG", "TwoPatterns", "UMD", "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ", "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"
]

SHORTER_DATASETS_NAMES = [
    'ACSF1',
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

LARGER_DATASETS_NAMES = [
    'ACSF1', 'Adiac', 'ChlorineConcentration', 'Computers',
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
    'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']
