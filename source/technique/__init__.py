# -*- coding: utf-8 -*-


__all__ = ['SearchTechnique',
           'SearchTechniqueCV',
           'SearchTechnique_CV_RFSF',
           'SearchTechnique_SG',
           'SearchTechnique_SG_RR',
           'SearchTechnique_KWS',
           'SearchTechnique_SG_CLF',
           'SearchTechnique_MR']



from source.technique.search_technique import SearchTechnique
from source.technique.search_technique_CV import SearchTechniqueCV
from source.technique.search_technique_CV_RFSF import SearchTechnique_CV_RFSF
from source.technique.search_technique_SG import SearchTechnique_SG
from source.technique.search_technique_SG_RR import SearchTechnique_SG_RR
from source.technique.search_technique_KWS import SearchTechnique_KWS
from source.technique.search_technique_SG_CLF import SearchTechnique_SG_CLF
from source.technique.search_technique_MR import SearchTechnique_MR

#from source.technique.word_ranking import WordRanking
#from source.technique.resolution_selector import ResolutionSelector