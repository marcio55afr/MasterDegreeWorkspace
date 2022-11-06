# Master's Degree Workspace

## Tasks

- [x] Set up the operational system (Manjaro)
- [x] Configure my personal Git
- [x] Download the project
- [x] Install and configure Pycharm
- [x] Configure a virtual environment
- [x] Install libraries and dependencies
  - [x] Install Python
  - [x] Install Sktime
  - [x] Install other dependencies
  - [x] Create a setup file to config the environment
- [x] Download all timeseries datasets
- [x] Centralize the receiving of datasets
- [x] Test the experiments
- [ ] Create a main with all experiments
- [ ] Organize the folder's project
- [ ] Clean up the code of each variant
- [ ] Create a Abstract Class called Variant to keep duplicated codes
- [ ] Rewrite the ensemble variant and the function _fit_discretizers()
- [ ] ...


## Search Technique - functions

    fit( data, labels ){

        resolutions = resolutions_definition( data )
      
        number_of_samples_per_class = 2
      
        while( resolutions.size > 1 ){

            samples = get_samples( data, labels, number_of_samples_per_class )

            word_sequences = discretization_extraction( samples, resolutions.windows )

            ngram_sequences = ngrams_definition( word_sequences, resolutions )

            bag_of_bags = frequency_counter( ngram_sequences )

            resolutions_rank = calcule_separability( bag_of_bags, resolutions )

            resolutions = get_first_half( resolutions_rank )

            number_of_samples_per_class = 2*number_of_samples_per_class
      
        }
      
        word_sequences = discretization_extraction( data, resolutions.windows )

        ngram_sequences = ngrams_definition( word_sequences, resolutions )

        bag_of_bags = frequency_counter( ngram_sequences )
        
        clf = new LogisticRegression()
        
        clf = clf.fit( bag_of_bags, labels )
    
    }


## Search Technique - Sequence Diagram

![alt text](https://github.com/marcio55afr/MasterDegreeWorkspace/blob/main/sequence-diagram.png?raw=true)
