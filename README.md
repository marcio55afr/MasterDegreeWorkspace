# Master's Degree Workspace


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
