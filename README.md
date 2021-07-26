# Master's Degree Workspace


## Search Technique - functions

    fit( data, labels ){

        resolutions = resolutions_definition( data )
      
        sample_size_per_class = 2
      
        while( resolutions.size > 1 ){

            sample = get_sample( data, labels, sample_size_per_class )

            word_sequences = discretization_extraction( sample, resolutions.windows )

            ngram_sequences = ngrams_definition( word_sequences, resolutions )

            bag_of_bags = frequency_counter( ngram_sequences )

            resolutions_rank = separability_calculation( bag_of_bags, resolutions )

            worst_resolutions = get_last_half( resolutions_rank )

            resolutions = resolutions.remove( worst_resolutions )

            sample_size_per_class = 2*sample_size_per_class
      
        }
      
        word_sequences = discretization_extraction( data, resolutions.windows )

        ngram_sequences = ngrams_definition( word_sequences, resolutions )

        bag_of_bags = frequency_counter( ngram_sequences )
        
        clf = new LogisticRegression()
        
        clf = clf.fit( bag_of_bags, labels )
    
    }
