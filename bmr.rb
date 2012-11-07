require 'Jama-1.0.2.jar'
require 'java'
require 'CSV'

import 'Jama.Matrix'

module BatchMultipleRegression
  class TrainingProgram
    def inputs
      @inputs
    end
    
    def inputCount
      @inputCount
    end
    
    def outputs
      @outputs
    end
    
    def outputCount
      @outputCount
    end
    
    def predictions
      @predictions
    end
    
    def observationCount
      @observationCount
    end
    
    def load_dataset(filePath)
      # Initialize the inputs as an array for easy concatenation.
      @inputs = Array.new
      
      # Initialize the outputs as an arrray for easy concatenation.
      @outputs = Array.new
      
      # Initialize with zero observations.
      @observationCount = 0
      
      # Open the file and interate through each parsed row...
      CSV.foreach(filePath) do
        |row|
        
        # Convert each element to a Fixnum from String.
        row.map! {
          |element|
          
          element.to_f
        }
        
        # Use every element except the last as inputs, and add the bias 
        # term.
        @inputs += row[0..row.length - 2] + [1]
        
        # Use the last element as the output.
        @outputs += [row[row.length - 1]]
        
        # Increment the observation count.
        @observationCount += 1    
      end
      
      # Set the input count.
      @inputCount = @inputs.length / @observationCount
      
      # Set the output count.
      # While not strictly necessary, this will make it easier to 
      # incorporate more outputs in the future.
      @outputCount = @outputs.length / @observationCount
      
      # Translate the inputs array into a matrix.
      @inputs = Matrix.new(@inputs.to_java(:double), @observationCount)
      
      # Translate the outputs array into a matrix.
      @outputs = Matrix.new(@outputs.to_java(:double), @observationCount)
    end
    
    def weights
      @weights
    end
    
    def initialize_weights
      random_weights = []
      
      @inputCount.times{
        random_weights += [rand * 2 - 1]
      }
    end
    
    def update_weights
    
    end
    
    def learn_weights
    
    end
    
    def learningRate
      @learningRate
    end
    
    def regularization
      @regulzarization
    end
    
    def write_final_model
    
    end
    
    def train_model
    
    end
  end
end

if __FILE__ == $0
  # Determine the dataset from the first command-line argument.
  datasetFilePath = ARGV[0]

  # If a dataset was not supplied, then default to the file data.csv.
  datasetFilePath ||= 'data.csv'

  # Create the program.
  defaultProgram = TrainingProgram.new

  # Load the dataset.
  defaultProgram.load_dataset(datasetFilePath)
end
