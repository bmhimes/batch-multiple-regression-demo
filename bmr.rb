require 'Jama-1.0.2.jar'
require 'java'
require 'CSV'

import 'Jama.Matrix'

# This entire module assumes a single output.  The intent is to eventually refactor to allow multiple outputs.
module BatchMultipleRegression


  def self.print_matrix(matrix, label = '', roundDigits = '', separator = ',')
    elements = matrix.getRowPackedCopy.to_a 
    rows = matrix.getRowDimension
    columns = matrix.getColumnDimension
    if label != ''
      puts "#{label}: "
    end

    rows.times do  |row|
      startIndex = row * columns
      endIndex = (row + 1) * columns - 1
      rowElements = elements[startIndex..endIndex]
      rowElements.map!{  |e| roundDigits ? e.to_f.round(roundDigits) : e}
      puts rowElements.join(separator)
    end
  end


  class TrainingProgram

    attr_accessor :learningRate

    attr_accessor :regularization

    attr_accessor :learningRateAdjustmentFactor

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

    def error
      @error
    end

    def record_error_history
      @iterationHistory[@iteration][:error] = @error
    end

    def errorImprovement
      @errorImprovement
    end

    def record_error_improvement_history
      @iterationHistory[@iteration][:errorImprovement] = @errorImprovement
    end

    def iteration
      @iteration
    end

    def iterationHistory
      @iterationHistory
    end

    def initialize
      @iteration = 0
      @iterationHistory = []
    end

    def record_learning_rate_history
      @iterationHistory[@iteration][:learningRate] = @learningRate
    end

    def record_learning_rate_adjustment_factor_history
      @iterationHistory[@iteration][:learningRateAdjustmentFactor] = @learningRateAdjustmentFactor
    end
    
    def load_dataset(filePath, addBias = true)
      # Initialize the inputs as an array for easy concatenation.
      inputs = Array.new
      
      # Initialize the outputs as an arrray for easy concatenation.
      @outputs = Array.new
      
      # Initialize with zero observations.
      @observationCount = 0
      
      # Open the file and interate through each parsed row...
      CSV.foreach(filePath) do  |row|
        
        # Convert each element to a Fixnum from String.
        row.map! {
          |element|
          
          element.to_f
        }
        
        # Use every element except the last as inputs, and add the bias 
        # term.

        inputs += row[0..row.length - 2]
        if addBias == true
          inputs += [1]
          # puts "Added row #{row[0..row.length - 2] + [1]}"
        else
          # puts "Added row #{row[0..row.length - 2]}"
        end
        
        # Use the last element as the output.
        @outputs += [row[row.length - 1]]
        # puts "Added output #{[row[row.length - 1]]}"
        
        # Increment the observation count.
        @observationCount += 1    
      end
      
      # Set the input count.
      @inputCount = inputs.length / @observationCount
      # puts "@inputCount = #{@inputCount}"
      
      # Set the output count.
      # While not strictly necessary, this will make it easier to 
      # incorporate more outputs in the future.
      @outputCount = @outputs.length / @observationCount
      # puts "@outputCount = #{@outputCount}"

      # puts "@observationCount = #{@observationCount}"
      
      # Translate the inputs array into a matrix.
      # Set up the empty matrix of zeros.
      @inputs = Matrix.new(@observationCount, @inputCount)
      # Assign each element explicitly.
      inputs.each_index do  |i|
        targetRow = (i / @inputCount).floor
        targetColumn = i % @inputCount
        element = inputs[i]
        @inputs.set(targetRow, targetColumn, element)
      end
      
      # Translate the outputs array into a matrix.
      @outputs = Matrix.new(@outputs.to_java(:double), @observationCount)

      # print_input_matrix(true)

      # print_output_matrix(true)
    end
    
    def weights
      @weights
    end
    
    def initialize_weights
      randomWeights = []
      
      @inputCount.times{
        randomWeights += [rand * 2 - 1]
      }

      @weights = Matrix.new(
                  randomWeights.to_java(:double),
                  @inputCount
                )
    end
    
    def update_weights
      # This portion might be repeated for each output when multiple outputs are implemented.

      # Calculate the errors.
      errors = @predictions.minus(@outputs)

      update_error(errors)

      check_learning_rate

      # print_error(true)

      print_error_improvement(true)

      # Expand the errors out to have a vector for each input.
      # This matrix is confirmed correct.
      errorsMatrix = errors.times(
                      Matrix.new(
                        @outputCount,
                        @inputCount,
                        1
                      )
                    )
      # BatchMultipleRegression.print_matrix(errorsMatrix, 'errorsMatrix', 3, ', ')

      # Create the input error matrix.
      # This matrix is confirmed correct.
      inputErrorGradient = errorsMatrix.arrayTimes(@inputs)
      # BatchMultipleRegression.print_matrix(inputErrorGradient, 'inputErrorGradient', 3, ', ')

      # Create the input-error, weight gradient.
      # This matrix is confirmed correct.
      inputWeightGradient = Matrix.new(
                              @outputCount,
                              @observationCount,
                              1
                            ).times(inputErrorGradient)
      # BatchMultipleRegression.print_matrix(inputWeightGradient, 'inputWeightGradient', 3, ', ')

      # Average the errors.
      # This matrix is confirmed correct.
      inputWeightGradient = inputWeightGradient.times(1.0 / @observationCount)
      # BatchMultipleRegression.print_matrix(inputWeightGradient, 'inputWeightGradient', 3, ', ')

      # Rotate the matrix before subtraction.
      inputWeightGradient = inputWeightGradient.transpose


      # Calculate the regularization gradient by squaring the weights.
      regularizationGradient = @weights.arrayTimes(@weights)

      # Multiply the square weights by the regularization factor.
      regularizationGradient.timesEquals(@regularization)

      # Set the last row of regularization to zero, because the bias element should not be penalized.
      lastRowID = regularizationGradient.getRowDimension - 1
      newLastRow = Matrix.new(
        ([0] * @outputCount).to_java(:double), 
        @outputCount
      )
      regularizationGradient.setMatrix(
        lastRowID,
        lastRowID,
        0,
        @outputCount - 1,
        newLastRow
      )

      # Average the regularization gradient.
      regularizationGradient.timesEquals(1.0 / @observationCount)

      # Add the regularization gradient.
      inputWeightGradient = inputWeightGradient.plus(regularizationGradient)

      # Reduce the gradient based on the learning rate.
      inputWeightGradient.timesEquals(@learningRate)

      # Update the weights.
      @weights = @weights.minus(inputWeightGradient)
    end
    
    def learn_weights(endingValue, valueType)
      if valueType.downcase == 'iterations'
        endingValue.times do
          @iteration += 1
          @iterationHistory[@iteration] = {}
          calculate_predictions
          update_weights
        end
      end

      if valueType.downcase == 'errorimprovement'
        6.times do
          @iteration += 1
          @iterationHistory[@iteration] = {}
          calculate_predictions
          update_weights
        end
        while average_error_change(5) > endingValue or @errorImprovement < 0
          @iteration += 1
          @iterationHistory[@iteration] = {}
          calculate_predictions
          update_weights
        end
      end
    end

    def calculate_predictions
      @predictions = @inputs.times(@weights)
    end
    
    def write_final_model
    
    end
    
    def train_model
    
    end

    # Print the input matrix for confirmation of correct loading.
    def print_input_matrix(useMatrixLabel = false)
      if useMatrixLabel == true then label = 'inputs' end
      BatchMultipleRegression.print_matrix(@inputs, label, 3, ', ')
    end

    # Print the output matrix for confirmation of correct loading.
    def print_output_matrix(useMatrixLabel = false)
      if useMatrixLabel == true then label = 'outputs' end
      BatchMultipleRegression.print_matrix(@outputs, label, 3, ', ')
    end

    # Print out the current iteration.
    def print_iteration(useLabel = false)
      if useLabel == true then label = 'iteration: ' end
      puts "#{label}#{@iteration.to_s}"
    end

    # Calculate the error from a matrix of errors.
    def update_error(errorMatrix)
      previousError = @error
      currentError = 0

      # Get the sum of squared errors.
      errorMatrix.getRowPackedCopy.to_a.each do  |element|
        currentError += element ** 2
      end

      # Get the average squared error per observation.
      currentError /= (2 * @observationCount)

      # Update the instance variable.
      @error = currentError

      # Calculate the error change.
      if previousError
        @errorImprovement = previousError - currentError
      end

      record_error_history
      record_error_improvement_history
    end

    # Print the current error.
    def print_error(useLabel = false)
      message = @error.to_s
      label = "error: "
      if useLabel == true
        message = label + message
      end
      puts message
    end

    # Print the error change.
    def print_error_improvement(useLabel = false)
      message = @errorImprovement.to_s
      label = "error change: "
      if useLabel == true
        message = label + message
      end
      puts message
    end

    # Write the weights file.
    def write_weights_file(fileName = 'weights.csv')
      file = CSV.open(fileName, 'wb')
      elements = @weights.getRowPackedCopy.to_a 
      rows = @weights.getRowDimension
      columns = @weights.getColumnDimension

      rows.times do  |row|
        startIndex = row * columns
        endIndex = (row + 1) * columns - 1
        rowElements = elements[startIndex..endIndex]
        file << rowElements
      end
    end

    # Write the final predictions of the model.
    def write_predictions_file(fileName = 'predictions.csv')
      file = CSV.open(fileName, 'wb')
      elements = @predictions.getRowPackedCopy.to_a 
      rows = @predictions.getRowDimension
      columns = @predictions.getColumnDimension

      rows.times do  |row|
        startIndex = row * columns
        endIndex = (row + 1) * columns - 1
        rowElements = elements[startIndex..endIndex]
        file << rowElements
      end
    end

    def check_learning_rate
      if @errorImprovement
        if @errorImprovement < 0
          @learningRate *= @learningRateAdjustmentFactor
          #puts "new learning rate: #{@learningRate}"
        else
          @learningRate /= @learningRateAdjustmentFactor
          @learningRateAdjustmentFactor = (1 - @learningRateAdjustmentFactor) / 2
          if @iteration > 4 and @iterationHistory[@iteration - 1][:errorImprovement] > 0
            @learningRate += @learningRate - @learningRate * @learningRateAdjustmentFactor
          else
            @learningRate *= @learningRateAdjustmentFactor
          end
        #   #puts "new learning rate: #{@learningRate}"
        end
      end
    end

    def average_error_change(iterations)
        if @iteration > iterations
          average = 0
          iterations.times do  |i|
            average += @iterationHistory[@iteration - i][:errorImprovement].abs
          end
          average /= iterations
        end
    end

    def write_iteration_history_table_file(fileName = 'history.csv', columns = 'all')
      file = CSV.open(fileName, 'wb')
      @iterationHistory.compact!
      file << @iterationHistory.last.keys
      if columns == 'all'
        @iterationHistory.each do  |iteration|
          file << iteration.values
        end
      else
        columns = columns.map{|c| "\"c\""}.join(',')
        @iterationHistory.each_index do  |i|
          file << iteration.values_at(columns)
        end
      end
    end
  end
end

if __FILE__ == $0
  # Determine the dataset from the first command-line argument.
  datasetFilePath = ARGV[0]

  # If a dataset was not supplied, then default to the file data.csv.
  datasetFilePath ||= 'data.csv'

  # Create the program.
  defaultProgram = BatchMultipleRegression::TrainingProgram.new

  # Load the dataset.
  defaultProgram.load_dataset(datasetFilePath)

  # Initialize with the default learning rate of 0.1.
  defaultProgram.learningRate = 0.1

  # Initialize with the default regularization of 0.
  defaultProgram.regularization = 0

  # Set the default learning rate adjustment factor of 0.5.
  defaultProgram.learningRateAdjustmentFactor = 0.5

  # Initialize the weights.
  defaultProgram.initialize_weights

  # Learn the optimal weights.
  # defaultProgram.learn_weights(1000, 'iterations')
  defaultProgram.learn_weights(1e-06, 'errorImprovement')

  # Print the weights.
  #defaultProgram.weights.print(6, 3)

  # Print the final error.
  defaultProgram.print_error(true)

  # Print the final iteration.
  defaultProgram.print_iteration(true)

  # Write the iteration history file.
  defaultProgram.write_iteration_history_table_file()
  # puts defaultProgram.iterationHistory

  # Write the weights file.
  defaultProgram.write_weights_file

  # Write the predictions file.
  defaultProgram.write_predictions_file
end
