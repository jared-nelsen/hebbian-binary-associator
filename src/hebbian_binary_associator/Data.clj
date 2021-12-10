
(ns hebbian-binary-associator.Data
  (:require [hebbian-binary-associator.NeuralNetwork :as NeuralNetwork]))

;; Defined and used during training by genetic algorithm
;; temorarily defined here for now
(def intermediateIndecesVector [1 2 3])

(def defaultDataConfig {:ioPairCount 10})

(defrecord Data [inputDataVector outputDataVector])

(defn vectorOfRandomIntegers
  "Generates a vector of random positive integers."
  [count]
  (loop [c 0
         newInts []]
    (if (= c count)
      newInts
      (recur (inc c) (conj newInts (rand-int (Integer/MAX_VALUE)))))))

(defn generateRandomData
  []
  (Data. (vectorOfRandomIntegers (:ioPairCount defaultDataConfig))
         (vectorOfRandomIntegers (:ioPairCount defaultDataConfig))))

(defn trainOnData
  [network data]
  (let [network (NeuralNetwork/loadActiveIntermediaryIndecesIntoNetwork network
                                                                        intermediateIndecesVector)]
    (loop [trainedNetwork network
           inputDataVector (:inputDataVector data)
           outputDataVector (:outputDataVector data)]
      (if (empty? inputDataVector)
        trainedNetwork
        (let [inputData (first inputDataVector)
              outputData (first outputDataVector)
              alteredNetwork (NeuralNetwork/train trainedNetwork inputData outputData)]
          (recur alteredNetwork (rest inputDataVector) (rest outputDataVector)))))))

(defn validate
  "Runs the input data against the output data and returns the number of
   times that the output did not match the output data."
  [network data]
  (loop [errorCount 0
         inputDataVector (:inputDataVector data)
         outputDataVector (:outputDataVector data)]
    (if (empty? inputDataVector)
      errorCount
      (let [inputData (first inputDataVector)
            outputData (first outputDataVector)
            networkOutput (NeuralNetwork/feedForwardNetwork network inputData)]
        (if (not= outputData networkOutput)
          (recur (inc errorCount) (rest inputDataVector) (rest outputDataVector))
          (recur errorCount (rest inputDataVector) (rest outputDataVector)))))))

