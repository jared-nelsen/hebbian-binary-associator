
(ns hebbian-binary-associator.NeuralNetwork)

(defrecord Neuron [id incomingConnections threshold activated])
(defrecord Connection [id sourceNeuronId targetNeuronId weight])
(defrecord NeuralNetwork [neuronMap connectionMap
                          inputNeuronIdSet
                          intermediaryNeuronIdSet activeIntermediaryIndeces
                          outputNeuronIdSet])

(def defaultNetworkConfig {:dataWidth 32
                           :intermediateNeuronCount 100
                           :startingActiveIntermediateNeuronCount 5
                           :defaultConnectionWeight 0.0
                           :hebbsRuleIncrementValue 0.05
                           :trainingMode 1})

;; Neurons

(defn getNeuronIds
  "Collects the ids of the given neurons into a vector in order."
  [neurons]
  (loop [neurons neurons
         keys []]
    (if (empty? neurons)
      keys
      (recur (rest neurons) (conj keys (:id (first neurons)))))))

(defn generateNeuron
  "Generates a Neuron."
  []
  (Neuron. (.toString (java.util.UUID/randomUUID)) [] 0.5 false))

(defn setNeuronActiveInNeuronMap
  "Sets the given neuron to activated in the given neuron map."
  [neuronMap neuronId]
  (let [neuron (get neuronMap neuronId)
        activatedNeuron (assoc neuron :activated true)]
    (assoc neuronMap neuronId activatedNeuron)))

(defn generateNNeurons
  "Generates N neurons in a vector."
  [N]
  (loop [count 0
         neurons []]
    (if (= count N)
      neurons
      (recur (inc count) (conj neurons (generateNeuron))))))

(defn getAllNeuronKeysInNetwork
  "Returns all the neuron keys in the network in order of the input -> intermediary -> output key sets."
  [network]
  (vec (concat (:inputNeuronIdSet network) (:intermediaryNeuronIdSet network) (:outputNeuronIdSet network))))

(defn getAllNeuronsInNetworkInOrder
  "Returns all neurons in the network in order of the input -> intermediary -> output key sets."
  [network]
  (let [neuronMap (:neuronMap network)]
    (loop [allKeys (getAllNeuronKeysInNetwork network)
           neurons []]
      (if (empty? allKeys)
        neurons
        (recur (rest allKeys) (conj neurons (get neuronMap (first allKeys))))))))

(defn copyNeuronVector
  "Copies the given vector of Neurons"
  [neurons]
  (loop [neurons neurons
         neuronCopies []]
    (if (empty? neurons)
      neuronCopies
      (let [n (first neurons)
            nCopy (Neuron. (:id n) (:incomingConnections n) (:threshold n) (:activated n))]
        (recur (rest neurons) (conj neuronCopies nCopy))))))

;; Neural Network

(defn generateNeuronMap
  "Generates the Neuron Map construct given all a set of neurons."
  [neurons]
  (loop [neuronMap {}
         neurons neurons]
    (if (empty? neurons)
      neuronMap
      (let [neuron (first neurons)
            neuronId (:id neuron)
            updatedNeuronMap (assoc neuronMap neuronId neuron)]
        (recur updatedNeuronMap (rest neurons))))))

(defn isIONeuron
  "Detects if the given neuron is an input or output neuron."
  [IONeuronIds neuronId]
  (not (nil? (some #{neuronId} IONeuronIds))))

(defn connectNeuronToOthers
  "Connects the given Neuron to all Neurons in the given other list and returns the connections
   in a new ConnectionMap. Avoids making two connections between neurons by keeping a set of
   Connections that have been made with swapped source and target neurons."
  [neuron otherNeurons currentConnectionMap IONeuronIds]
  (loop [otherNeurons otherNeurons
         newConnMap {}]
    (if (empty? otherNeurons)
      newConnMap
      (let [currentId (:id neuron)
            otherId (:id (first otherNeurons))
            newConnId (.toString (java.util.UUID/randomUUID))
            newConnection (Connection. newConnId currentId
                                       otherId (:defaultConnectionWeight defaultNetworkConfig))
            existingSourceNeuronIds (map #(:sourceNeuronId %) (vals currentConnectionMap))]
        (if (or (some #{otherId} existingSourceNeuronIds)
                (and (isIONeuron IONeuronIds currentId)
                     (isIONeuron IONeuronIds otherId)))
          (recur (rest otherNeurons) newConnMap)
          (let [updatedConnMap (assoc newConnMap newConnId newConnection)]
            (recur (rest otherNeurons)
                   updatedConnMap)))))))

(defn generateConnectionMap
  "Fully connects the given Neural Network and returns the connections in connection map form."
  [allNeurons IONeuronIds]
  (loop [allNeurons allNeurons
         allNeuronsCopy (copyNeuronVector allNeurons)
         connectionMap {}]
    (if (empty? allNeurons)
      connectionMap
      (let [currentNeuron (first allNeurons)
            otherNeurons (filter (fn [neuronCopy]
                                   (not= (:id neuronCopy) (:id currentNeuron)))
                                 allNeuronsCopy)
            newConnectionMapEntries (connectNeuronToOthers currentNeuron otherNeurons
                                                           connectionMap IONeuronIds)
            updatedConnectionMap (merge connectionMap newConnectionMapEntries)]
        (recur (rest allNeurons) allNeuronsCopy updatedConnectionMap)))))

(defn collectConnectionIdsIncomingToNeuron
  [neuronId connectionMap]
  (loop [connections (vals connectionMap)
         incomingConnIds []]
    (if (empty? connections)
      incomingConnIds
      (let [currentConn (first connections)
            currentConnId (:id currentConn)
            currentConnTarget (:targetNeuronId currentConn)]
        (if (= currentConnTarget neuronId)
          (recur (rest connections) (conj incomingConnIds currentConnId))
          (recur (rest connections) incomingConnIds))))))

(defn loadIncomingConnectionsIntoNeuronMap
  "When the Connections are made they are not added to the Neurons list of incoming neurons.
   This function does that."
  [neuronMap connectionMap]
  (loop [neurons (vals neuronMap)
         newNeuronMap {}]
      (if (empty? neurons)
        newNeuronMap
        (let [neuron (first neurons)
              neuronId (:id neuron)
              incomingConnIds (collectConnectionIdsIncomingToNeuron neuronId connectionMap)
              newNeuron (Neuron. neuronId incomingConnIds (:threshold neuron) (:activated neuron))
              updatedNewNeuronMap (assoc newNeuronMap neuronId newNeuron)]
          (recur (rest neurons) updatedNewNeuronMap)))))

(defn generateNeuralNetwork
  "Generates a Neural Network according to the given config."
  []
  (let [inputNeurons (generateNNeurons (:dataWidth defaultNetworkConfig))
        inputNeuronIds (getNeuronIds inputNeurons)
        intermediateNeurons (generateNNeurons (:intermediateNeuronCount defaultNetworkConfig))
        intermediateNeuronIds (getNeuronIds intermediateNeurons)
        outputNeurons (generateNNeurons (:dataWidth defaultNetworkConfig))
        outputNeuronIds (getNeuronIds outputNeurons)
        allNeurons (vec (concat [] inputNeurons intermediateNeurons outputNeurons))
        neuronMap (generateNeuronMap allNeurons)
        IONeuronIds (vec (concat [] inputNeuronIds outputNeuronIds))
        connectionMap (generateConnectionMap allNeurons IONeuronIds)
        neuronMap (loadIncomingConnectionsIntoNeuronMap neuronMap connectionMap)
        activeIndeces []]
    (NeuralNetwork. neuronMap connectionMap inputNeuronIds intermediateNeuronIds activeIndeces outputNeuronIds)))

;; Input, Output, and Activity

(defn resetActivationsInNetwork
  [network]
  (loop [neuronMap (:neuronMap network)
         allNeuronKeys (getAllNeuronKeysInNetwork network)]
    (if (empty? allNeuronKeys)
      (assoc network :neuronMap neuronMap)
      (let [currentId (first allNeuronKeys)
            currentNeuron (get neuronMap currentId)
            resetNeuron (assoc currentNeuron :activated false)
            updatedNeuronMap (assoc neuronMap currentId resetNeuron)]
        (recur updatedNeuronMap (rest allNeuronKeys))))))

(defn convertIntegerIntoNeuronActivityVector
  "Activity Vectors are vectors of binary integers representing on/off states. This function converts Integer
   inputs into its Activity Vector representation."
  [data]
  (let [binStr (Long/toBinaryString data)
        binStr (clojure.string/replace (format "%32s" binStr) \space \0)
        binVector (clojure.string/split binStr #"")
        activityVector (map #(Integer/parseInt %) binVector)]
    activityVector))

(defn impressActivityVectorIntoNetwork
  "Sets the activity of the target neurons in the network based on the activity vector passed in."
  [network targetNeuronIds activityVector]
  (loop [neuronMap (:neuronMap network)
         targetNeuronIdSet targetNeuronIds
         activityVector activityVector]
      (if (empty? targetNeuronIdSet)
        (assoc network :neuronMap neuronMap)
        (let [currentNeuronId (first targetNeuronIdSet)
              currentActivityIndicator (first activityVector)]
          (if (= 1 currentActivityIndicator)
            (recur (setNeuronActiveInNeuronMap neuronMap currentNeuronId) (rest targetNeuronIdSet) (rest activityVector))
            (recur neuronMap (rest targetNeuronIdSet) (rest activityVector)))))))

(defn loadInputIntoNetwork
  "Loads input data into the network by setting activations for the input neurons. Currently data are 32 bit Integers."
  [network input]
  (let [inputNeuronIds (:inputNeuronIdSet network)
        inputActivityVector (convertIntegerIntoNeuronActivityVector input)]
    (impressActivityVectorIntoNetwork network inputNeuronIds inputActivityVector)))

(defn loadOutputIntoNetwork
  "Loads output data into the network during training by setting activations for the ouput neurons.
   Currently data are 32 bit integers."
  [network output]
  (let [outputNeuronIds (:outputNeuronIdSet network)
        outputActivityVector (convertIntegerIntoNeuronActivityVector output)]
    (impressActivityVectorIntoNetwork network outputNeuronIds outputActivityVector)))

(defn gatherOutputOfNetwork
  "Loads and returns a vector of binary integers that represent the activations of the given network."
  [network]
  (loop [outputNeuronIdSet (:outputNeuronIdSet network)
         activationVector []]
    (if (empty? outputNeuronIdSet)
      activationVector
      (let [currentId (first outputNeuronIdSet)
            currentNeuron (get (:neuronMap network) currentId)
            currentNeuronActivated? (:activated currentNeuron)]
        (if (= true currentNeuronActivated?)
          (recur (rest outputNeuronIdSet) (conj activationVector 1))
          (recur (rest outputNeuronIdSet) (conj activationVector 0)))))))

(defn retrieveOutputFromNetwork
  "Retrieves the output from the given neural network's output neuron in Integer form."
  [network]
  (let [outputActivationIntegerVector (gatherOutputOfNetwork network)
        outputActivationString (clojure.string/join outputActivationIntegerVector)]
    (Integer/parseInt outputActivationString 2)))

;; Intermediary Activation

(defn loadActiveIntermediaryIndecesIntoNetwork
  "Loads a new vector of intermediary indeces into the given network."
  [network indeces]
  (assoc network :activeIntermediaryIndeces indeces))

(defn impressIntermediaryNeuronActivityIntoNetwork
  "Activates the neurons in the network according to intermediary activity pattern. This pattern
   is loaded into the network prior to this step. The indeces in the vector indicate the position
   of the neuron Id in the intermediary neuron id set whose associated neuron should be activated."
  [network]
  (loop [neuronMap (:neuronMap network)
         intermediaryNeuronIdSet (:intermediaryNeuronIdSet network)
         activeNeuronIndeces (:activeIntermediaryIndeces network)]
    (if (empty? activeNeuronIndeces)
      (assoc network :neuronMap neuronMap)
      (let [currentIndex (first activeNeuronIndeces)
            currentNeuronId (nth intermediaryNeuronIdSet currentIndex)
            updatedNeuronMap (setNeuronActiveInNeuronMap neuronMap currentNeuronId)]
        (recur updatedNeuronMap intermediaryNeuronIdSet (rest activeNeuronIndeces))))))

;; Running

(defn sumIncomingActivations
  "Sums the incoming activations to the given neuron. Activations are defined as the source neuron
   of the given incoming connection's activation state + the weight of that connection."
  [neuron neuronMap connectionMap]
  (loop [activationSum 0.0
         incomingConnections (:incomingConnections neuron)]
    (if (empty? incomingConnections)
      activationSum
      (let [connectionId (first incomingConnections)
            incomingConnection (get connectionMap connectionId)
            connectionWeight (:weight incomingConnection)
            sourceNeuronId (:sourceNeuronId incomingConnection)
            sourceNeuron (get neuronMap sourceNeuronId)
            sourceNeuronActivated? (:activated sourceNeuron)]
        (if (= true sourceNeuronActivated?)
          (recur (+ activationSum connectionWeight) (rest incomingConnections))
          (recur activationSum (rest incomingConnections)))))))

(defn attemptToActivateNeuron
  "Attempts to activate the given neuron using the given connection map and neuron map."
  [neuron neuronMap connectionMap]
  (let [neuronFiringThreshold (:threshold neuron)
        actionPotential (sumIncomingActivations neuron neuronMap connectionMap)]
    (if (>= actionPotential neuronFiringThreshold)
      (assoc neuron :activated true)
      neuron)))

(defn runNetwork
  "Runs the given neural network."
  [network]
  (let [connectionMap (:connectionMap network)]
    (loop [neuronMap (:neuronMap network)
           neuronIds (getAllNeuronKeysInNetwork network)]
    (if (empty? neuronIds)
      (assoc network :neuronMap neuronMap)
      (let [currentId (first neuronIds)
            currentNeuron (get neuronMap currentId)
            updatedNeuron (attemptToActivateNeuron currentNeuron neuronMap connectionMap)
            updatedNeuronMap (assoc neuronMap currentId updatedNeuron)]
        (recur updatedNeuronMap (rest neuronIds)))))))

(defn feedForwardNetwork
  "Takes in the data, applies it to the network, runs the network, and returns the output."
  [network data]
  (let [network (resetActivationsInNetwork network)
        network (loadInputIntoNetwork network data)
        network (runNetwork network)
        output (retrieveOutputFromNetwork network)]
    output))

;; Training

;; There are two training methods to exeriment with:
;; 1. applyHebbsRuleFully - Sets the weights immediately to firing levels. This will focus the training
;;                          phase onto the configuration of firing neuron indeces that the GA is
;;                          responsible for.
;; 2. applyHebbsRuleIncrementally - Bumps the weights by a tiny fraction each time Hebbs rule is run.
;;                                  This will cause the training to focus on both the active indece
;;                                  configuration and careful weight change.

(defn applyHebbsRule
  "Hebbs rule states that neurons that fire together, wire together. Go through all of the connections
   and if both source and target neuron are active then set the weight to be above the firing
   threshold."
  [network bumpValue]
  (loop [connections (vals (:connectionMap network))
         newConnectionMap {}]
    (if (empty? connections)
      (assoc network :connectionMap newConnectionMap)
      (let [connection (first connections)
            connId (:id connection)
            neuronMap (:neuronMap network)
            sourceNeuronId (:sourceNeuronId connection)
            sourceNeuron (get neuronMap sourceNeuronId)
            targetNeuronId (:targetNeuronId connection)
            targetNeuron (get neuronMap targetNeuronId)]
        (if (and (:activated sourceNeuron) (:activated targetNeuron))
          (let [weight (:weight connection)
                bumpedWeight (+ bumpValue weight)
                updatedConnection (assoc connection :weight bumpedWeight)]
            (recur (rest connections) (assoc newConnectionMap connId updatedConnection)))
          (recur (rest connections) (assoc newConnectionMap connId connection)))))))

(defn applyHebbsRuleFully
  "Applys Hebb's Rule so that the firing threshold is immediately surpassed."
  [network]
  (applyHebbsRule network 1.0))

(defn applyHebbsRuleIncrementally
  "Applys Hebb's Rule by the value specified in the network config."
  [network]
  (applyHebbsRule network (:hebbsRuleIncrementValue defaultNetworkConfig)))

(defn trainOnePass
  "Trains the network by applying Hebbs rule in one pass."
  [network input output]
  (let [network (loadInputIntoNetwork network input)
        network (impressIntermediaryNeuronActivityIntoNetwork network)
        network (loadOutputIntoNetwork network output)]
    (applyHebbsRuleFully network)))

(defn trainIncrementally
  "Trains the network by applying Hebbs rule incrementally."
  [network input output]
  (let [network (loadInputIntoNetwork network input)
        network (impressIntermediaryNeuronActivityIntoNetwork network)
        network (loadOutputIntoNetwork network output)]
    (applyHebbsRuleIncrementally network)))

(defn train
  "Trains the given network in the mode specified in the default
   network config."
  [network input output]
  (if (= 1 (:trainingMode defaultNetworkConfig))
    (trainOnePass network input output)
    (trainIncrementally network input output)))
