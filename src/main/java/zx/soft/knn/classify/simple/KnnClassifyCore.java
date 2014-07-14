package zx.soft.knn.classify.simple;

import java.util.Arrays;

/**
 * KNN算法的核心类。
 *     分类的时候：假设有0、1标签
 *     回归的时候：假设真实的标签分布在0～1之间
 * 
 * @author wanggang
 *
 */
public class KnnClassifyCore {

	// 数据集
	private final InstanceSet instanceSet;
	// k值
	private int[] k;
	// 核宽度
	private int[] kernel;
	// 训练集大小
	private final int trainSetSize;
	// 测试集大小
	private final int testSetSize;
	// 训练集
	public final KnnInstance[] trainSet;
	// 测试集
	public final KnnInstance[] testSet;
	// 记录正确分类的实例
	private int[][] numCorrect;
	// 记录平方差
	private double[][] squaredError;
	// 预测结果
	private double[][][] predictions;

	/**
	 * 初始化训练集和测试集，并且训练集和测试集分别来自不同的文件
	**/
	public KnnClassifyCore(InstanceSet instanceSet, int trainSetSize) {
		this.instanceSet = instanceSet;

		// 训练集
		this.trainSetSize = trainSetSize;
		trainSet = new KnnInstance[trainSetSize];
		for (int i = 0; i < trainSetSize; i++)
			trainSet[i] = new KnnInstance(instanceSet.getInstances(i), i);

		// 测试集
		this.testSetSize = instanceSet.getSize() - trainSetSize;
		testSet = new KnnInstance[testSetSize];
		int index = 0;
		for (int i = trainSetSize; i < instanceSet.getSize(); i++) {
			testSet[index] = new KnnInstance(instanceSet.getInstances(i), i);
			index++;
		}

		// 初始化特征权重
		scaleFeatureWeights(1);
	}

	/**
	 * 使用k[]和kernel[]进行训练；使用LOOCV选择最佳k值，以及核宽度；
	 * 假设k[]和kernelWidth[]是有序的，那么返回最佳k值以及核宽度
	 */
	public int[] train(int k[], int kernel[]) {
		numCorrect = new int[kernel.length][k.length];
		squaredError = new double[kernel.length][k.length];
		predictions = new double[trainSetSize][kernel.length][k.length];

		this.k = k;
		this.kernel = kernel;

		// 在训练集上反复测试每个查询实例，更新正确分类的实例以及它们的均方误差
		KnnInstance queryInstance;
		KnnInstance[] subTrainSet = new KnnInstance[trainSetSize - 1];
		int index = 0;
		for (int i = 0; i < trainSetSize; i++) {
			index = 0;
			queryInstance = trainSet[i];
			// 在每个查询实例上，建立子训练集用于测试
			for (int j = 0; j < trainSetSize; j++) {
				if (!queryInstance.equals(trainSet[j])) {
					subTrainSet[index] = trainSet[j];
					index++;
				}
			}
			// 在子训练集上测试查询实例
			predictions[i] = testSingle(queryInstance, subTrainSet);
			// 用最小核宽核最小k打印出结果
			if (i % 25 == 0)
				printSimple(i);
		}

		// 打印最终结果
		printResults(trainSetSize - 1);
		System.out.println("-------------------------------------------------------");
		//printPredictions();

		return getBestValues();
	}

	/**
	 * Test the test set with values with given values of k[] and kernel[].
	**/
	public void test(int[] bestK, int[] bestKernel) {
		// Initialize numCorrect and squaredError arrays
		numCorrect = new int[bestKernel.length][bestK.length];
		squaredError = new double[bestKernel.length][bestK.length];
		predictions = new double[testSetSize][bestKernel.length][bestK.length];

		// Set up the number of nearest neighbors and the kernel widths
		this.k = bestK;
		this.kernel = bestKernel;

		// Evaluate each example from the test set onto the training set
		KnnInstance testExample;
		for (int i = 0; i < testSetSize; i++) {
			testExample = testSet[i]; // see about evaluation set if makes more sense
			predictions[i] = testSingle(testExample, trainSet);
			if (i % 25 == 0)
				printSimple(i);
		}
		printResults(testSetSize - 1);
		System.out.println();
		System.out.println("-------------------------------------------------------");
		printPredictions();
	}

	/**
	 * Test the single example on a train set
	 * Updates the squared error and the number of correctly classified examples
	**/
	private double[][] testSingle(Instance testExample, KnnInstance[] trainSet) {
		// Sort the train set to get the nearest neighbors
		// first set the distance each example is from the test example
		for (int i = 0; i < trainSet.length; i++) {
			trainSet[i].setRelativeDist(testExample);
		}
		// only sort if the maximum of neighbors < the size of the train set
		if (k[0] < trainSetSize - 1) {
			KnnComparator comparator = new KnnComparator();
			Arrays.sort(trainSet, comparator);
		}

		// Update the squared error and the number of correctly classified examples
		// for each kernel width and each number of nearest neighbors considered
		int targetLabel = testExample.getLabel();
		KnnInstance neighbor; // neighbor considered
		double neighborWeight; // its weight
		double neighborValue; // its value (ie class label)
		double sumWeightedValue; // weighted sum of each neighbor so far
		double sumAllWeight; // sum of the weights of each neighbor so far
		double probaLabel;
		double[][] predictions = new double[kernel.length][k.length];
		int kIndex; // indexes k[] array
		// for each kernel width from kernel[]
		for (int kernelIndex = 0; kernelIndex < kernel.length; kernelIndex++) {
			sumWeightedValue = 0;
			sumAllWeight = 0;
			kIndex = 0;
			// for each neighbor from 0 to max k[]
			for (int neighborNo = 0; neighborNo < k[k.length - 1]; neighborNo++) {
				// compute the probability of each neighbor
				neighbor = trainSet[neighborNo];
				neighbor.setWeight(kernel[kernelIndex]);
				neighborWeight = neighbor.getWeight();
				neighborValue = neighbor.getLabel();
				sumWeightedValue = sumWeightedValue + neighborValue * neighborWeight;
				sumAllWeight = sumAllWeight + neighborWeight;
				probaLabel = sumWeightedValue / sumAllWeight;
				// update results after having seen k[kIndex] - 1 neighbors
				if (neighborNo == k[kIndex] - 1) {
					if (Math.abs(targetLabel - probaLabel) <= 0.5) {
						(numCorrect[kernelIndex][kIndex])++;
					}
					squaredError[kernelIndex][kIndex] = squaredError[kernelIndex][kIndex]
							+ Math.pow(targetLabel - probaLabel, 2);
					predictions[kernelIndex][kIndex] = probaLabel;
					kIndex++;
				}
			}
		}
		return predictions;
	}

	/**
	 * Returns as first coordinate the best kernel.
	 * Returns as second coordinate the best number of nearest neighbors.
	 * Assumes train or test method has been called before.
	**/
	private int[] getBestValues() {
		double lowestError = 0;
		int[] bestValues = new int[2];
		for (int kernelIndex = 0; kernelIndex < kernel.length; kernelIndex++) {
			for (int kIndex = 0; kIndex < k.length; kIndex++) {
				if (squaredError[kernelIndex][kIndex] < lowestError) {
					lowestError = squaredError[kernelIndex][kIndex];
					bestValues[0] = kernel[kernelIndex];
					bestValues[1] = k[kIndex];
				}
			}
		}
		return bestValues;
	}

	/**
	 * Scale the feature weights by 1/(max-min) or 1/var depending on mode
	 * using a train set.
	**/
	private void scaleFeatureWeights(int mode) {
		double max, min, var, sum, sumSquared;
		int numAttributeVal = instanceSet.getAttributeNum();
		double[] attributeVal = new double[trainSetSize];
		double[] featureWeights = new double[numAttributeVal];
		double[] featureWeights1 = new double[numAttributeVal];
		double[] featureWeights2 = new double[numAttributeVal];

		// Find max min and variance of each attribute value
		for (int i = 0; i < numAttributeVal; i++) {
			sum = 0;
			sumSquared = 0;
			for (int j = 0; j < trainSetSize; j++) {
				attributeVal[j] = instanceSet.getAttribute(j, i);
				sum = sum + attributeVal[j];
				sumSquared = sumSquared + Math.pow(attributeVal[j], 2);
			}
			Arrays.sort(attributeVal);
			max = attributeVal[trainSetSize - 1];
			min = attributeVal[0];
			var = sumSquared / trainSetSize - Math.pow(sum / trainSetSize, 2);
			featureWeights[i] = 1;
			if (max - min == 0)
				featureWeights1[i] = 1;
			else
				featureWeights1[i] = 1 / Math.exp(max - min);
			if (var == 0)
				featureWeights2[i] = 1;
			else
				featureWeights2[i] = 1 / var;
		}

		// Default feature weights to 1
		if (mode == 0)
			KnnInstance.setFeatureWeights(featureWeights);
		// 1/(max-min)
		else if (mode == 1)
			KnnInstance.setFeatureWeights(featureWeights1);
		// 1/var
		else if (mode == 2)
			KnnInstance.setFeatureWeights(featureWeights2);
	}

	/**
	 * Print the RMSE and accuracy after having seen a number of examples.
	**/
	private void printResults(int iteration) {
		System.out.println();
		System.out.println("Iteration =  " + iteration);
		double[] accuracy = new double[k.length];
		double[] RMSE = new double[k.length];
		//		double[] predictions;
		//		double accuracy1;
		//		double RMSE1;
		for (int kernelIndex = 0; kernelIndex < kernel.length; kernelIndex++) {
			System.out.println("Kernel Width =  " + kernel[kernelIndex]);
			System.out.println("------------------------------------------------");
			for (int kIndex = 0; kIndex < k.length; kIndex++) {
				accuracy[kIndex] = 100.0 * numCorrect[kernelIndex][kIndex] / (iteration + 1);
				RMSE[kIndex] = Math.sqrt(squaredError[kernelIndex][kIndex] / (iteration + 1));
				System.out.println("k =  " + k[kIndex] + " , Accuracy =  " + accuracy[kIndex] + "% , RMSE =  "
						+ RMSE[kIndex]);

				//        predictions=getPredictions(kernelIndex,kIndex);
				//        accuracy1=dataFile.returnAccuracy(predictions);
				//        RMSE1=dataFile.returnRMSE(predictions);
				//        System.out.println("Other k =  " + k[kIndex] + " , Accuracy =  " + accuracy1 + "% , RMSE =  " + RMSE1);
			}
		}
		// used for experiments...
		// KnnClassifyCore experiments
		System.out.print(trainSetSize + " ");

		for (int kIndex = 0; kIndex < k.length; kIndex++) {
			accuracy[kIndex] = 100.0 * numCorrect[0][kIndex] / (iteration + 1);
			RMSE[kIndex] = Math.sqrt(squaredError[0][kIndex] / (iteration + 1));
			System.out.print(accuracy[kIndex] + " " + RMSE[kIndex] + " ");
		}

		// locally weighted experiments
		//    for (int kernelIndex=0; kernelIndex < kernel.length; kernelIndex++)
		//    {
		//      accuracy[0] = 100.0 * numCorrect[kernelIndex][0]/(iteration+1);
		//      RMSE[0] = Math.sqrt(squaredError[kernelIndex][0]/(iteration+1));
		//      System.out.print(accuracy[0] + " " + RMSE[0] + " ");
		//    }
	}

	/**
	 * Modeified...
	 *
	 * Return predictions
	**/
	public double[] getPredictions(int kernel, int k) {
		double[] pred = new double[testSetSize];
		for (int i = 0; i < testSetSize; i++)
			pred[i] = this.predictions[i][kernel][k];

		return pred;
	}

	public void printPredictions() {
		System.out.println("*********************Predictions**************************");
		for (int i = 0; i < predictions.length; i++) {
			for (int j = 0; j < k.length; j++)
				System.out.print(predictions[i][0][j] + " ");
			System.out.print("\n");
		}
	}

	/**
	 * Print the 1-nearest neighbor results.
	 * Used to see how fast the algorithm performs.
	 *
	 * Need to get predictions on the test set need to rethink all that...
	 *
	**/
	private void printSimple(int iteration) {
		double accuracy;
		double RMSE;
		int kIndex = 0;
		int kernelIndex = 0;

		System.out.println("Iteration =  " + iteration);
		accuracy = (100.0 * numCorrect[kernelIndex][kIndex]) / (iteration + 1);
		RMSE = Math.sqrt(squaredError[kernelIndex][kIndex] / (iteration + 1));
		System.out.println("k =  " + k[kIndex] + " , Accuracy =  " + accuracy + "% , RMSE =  " + RMSE);

		// not good here does
		//    double[] predictions=getPredictions(kernelIndex,kIndex);
		//    double accuracy1=dataFile.returnAccuracy(predictions);
		//    double RMSE1=dataFile.returnRMSE(predictions);
		//System.out.println("Other k =  " + k[kIndex] + " , Accuracy =  " + accuracy1 + "% , RMSE =  " + RMSE1);
	}

	public void prettyPrintParameters() {
		//
	}

	/**
	 * Report the baseline accuracy of the training set
	*/
	public void reportBaseline() {
		int[] count = new int[2];
		int bestCount = 0;
		int label;
		for (int i = 0; i < trainSetSize; i++) {
			label = trainSet[i].getLabel();
			count[label]++;
			if (count[label] > bestCount) {
				bestCount = count[label];
			}
		}
		System.out.println("Size of the training Set: " + trainSetSize);
		System.out.println("Baseline Accuracy of Training Set =  " + 100.0 * bestCount / trainSetSize + "%");
	}

}
