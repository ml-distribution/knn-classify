package zx.soft.knn.classify.simple;

public class KnnClassifyDemo {

	/**
	 * Mode is 0 for Unweighted KnnClassifyCore
	 *         1 for Weighted KnnClassifyCore
	 *         2 for Locally weighted Averaging
	 */
	public static void main(String[] args) {
		// 输入文件
		InstanceSet instanceSet = new InstanceSet("sample/train-data");
		// 训练集大小
		int trainSetSize = 4000;
		// 算法类型
		int mode = 1;
		// k值
		int kValue = 50;

		// Feature selection
		//			int numFeatures = 58;
		//			int[] features = new int[numFeatures];
		//      // ...with ANN
		//      System.out.println("Proceeding with feature selection...");
		//      InstanceSet featureSet=dataFile.cut(0,4000);
		//      features=featureSet.ANNFeatureSelector(numFeatures);
		// ...with DT
		//      int[] DTFeatures={1,15,18,24,42,5,6,56,44,29,45,58,31,
		//                       35,3,39,16,43,54,4,46,33,40,41,23,11,
		//                       30,57,7,20,28,55,49,47,13,52,9,51,19,
		//                       12,8,38,22,17,21,10,14,27,2,50,48,53,
		//                       32,26,25,34,36,37};
		//      for (int i=0;i<features.length;i++)
		//        features[i]=DTFeatures[i]-1;
		//      dataFile=dataFile.select(features);

		// Init the KnnClassifyCore algorithm by setting up the train and test set
		KnnClassifyCore knnClassifyCore = new KnnClassifyCore(instanceSet, trainSetSize);

		// Set up k[] and kernel[] based on the mode
		// we consider values of k from 1 to lastParam (note we could have considered smtg else)
		int k[] = new int[kValue];
		int kernel[] = new int[kValue + 1];
		for (int i = 0; i < kValue; i++)
			k[i] = i + 1;
		// we consider values of the kernel width from 1 to lastParam (note we could have considered smtg else)
		for (int i = 0; i <= kValue; i++)
			kernel[i] = i;
		// depending on mode we change some specs
		switch (mode) {
		// unweighted KnnClassifyCore
		case 0:
			kernel = new int[1];
			kernel[0] = 0;
			break;
		// weighted KnnClassifyCore
		case 1:
			kernel = new int[1];
			kernel[0] = 1;
			break;
		// locally weighted averaging
		case 2:
			k = new int[1];
			k[0] = trainSetSize - 1;
			break;
		}

		// Train with these values of k[] and kernel[]
		knnClassifyCore.reportBaseline();
		//			int[] bestValues = new int[2];
		//bestValues = mykNN.train(k, kernel);
		// Test with the best values of k[] and kernel[]
		int[] bestKernel = new int[1];
		bestKernel[0] = 0;
		int[] bestK = { 2, 4, 6, 8, 12, 17, 18, 19 };
		//System.out.println("Best k = " + bestK[0]);
		System.out.println(knnClassifyCore.testSet.length);
		knnClassifyCore.test(bestK, bestKernel);
	}

}
