package zx.soft.knn.classify.simple;

/**
 * 适用于KNN算法的数据结构
 * 
 * @author wanggang
 *
 */
public class KnnInstance extends Instance {

	// 两点之间的距离
	private double relativeDistance;
	// 该点的权重
	private double weight;
	// 该点的ID
	private final int id;
	// 特征项的权重，用于计算基于权重的距离
	private static double[] featureWeights;

	public KnnInstance(Instance instance, int id) {
		super(instance.getLabel(), instance.getAttributes());
		this.id = id;
		// 默认权重是1
		this.weight = 1;
	}

	/**
	 * 基于核宽度（kernel width）的方式，设置该实例或者该点的权重。
	 *     如果kernel = 0，非权重的KNN算法；
	 *     如果kernel = 1，基于权重的KNN算法。
	 */
	public void setWeight(int kernel) {
		this.weight = 1.0 / Math.exp(kernel * relativeDistance);
	}

	/**
	 * 当特征权重设置好的话，计算基于权重的距离
	*/
	public void setRelativeDist(Instance example) {
		double sum = 0;
		for (int i = 0; i < example.getAttributes().length; i++) {
			sum = sum + featureWeights[i] * (Math.pow(getAttribute(i) - example.getAttribute(i), 2));
		}
		this.relativeDistance = Math.sqrt(sum);
	}

	/**
	 * 如果两个实例的ID相同，那么实例相同
	 */
	public boolean equals(KnnInstance instance) {
		return (this.id == instance.getId());
	}

	public double getRelativeDist() {
		return relativeDistance;
	}

	public double getWeight() {
		return weight;
	}

	public int getId() {
		return id;
	}

	/**
	 * 设置特征权重
	 */
	public static void setFeatureWeights(double[] featureWeights) {
		int numWeights = featureWeights.length;
		KnnInstance.featureWeights = new double[numWeights];
		for (int i = 0; i < numWeights; i++)
			KnnInstance.featureWeights[i] = featureWeights[i];
	}

	/**
	 * 默认的特征权重
	 */
	public static void setFeatureWeights(int attributeNum) {
		KnnInstance.featureWeights = new double[attributeNum];
		for (int i = 0; i < attributeNum; i++)
			KnnInstance.featureWeights[i] = 1;
	}

}