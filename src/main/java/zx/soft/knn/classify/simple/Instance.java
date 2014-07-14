package zx.soft.knn.classify.simple;

/**
 * 表示一条数据或者一个实例的数据结构
 *  
 * @author wanggang
 *
 */
public class Instance {

	private int label;

	private double[] attributes;

	public Instance(int label, double[] attributes) {
		this.label = label;
		this.attributes = new double[attributes.length];
		for (int i = 0; i < attributes.length; i++) {
			this.attributes[i] = attributes[i];
		}
	}

	public Instance() {
		//
	}

	public int getLabel() {
		return label;
	}

	public double getAttribute(int index) {
		return attributes[index];
	}

	public double[] getAttributes() {
		return attributes;
	}

	public void setAttributes(double[] attributes) {
		this.attributes = attributes;
	}

}