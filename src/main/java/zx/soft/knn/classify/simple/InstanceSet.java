package zx.soft.knn.classify.simple;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * KNN算法使用到的数据集 
 * 
 * @author wanggang
 *
 */
public class InstanceSet {

	private int size = 0;
	private int attributeNum = 0;
	private Instance[] instances;

	public InstanceSet(String filename) {
		init(filename);
		instances = new Instance[size];
		for (int i = 0; i < size; i++) {
			instances[i] = new Instance();
		}
		String str;
		String[] strs;
		int count = 0;
		try (BufferedReader br = new BufferedReader(new FileReader(filename));) {
			while ((str = br.readLine()) != null) {
				str = str.trim();
				strs = str.split("\\s");
				double[] attributes = new double[attributeNum];
				for (int i = 0; i < attributeNum; i++) {
					attributes[i] = Double.parseDouble(strs[i]);
				}
				instances[count++].setAttributes(attributes);
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private void init(String filename) {
		String str;
		try (BufferedReader br = new BufferedReader(new FileReader(filename));) {
			while ((str = br.readLine()) != null) {
				str = str.trim();
				size++;
				attributeNum = str.split("\\s").length;
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public int getSize() {
		return size;
	}

	public void setSize(int size) {
		this.size = size;
	}

	public int getAttributeNum() {
		return attributeNum;
	}

	public void setAttributeNum(int attributeNum) {
		this.attributeNum = attributeNum;
	}

	public double getAttribute(int i, int j) {
		return instances[i].getAttribute(j);
	}

	public Instance[] getInstances() {
		return instances;
	}

	public Instance getInstances(int i) {
		return instances[i];
	}

	public void setInstances(Instance[] instances) {
		this.instances = instances;
	}

}