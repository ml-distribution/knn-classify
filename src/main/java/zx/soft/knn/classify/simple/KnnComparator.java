package zx.soft.knn.classify.simple;

import java.util.Comparator;

/**
 * 比较类
 * 
 * @author wanggang
 *
 */
public class KnnComparator implements Comparator<Object> {

	public KnnComparator() {
		//		
	}

	/**
	 * 根据相对距离进行比较
	 */
	@Override
	public int compare(Object o1, Object o2) {
		KnnInstance example1 = (KnnInstance) o1;
		KnnInstance example2 = (KnnInstance) o2;
		double dist1 = example1.getRelativeDist();
		double dist2 = example2.getRelativeDist();
		if (dist1 < dist2)
			return -1;
		else if (dist1 == dist2)
			return 0;
		else
			return 1;
	}

	@Override
	public boolean equals(Object o) {
		return (this.equals(o));
	}

}
