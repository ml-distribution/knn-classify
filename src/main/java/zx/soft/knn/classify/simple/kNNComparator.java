package zx.soft.knn.classify.simple;

import java.util.Comparator;

/**
 * 
 * @author wanggang
 *
 */
public class kNNComparator implements Comparator<Object> {

	public kNNComparator() {
		//		/
	}

	/**
	 * Compare two kNNSample based on their realtive distance to a query point
	*/
	@Override
	public int compare(Object o1, Object o2) {
		kNNSample example1 = (kNNSample) o1;
		kNNSample example2 = (kNNSample) o2;
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
