package zx.soft.mahout.knn.search;

import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;

/**
 * Describes how we search vectors.  A class should extend UpdatableSearch only if it can handle a remove function.
 */
public abstract class UpdatableSearcher extends Searcher {

	public UpdatableSearcher(DistanceMeasure distanceMeasure) {
		super(distanceMeasure);
	}

	@Override
	public abstract boolean remove(Vector v, double epsilon);

	@Override
	public abstract void clear();

}
