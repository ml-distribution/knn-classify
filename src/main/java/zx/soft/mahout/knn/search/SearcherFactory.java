package zx.soft.mahout.knn.search;

import org.apache.mahout.common.distance.DistanceMeasure;

public interface SearcherFactory {

	public Searcher create(DistanceMeasure distanceMeasure);

}
