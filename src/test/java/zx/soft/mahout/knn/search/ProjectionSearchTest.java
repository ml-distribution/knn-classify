package zx.soft.mahout.knn.search;

import org.apache.mahout.common.distance.EuclideanDistanceMeasure;

import zx.soft.mahout.knn.search.ProjectionSearch;
import zx.soft.mahout.knn.search.UpdatableSearcher;

public class ProjectionSearchTest extends FastProjectionSearchTest {

	@Override
	public UpdatableSearcher getSearch(int n) {
		return new ProjectionSearch(new EuclideanDistanceMeasure(), 4, 20);
	}

}
