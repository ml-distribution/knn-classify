package org.apache.mahout.knn.search;

import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Matrix;

import zx.soft.mahout.knn.search.ProjectionSearch;
import zx.soft.mahout.knn.search.UpdatableSearcher;

public class ProjectionSearchTest extends FastProjectionSearchTest {

	private static Matrix data;
	private static final int QUERIES = 20;
	private static final int SEARCH_SIZE = 300;
	private static final int MAX_DEPTH = 100;

	@Override
	public UpdatableSearcher getSearch(int n) {
		return new ProjectionSearch(new EuclideanDistanceMeasure(), 4, 20);
	}

}
